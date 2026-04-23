#include <aslam/cameras/apriltag_internal/CalibrationStateBundle.hpp>

#include <cmath>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>

namespace aslam {
namespace cameras {
namespace apriltag_internal {

namespace {

std::string Trim(const std::string& value) {
  const std::size_t begin = value.find_first_not_of(" \t\r\n");
  if (begin == std::string::npos) {
    return "";
  }
  const std::size_t end = value.find_last_not_of(" \t\r\n");
  return value.substr(begin, end - begin + 1);
}

bool ParseKeyValueFile(const std::string& path,
                       std::map<std::string, std::string>* key_values,
                       std::string* error_message) {
  if (key_values == nullptr) {
    if (error_message != nullptr) {
      *error_message = "ParseKeyValueFile requires a valid key_values pointer.";
    }
    return false;
  }
  std::ifstream input(path.c_str());
  if (!input.is_open()) {
    if (error_message != nullptr) {
      *error_message = "Failed to open summary file: " + path;
    }
    return false;
  }

  std::string line;
  while (std::getline(input, line)) {
    const std::size_t separator = line.find(':');
    if (separator == std::string::npos) {
      continue;
    }
    const std::string key = Trim(line.substr(0, separator));
    const std::string value = Trim(line.substr(separator + 1));
    if (!key.empty()) {
      (*key_values)[key] = value;
    }
  }
  return true;
}

int GetIntValue(const std::map<std::string, std::string>& key_values,
                const std::string& key,
                int default_value) {
  const auto it = key_values.find(key);
  if (it == key_values.end() || it->second.empty()) {
    return default_value;
  }
  return std::stoi(it->second);
}

double GetDoubleValue(const std::map<std::string, std::string>& key_values,
                      const std::string& key,
                      double default_value) {
  const auto it = key_values.find(key);
  if (it == key_values.end() || it->second.empty()) {
    return default_value;
  }
  return std::stod(it->second);
}

std::string GetStringValue(const std::map<std::string, std::string>& key_values,
                           const std::string& key,
                           const std::string& default_value) {
  const auto it = key_values.find(key);
  if (it == key_values.end()) {
    return default_value;
  }
  return it->second;
}

}  // namespace

std::string CalibrationStateBundle::SummaryString() const {
  std::ostringstream stream;
  stream << "success: " << (success ? 1 : 0) << "\n";
  stream << "failure_reason: " << failure_reason << "\n";
  stream << "ready_for_backend: " << (ready_for_backend ? 1 : 0) << "\n";
  stream << "bundle_version: " << bundle_version << "\n";
  stream << "baseline_protocol_label: " << baseline_protocol_label << "\n";
  stream << "training_split_signature: " << training_split_signature << "\n";
  stream << "round_index: " << round_index << "\n";
  stream << "reference_board_id: " << scene_state.reference_board_id << "\n";
  stream << "dataset_label: " << scene_state.dataset_label << "\n";
  stream << "source_pipeline_label: " << scene_state.source_pipeline_label << "\n";
  stream << "source_stage_label: " << measurement_dataset.source_stage_label << "\n";
  stream << "camera_model: " << scene_state.camera_model << "\n";
  stream << "scene_state_level: " << scene_state.coarse_or_optimized_level << "\n";
  stream << "accepted_frame_count: " << measurement_dataset.accepted_frame_count << "\n";
  stream << "accepted_board_observation_count: "
         << measurement_dataset.accepted_board_observation_count << "\n";
  stream << "accepted_outer_point_count: " << measurement_dataset.accepted_outer_point_count << "\n";
  stream << "accepted_internal_point_count: "
         << measurement_dataset.accepted_internal_point_count << "\n";
  stream << "accepted_total_point_count: " << measurement_dataset.accepted_total_point_count << "\n";
  stream << "overall_rmse: " << residual_result.overall_rmse << "\n";
  stream << "outer_only_rmse: " << residual_result.outer_only_rmse << "\n";
  stream << "internal_only_rmse: " << residual_result.internal_only_rmse << "\n";
  stream << "camera_xi: " << scene_state.camera.xi << "\n";
  stream << "camera_alpha: " << scene_state.camera.alpha << "\n";
  stream << "camera_fu: " << scene_state.camera.fu << "\n";
  stream << "camera_fv: " << scene_state.camera.fv << "\n";
  stream << "camera_cu: " << scene_state.camera.cu << "\n";
  stream << "camera_cv: " << scene_state.camera.cv << "\n";
  stream << "camera_resolution_width: " << scene_state.camera.resolution.width << "\n";
  stream << "camera_resolution_height: " << scene_state.camera.resolution.height << "\n";
  return stream.str();
}

CalibrationSceneState BuildCalibrationSceneState(
    const JointReprojectionSceneState& scene_state,
    const std::string& level,
    const CalibrationBundleMetadata& metadata) {
  CalibrationSceneState result;
  result.reference_board_id = scene_state.reference_board_id;
  result.camera_model = "ds";
  result.coarse_or_optimized_level = level;
  result.bundle_version = metadata.bundle_version;
  result.baseline_protocol_label = metadata.baseline_protocol_label;
  result.training_split_signature = metadata.training_split_signature;
  result.camera = scene_state.camera;
  result.boards = scene_state.boards;
  result.frames = scene_state.frames;
  result.dataset_label = metadata.dataset_label;
  result.source_pipeline_label = metadata.source_pipeline_label;
  result.warnings = scene_state.warnings;
  return result;
}

CalibrationMeasurementDataset BuildCalibrationMeasurementDataset(
    const JointMeasurementSelectionResult& selection_result,
    const JointMeasurementBuildResult& measurement_result,
    const std::string& source_stage_label,
    const CalibrationBundleMetadata& metadata) {
  CalibrationMeasurementDataset result;
  result.reference_board_id = selection_result.reference_board_id;
  result.bundle_version = metadata.bundle_version;
  result.baseline_protocol_label = metadata.baseline_protocol_label;
  result.training_split_signature = metadata.training_split_signature;
  result.frames = selection_result.selected_measurement_result.frames;
  result.solver_observations = selection_result.selected_measurement_result.solver_observations;
  result.accepted_frame_indices = selection_result.accepted_frame_indices;
  result.accepted_board_observation_keys = selection_result.accepted_board_observation_keys;
  result.accepted_frame_count = selection_result.accepted_frame_count;
  result.accepted_board_observation_count = selection_result.accepted_board_observation_count;
  result.accepted_outer_point_count = selection_result.accepted_outer_point_count;
  result.accepted_internal_point_count = selection_result.accepted_internal_point_count;
  result.accepted_total_point_count = selection_result.selected_measurement_result.used_total_point_count;
  result.dataset_label = metadata.dataset_label;
  result.source_stage_label = source_stage_label;
  result.warnings = selection_result.warnings;
  result.failure_reason = selection_result.failure_reason;

  if (result.accepted_total_point_count == 0) {
    result.accepted_total_point_count = measurement_result.used_total_point_count;
  }
  return result;
}

CalibrationStateBundle BuildCalibrationStateBundleFromJointOptimizationResult(
    const JointOptimizationResult& optimization_result,
    const JointMeasurementSelectionResult& selection_result,
    const JointMeasurementBuildResult& measurement_result,
    int round_index,
    const std::string& dataset_label) {
  CalibrationBundleMetadata metadata;
  metadata.dataset_label = dataset_label;
  return BuildCalibrationStateBundleFromJointOptimizationResult(
      optimization_result, selection_result, measurement_result, round_index, metadata);
}

CalibrationStateBundle BuildCalibrationStateBundleFromJointOptimizationResult(
    const JointOptimizationResult& optimization_result,
    const JointMeasurementSelectionResult& selection_result,
    const JointMeasurementBuildResult& measurement_result,
    int round_index,
    const CalibrationBundleMetadata& metadata) {
  CalibrationStateBundle bundle;
  bundle.round_index = round_index;
  bundle.bundle_version = metadata.bundle_version;
  bundle.baseline_protocol_label = metadata.baseline_protocol_label;
  bundle.training_split_signature = metadata.training_split_signature;

  if (!optimization_result.success) {
    bundle.failure_reason = "Joint optimization result is not successful.";
    return bundle;
  }
  if (!selection_result.success) {
    bundle.failure_reason = "Joint measurement selection result is not successful.";
    return bundle;
  }
  if (!measurement_result.success) {
    bundle.failure_reason = "Joint measurement build result is not successful.";
    return bundle;
  }
  if (!optimization_result.optimized_residual.success) {
    bundle.failure_reason = "Optimized residual result is not successful.";
    return bundle;
  }
  if (optimization_result.reference_board_id != selection_result.reference_board_id ||
      optimization_result.reference_board_id != measurement_result.reference_board_id ||
      optimization_result.reference_board_id != optimization_result.optimized_residual.reference_board_id ||
      optimization_result.reference_board_id != optimization_result.optimized_state.reference_board_id) {
    bundle.failure_reason = "Reference board id mismatch across Stage 5 bundle inputs.";
    return bundle;
  }

  std::ostringstream stage_label;
  stage_label << "round" << round_index << "_optimized";
  bundle.scene_state = BuildCalibrationSceneState(
      optimization_result.optimized_state,
      stage_label.str(),
      metadata);
  bundle.measurement_dataset = BuildCalibrationMeasurementDataset(
      selection_result, measurement_result, stage_label.str(), metadata);
  bundle.residual_result = optimization_result.optimized_residual;
  bundle.selection_result = selection_result;
  bundle.success = true;
  bundle.ready_for_backend =
      bundle.scene_state.IsValid() &&
      bundle.measurement_dataset.accepted_frame_count > 0 &&
      bundle.measurement_dataset.accepted_board_observation_count > 0 &&
      bundle.measurement_dataset.accepted_total_point_count > 0;
  bundle.warnings = optimization_result.warnings;
  bundle.warnings.insert(bundle.warnings.end(),
                         selection_result.warnings.begin(),
                         selection_result.warnings.end());
  if (!bundle.ready_for_backend) {
    bundle.failure_reason = "Stage 5 bundle is missing accepted data or valid camera state.";
    bundle.success = false;
  }
  return bundle;
}

CalibrationBackendProblemInput BuildBackendProblemInput(
    const CalibrationStateBundle& bundle,
    const BackendProblemOptions& options) {
  if (!bundle.IsReadyForBackend()) {
    throw std::runtime_error("BuildBackendProblemInput requires a ready-for-backend bundle.");
  }
  CalibrationBackendProblemInput result;
  result.reference_board_id = bundle.scene_state.reference_board_id;
  result.bundle_version = bundle.bundle_version;
  result.baseline_protocol_label = bundle.baseline_protocol_label;
  result.training_split_signature = bundle.training_split_signature;
  result.dataset_label = bundle.scene_state.dataset_label;
  result.scene_state = bundle.scene_state;
  result.measurement_dataset = bundle.measurement_dataset;
  result.residual_result = bundle.residual_result;
  result.parameterization.camera_model = bundle.scene_state.camera_model;
  result.parameterization.pose_parameterization = "se3";
  result.parameterization.reference_board_fixed = true;
  result.optimization_masks.optimize_frame_poses = options.optimize_frame_poses;
  result.optimization_masks.optimize_board_poses = options.optimize_board_poses;
  result.optimization_masks.optimize_intrinsics = options.optimize_intrinsics;
  result.optimization_masks.delayed_intrinsics_release = options.delayed_intrinsics_release;
  result.optimization_masks.intrinsics_release_iteration = options.intrinsics_release_iteration;
  result.priors.use_intrinsics_anchor_prior = true;
  result.priors.intrinsics_anchor_weight_xi_alpha = options.intrinsics_anchor_weight_xi_alpha;
  result.priors.intrinsics_anchor_weight_focal = options.intrinsics_anchor_weight_focal;
  result.priors.intrinsics_anchor_weight_principal = options.intrinsics_anchor_weight_principal;
  result.diagnostics_seed.overall_rmse = bundle.residual_result.overall_rmse;
  result.diagnostics_seed.outer_only_rmse = bundle.residual_result.outer_only_rmse;
  result.diagnostics_seed.internal_only_rmse = bundle.residual_result.internal_only_rmse;
  result.diagnostics_seed.accepted_total_point_count =
      bundle.measurement_dataset.accepted_total_point_count;
  result.diagnostics_seed.warnings = bundle.warnings;
  return result;
}

JointReprojectionSceneState BuildJointSceneStateFromCalibrationSceneState(
    const CalibrationSceneState& scene_state) {
  JointReprojectionSceneState result;
  result.reference_board_id = scene_state.reference_board_id;
  result.camera = scene_state.camera;
  result.boards = scene_state.boards;
  result.frames = scene_state.frames;
  result.warnings = scene_state.warnings;
  return result;
}

void WriteCalibrationStateBundleSummary(const std::string& path,
                                        const CalibrationStateBundle& bundle) {
  std::ofstream output(path.c_str());
  output << bundle.SummaryString();
  for (const std::string& warning : bundle.warnings) {
    output << "warning: " << warning << "\n";
  }
}

void WriteCalibrationBackendProblemSummary(
    const std::string& path,
    const CalibrationBackendProblemInput& backend_problem_input) {
  std::ofstream output(path.c_str());
  output << "reference_board_id: " << backend_problem_input.reference_board_id << "\n";
  output << "bundle_version: " << backend_problem_input.bundle_version << "\n";
  output << "baseline_protocol_label: " << backend_problem_input.baseline_protocol_label << "\n";
  output << "training_split_signature: " << backend_problem_input.training_split_signature << "\n";
  output << "dataset_label: " << backend_problem_input.dataset_label << "\n";
  output << "camera_model: " << backend_problem_input.scene_state.camera_model << "\n";
  output << "scene_state_level: "
         << backend_problem_input.scene_state.coarse_or_optimized_level << "\n";
  output << "accepted_frame_count: "
         << backend_problem_input.measurement_dataset.accepted_frame_count << "\n";
  output << "accepted_board_observation_count: "
         << backend_problem_input.measurement_dataset.accepted_board_observation_count << "\n";
  output << "accepted_total_point_count: "
         << backend_problem_input.measurement_dataset.accepted_total_point_count << "\n";
  output << "overall_rmse_seed: " << backend_problem_input.diagnostics_seed.overall_rmse << "\n";
  output << "outer_only_rmse_seed: "
         << backend_problem_input.diagnostics_seed.outer_only_rmse << "\n";
  output << "internal_only_rmse_seed: "
         << backend_problem_input.diagnostics_seed.internal_only_rmse << "\n";
  output << "optimize_frame_poses: "
         << (backend_problem_input.optimization_masks.optimize_frame_poses ? 1 : 0) << "\n";
  output << "optimize_board_poses: "
         << (backend_problem_input.optimization_masks.optimize_board_poses ? 1 : 0) << "\n";
  output << "optimize_intrinsics: "
         << (backend_problem_input.optimization_masks.optimize_intrinsics ? 1 : 0) << "\n";
  output << "delayed_intrinsics_release: "
         << (backend_problem_input.optimization_masks.delayed_intrinsics_release ? 1 : 0)
         << "\n";
  output << "intrinsics_release_iteration: "
         << backend_problem_input.optimization_masks.intrinsics_release_iteration << "\n";
}

bool LoadCalibrationStateBundleSummary(const std::string& path,
                                       CalibrationStateBundle* bundle,
                                       std::string* error_message) {
  if (bundle == nullptr) {
    if (error_message != nullptr) {
      *error_message = "LoadCalibrationStateBundleSummary requires a valid bundle pointer.";
    }
    return false;
  }

  std::map<std::string, std::string> key_values;
  if (!ParseKeyValueFile(path, &key_values, error_message)) {
    return false;
  }

  bundle->success = GetIntValue(key_values, "success", 0) != 0;
  bundle->ready_for_backend = GetIntValue(key_values, "ready_for_backend", 0) != 0;
  bundle->bundle_version = GetStringValue(key_values, "bundle_version", "stage5_bundle_v1");
  bundle->baseline_protocol_label =
      GetStringValue(key_values, "baseline_protocol_label", "frozen_round2_v1");
  bundle->training_split_signature =
      GetStringValue(key_values, "training_split_signature", "all_frames");
  bundle->round_index = GetIntValue(key_values, "round_index", -1);
  bundle->failure_reason = GetStringValue(key_values, "failure_reason", "");
  bundle->scene_state.reference_board_id = GetIntValue(key_values, "reference_board_id", 1);
  bundle->scene_state.bundle_version = bundle->bundle_version;
  bundle->scene_state.baseline_protocol_label = bundle->baseline_protocol_label;
  bundle->scene_state.training_split_signature = bundle->training_split_signature;
  bundle->scene_state.dataset_label = GetStringValue(key_values, "dataset_label", "");
  bundle->scene_state.source_pipeline_label =
      GetStringValue(key_values, "source_pipeline_label", "");
  bundle->scene_state.camera_model = GetStringValue(key_values, "camera_model", "ds");
  bundle->scene_state.coarse_or_optimized_level =
      GetStringValue(key_values, "scene_state_level", "");
  bundle->scene_state.camera.xi = GetDoubleValue(key_values, "camera_xi", 0.0);
  bundle->scene_state.camera.alpha = GetDoubleValue(key_values, "camera_alpha", 0.0);
  bundle->scene_state.camera.fu = GetDoubleValue(key_values, "camera_fu", 0.0);
  bundle->scene_state.camera.fv = GetDoubleValue(key_values, "camera_fv", 0.0);
  bundle->scene_state.camera.cu = GetDoubleValue(key_values, "camera_cu", 0.0);
  bundle->scene_state.camera.cv = GetDoubleValue(key_values, "camera_cv", 0.0);
  bundle->scene_state.camera.resolution = cv::Size(
      GetIntValue(key_values, "camera_resolution_width", 0),
      GetIntValue(key_values, "camera_resolution_height", 0));
  bundle->measurement_dataset.reference_board_id = bundle->scene_state.reference_board_id;
  bundle->measurement_dataset.bundle_version = bundle->bundle_version;
  bundle->measurement_dataset.baseline_protocol_label = bundle->baseline_protocol_label;
  bundle->measurement_dataset.training_split_signature = bundle->training_split_signature;
  bundle->measurement_dataset.dataset_label = bundle->scene_state.dataset_label;
  bundle->measurement_dataset.source_stage_label =
      GetStringValue(key_values, "source_stage_label", "");
  bundle->measurement_dataset.accepted_frame_count =
      GetIntValue(key_values, "accepted_frame_count", 0);
  bundle->measurement_dataset.accepted_board_observation_count =
      GetIntValue(key_values, "accepted_board_observation_count", 0);
  bundle->measurement_dataset.accepted_outer_point_count =
      GetIntValue(key_values, "accepted_outer_point_count", 0);
  bundle->measurement_dataset.accepted_internal_point_count =
      GetIntValue(key_values, "accepted_internal_point_count", 0);
  bundle->measurement_dataset.accepted_total_point_count =
      GetIntValue(key_values, "accepted_total_point_count", 0);
  bundle->residual_result.success = bundle->success;
  bundle->residual_result.reference_board_id = bundle->scene_state.reference_board_id;
  bundle->residual_result.overall_rmse = GetDoubleValue(key_values, "overall_rmse", 0.0);
  bundle->residual_result.outer_only_rmse = GetDoubleValue(key_values, "outer_only_rmse", 0.0);
  bundle->residual_result.internal_only_rmse =
      GetDoubleValue(key_values, "internal_only_rmse", 0.0);
  bundle->selection_result.success = bundle->success;
  bundle->selection_result.reference_board_id = bundle->scene_state.reference_board_id;
  bundle->selection_result.accepted_frame_count = bundle->measurement_dataset.accepted_frame_count;
  bundle->selection_result.accepted_board_observation_count =
      bundle->measurement_dataset.accepted_board_observation_count;
  bundle->selection_result.accepted_outer_point_count =
      bundle->measurement_dataset.accepted_outer_point_count;
  bundle->selection_result.accepted_internal_point_count =
      bundle->measurement_dataset.accepted_internal_point_count;
  return true;
}

}  // namespace apriltag_internal
}  // namespace cameras
}  // namespace aslam
