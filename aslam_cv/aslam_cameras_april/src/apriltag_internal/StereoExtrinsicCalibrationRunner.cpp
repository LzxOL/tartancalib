#include <aslam/cameras/apriltag_internal/StereoExtrinsicCalibrationRunner.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <numeric>
#include <queue>
#include <cctype>
#include <set>
#include <sstream>
#include <stdexcept>
#include <utility>

#include <Eigen/SVD>

#include <opencv2/calib3d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <boost/filesystem.hpp>
#include <boost/make_shared.hpp>
#include <boost/shared_ptr.hpp>

#include <aslam/backend/CameraDesignVariable.hpp>
#include <aslam/backend/DesignVariable.hpp>
#include <aslam/backend/ErrorTerm.hpp>
#include <aslam/backend/ErrorTermDs.hpp>
#include <aslam/backend/ErrorTermTransformation.hpp>
#include <aslam/backend/HomogeneousExpression.hpp>
#include <aslam/backend/MapTransformation.hpp>
#include <aslam/backend/MEstimatorPolicies.hpp>
#include <aslam/backend/MappedEuclideanPoint.hpp>
#include <aslam/backend/MappedRotationQuaternion.hpp>
#include <aslam/backend/OptimizationProblem.hpp>
#include <aslam/backend/Optimizer.hpp>
#include <aslam/backend/TransformationExpression.hpp>
#include <aslam/calibration/core/IncrementalEstimator.h>
#include <aslam/calibration/core/LinearSolverOptions.h>
#include <aslam/calibration/core/OptimizationProblem.h>
#include <aslam/cameras/apriltag_internal/DoubleSphereCameraModel.hpp>
#include <aslam/cameras/apriltag_internal/AngularResidualGeometry.hpp>
#include <aslam/cameras/apriltag_internal/JointReprojectionCostCore.hpp>
#include <aslam/cameras/apriltag_internal/MultiBoardCoObservationConsistency.hpp>
#include <aslam/cameras/apriltag_internal/Stage6IncrementalBatchEstimator.hpp>
#include <aslam/cameras/apriltag_internal/StereoResidualEvaluator.hpp>
#include <aslam/cameras.hpp>
#include <sm/kinematics/Transformation.hpp>

namespace aslam {
namespace cameras {
namespace apriltag_internal {
namespace {

using Clock = std::chrono::steady_clock;
using PairBoardKey = std::pair<int, int>;
using DsGeometry = aslam::cameras::DoubleSphereCameraGeometry;
using DsProjection =
    aslam::cameras::DoubleSphereProjection<aslam::cameras::NoDistortion>;
using EucmGeometry = aslam::cameras::ExtendedUnifiedCameraGeometry;
using EucmProjection =
    aslam::cameras::ExtendedUnifiedProjection<aslam::cameras::NoDistortion>;
using PinholeEquiGeometry =
    aslam::cameras::EquidistantDistortedPinholeCameraGeometry;
using PinholeEquiProjection =
    aslam::cameras::PinholeProjection<aslam::cameras::EquidistantDistortion>;
using OmniGeometry = aslam::cameras::OmniCameraGeometry;
using OmniProjection =
    aslam::cameras::OmniProjection<aslam::cameras::NoDistortion>;
using OmniRadtanGeometry = aslam::cameras::DistortedOmniCameraGeometry;
using OmniRadtanProjection =
    aslam::cameras::OmniProjection<
        aslam::cameras::RadialTangentialDistortion>;
namespace fs = boost::filesystem;
using CalibrationBatch = aslam::calibration::OptimizationProblem;

constexpr std::size_t kStage6StereoExtrinsicInformationGroupId = 0;
constexpr std::size_t kStage6BoardLayoutGroupId = 1;
constexpr std::size_t kStage6PairPoseGroupId = 2;

const char* StereoIntrinsicsModeToString(StereoIntrinsicsMode mode) {
  switch (mode) {
    case StereoIntrinsicsMode::FixedStage5:
      return "fixed_stage5";
    case StereoIntrinsicsMode::KalibrJointProjection:
      return "kalibr_joint_projection";
    case StereoIntrinsicsMode::RegularizedJointProjection:
      return "regularized_joint_projection";
    case StereoIntrinsicsMode::AdaptiveRegularizedJointProjection:
      return "adaptive_regularized_joint_projection";
  }
  return "unknown";
}

struct StereoIntrinsicsPolicyDecision {
  StereoIntrinsicsMode requested_mode = StereoIntrinsicsMode::FixedStage5;
  StereoIntrinsicsMode effective_mode = StereoIntrinsicsMode::FixedStage5;
  bool projection_active = false;
  bool projection_prior_enabled = false;
  bool distortion_active = false;
  bool distortion_prior_enabled = false;
  int training_pair_count = 0;
  int shared_pair_board_count = 0;
  int distinct_board_count = 0;
  int observation_point_count = 0;
  std::string reason;
};

StereoIntrinsicsPolicyDecision EvaluateStereoIntrinsicsPolicy(
    const StereoMeasurementDataset& dataset,
    const StereoExtrinsicSolverOptions& options) {
  StereoIntrinsicsPolicyDecision decision;
  decision.requested_mode = options.intrinsics_mode;
  decision.effective_mode = options.intrinsics_mode;
  const std::set<int> training_pairs(dataset.training_pair_indices.begin(),
                                     dataset.training_pair_indices.end());
  decision.training_pair_count = static_cast<int>(training_pairs.size());
  std::set<int> distinct_boards;
  std::set<PairBoardKey> shared_pair_boards;
  for (int pair_index : training_pairs) {
    const auto shared_it = dataset.pair_shared_board_ids.find(pair_index);
    if (shared_it == dataset.pair_shared_board_ids.end()) {
      continue;
    }
    for (int board_id : shared_it->second) {
      shared_pair_boards.insert(PairBoardKey(pair_index, board_id));
      distinct_boards.insert(board_id);
    }
  }
  decision.shared_pair_board_count =
      static_cast<int>(shared_pair_boards.size());
  decision.distinct_board_count = static_cast<int>(distinct_boards.size());
  for (const StereoObservation& observation : dataset.observations) {
    if (!observation.used_in_solver ||
        training_pairs.count(observation.pair_index) == 0 ||
        shared_pair_boards.count(
            PairBoardKey(observation.pair_index, observation.board_id)) == 0) {
      continue;
    }
    ++decision.observation_point_count;
  }

  if (options.intrinsics_mode == StereoIntrinsicsMode::FixedStage5) {
    decision.projection_active = false;
    decision.reason = "requested_fixed_stage5";
    return decision;
  }
  if (options.intrinsics_mode ==
      StereoIntrinsicsMode::KalibrJointProjection) {
    decision.projection_active = true;
    decision.distortion_active = true;
    decision.reason = "requested_unregularized_kalibr_joint_projection";
    return decision;
  }
  if (options.intrinsics_mode ==
      StereoIntrinsicsMode::RegularizedJointProjection) {
    decision.projection_active = true;
    decision.projection_prior_enabled = true;
    decision.distortion_active = true;
    decision.distortion_prior_enabled = true;
    decision.reason = "requested_regularized_joint_projection";
    return decision;
  }

  const bool enough_pairs =
      decision.training_pair_count >=
      std::max(1, options.adaptive_joint_projection_min_training_pairs);
  const bool enough_pair_boards =
      decision.shared_pair_board_count >=
      std::max(1, options.adaptive_joint_projection_min_shared_pair_boards);
  const bool enough_boards =
      decision.distinct_board_count >=
      std::max(1, options.adaptive_joint_projection_min_distinct_boards);
  const bool enough_points =
      decision.observation_point_count >=
      std::max(1, options.adaptive_joint_projection_min_observation_points);
  if (enough_pairs && enough_pair_boards && enough_boards && enough_points) {
    decision.effective_mode = StereoIntrinsicsMode::RegularizedJointProjection;
    decision.projection_active = true;
    decision.projection_prior_enabled = true;
    decision.distortion_active = true;
    decision.distortion_prior_enabled = true;
    decision.reason = "adaptive_data_sufficiency_passed";
  } else {
    decision.effective_mode = StereoIntrinsicsMode::FixedStage5;
    decision.projection_active = false;
    std::ostringstream reason;
    reason << "adaptive_data_sufficiency_failed"
           << " pairs=" << decision.training_pair_count << "/"
           << options.adaptive_joint_projection_min_training_pairs
           << " pair_boards=" << decision.shared_pair_board_count << "/"
           << options.adaptive_joint_projection_min_shared_pair_boards
           << " boards=" << decision.distinct_board_count << "/"
           << options.adaptive_joint_projection_min_distinct_boards
           << " points=" << decision.observation_point_count << "/"
           << options.adaptive_joint_projection_min_observation_points;
    decision.reason = reason.str();
  }
  return decision;
}

struct StereoCandidateTraversalBudget {
  StereoCandidateBudgetMode mode = StereoCandidateBudgetMode::Fixed;
  int traversal_limit = 0;
  int runtime_safety_ceiling = 0;
  std::string max_candidate_additions_effective;
};

struct BearingWhiteningStatsAccumulator {
  int count = 0;
  double sigma_sum = 0.0;
  double sigma_min = std::numeric_limits<double>::infinity();
  double sigma_max = 0.0;
  double weight_sum = 0.0;
  double weight_min = std::numeric_limits<double>::infinity();
  double weight_max = 0.0;

  void Add(const BearingCovarianceResult& result) {
    if (!result.success) {
      return;
    }
    ++count;
    sigma_sum += result.tangent_sigma_mean_rad;
    sigma_min = std::min(sigma_min, result.tangent_sigma_min_rad);
    sigma_max = std::max(sigma_max, result.tangent_sigma_max_rad);
    weight_sum += result.whitening_weight_mean;
    weight_min = std::min(weight_min, result.whitening_weight_min);
    weight_max = std::max(weight_max, result.whitening_weight_max);
  }

  void WriteTo(StereoGlobalSparseBaSummary* summary) const {
    if (summary == nullptr || count <= 0) {
      return;
    }
    summary->spherical_tangent_sigma_mean_rad =
        sigma_sum / static_cast<double>(count);
    summary->spherical_tangent_sigma_min_rad = sigma_min;
    summary->spherical_tangent_sigma_max_rad = sigma_max;
    summary->spherical_whitening_weight_mean =
        weight_sum / static_cast<double>(count);
    summary->spherical_whitening_weight_min = weight_min;
    summary->spherical_whitening_weight_max = weight_max;
  }
};

StereoCandidateTraversalBudget ComputeStereoCandidateTraversalBudget(
    StereoCandidateBudgetMode mode,
    int valid_candidate_count,
    int fixed_max_candidate_additions,
    double adaptive_budget_ratio,
    int adaptive_budget_min,
    int adaptive_budget_max,
    int runtime_safety_ceiling) {
  StereoCandidateTraversalBudget budget;
  budget.mode = mode;
  budget.runtime_safety_ceiling = std::max(0, runtime_safety_ceiling);
  const int valid_count = std::max(0, valid_candidate_count);
  if (mode == StereoCandidateBudgetMode::KalibrStyle) {
    budget.traversal_limit =
        budget.runtime_safety_ceiling > 0
            ? budget.runtime_safety_ceiling
            : std::numeric_limits<int>::max();
    budget.max_candidate_additions_effective =
        "ignored_in_kalibr_style_batch";
    return budget;
  }
  if (mode == StereoCandidateBudgetMode::Adaptive) {
    const int min_budget = std::max(0, adaptive_budget_min);
    const int max_budget =
        adaptive_budget_max > 0
            ? std::max(min_budget, adaptive_budget_max)
            : std::numeric_limits<int>::max();
    int effective_budget = static_cast<int>(
        std::ceil(static_cast<double>(valid_count) *
                  std::max(0.0, adaptive_budget_ratio)));
    effective_budget = std::max(effective_budget, min_budget);
    effective_budget = std::min(effective_budget, max_budget);
    budget.traversal_limit = std::max(0, effective_budget);
  } else {
    budget.traversal_limit = std::max(0, fixed_max_candidate_additions);
  }
  budget.max_candidate_additions_effective =
      std::to_string(budget.traversal_limit);
  return budget;
}

template <typename T>
bool VectorsEqual(const std::vector<T>& lhs, const std::vector<T>& rhs) {
  return lhs == rhs;
}

std::string FormatDoubleVector(const std::vector<double>& values) {
  std::ostringstream stream;
  stream << "[";
  for (size_t i = 0; i < values.size(); ++i) {
    if (i > 0) {
      stream << ", ";
    }
    stream << values[i];
  }
  stream << "]";
  return stream.str();
}

std::string FormatIntVector(const std::vector<int>& values) {
  std::ostringstream stream;
  stream << "[";
  for (size_t i = 0; i < values.size(); ++i) {
    if (i > 0) {
      stream << ", ";
    }
    stream << values[i];
  }
  stream << "]";
  return stream.str();
}

std::string TrimCopy(std::string value) {
  const size_t first = value.find_first_not_of(" \t");
  if (first == std::string::npos) {
    return "";
  }
  const size_t last = value.find_last_not_of(" \t");
  return value.substr(first, last - first + 1);
}

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

bool EstimateCameraBoardPoseWithRmse(const StereoMeasurementDataset& dataset,
                                     const StereoCameraFixedCalibration& calibration,
                                     int pair_index,
                                     int camera_index,
                                     int board_id,
                                     Eigen::Isometry3d* T_camera_board,
                                     double* rmse);
int CountOuterObservationsForCamera(const StereoMeasurementDataset& dataset,
                                    int pair_index,
                                    int camera_index,
                                    int board_id);

double SharedBoardOuterRmseThreshold(
    const StereoExtrinsicSolverOptions& options);

IntermediateCameraConfig MakeCameraConfig(
    const StereoCameraFixedCalibration& calibration) {
  IntermediateCameraConfig config;
  config.camera_model = calibration.camera_model;
  config.distortion_model = calibration.distortion_model;
  config.intrinsics = calibration.intrinsics;
  config.distortion_coeffs = calibration.distortion_coeffs;
  config.resolution = calibration.resolution;
  return config;
}

OuterBootstrapCameraIntrinsics MakeOuterBootstrapCameraIntrinsics(
    const StereoCameraFixedCalibration& calibration) {
  OuterBootstrapCameraIntrinsics intrinsics;
  intrinsics.camera_model = calibration.camera_model;
  intrinsics.distortion_model = calibration.distortion_model;
  intrinsics.SetIntrinsicsVector(calibration.intrinsics);
  intrinsics.SetDistortionVector(calibration.distortion_coeffs);
  if (calibration.resolution.size() == 2) {
    intrinsics.resolution =
        cv::Size(calibration.resolution[0], calibration.resolution[1]);
  }
  return intrinsics;
}

template <typename GeometryT>
IntermediateCameraConfig MakeCameraConfigFromGeometry(
    const GeometryT& geometry,
    const StereoCameraFixedCalibration& seed_calibration);

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

double RotationDistanceRadians(const Eigen::Isometry3d& lhs,
                               const Eigen::Isometry3d& rhs) {
  const Eigen::Matrix3d delta = lhs.linear() * rhs.linear().transpose();
  const double trace = std::max(-1.0, std::min(3.0, delta.trace()));
  const double cosine = std::max(-1.0, std::min(1.0, 0.5 * (trace - 1.0)));
  return std::acos(cosine);
}

double TransformDistanceScore(const Eigen::Isometry3d& lhs,
                              const Eigen::Isometry3d& rhs) {
  return RotationDistanceRadians(lhs, rhs) +
         (lhs.translation() - rhs.translation()).norm();
}

struct TransformCandidateWithMeta {
  int pair_index = -1;
  int board_id = -1;
  Eigen::Isometry3d transform = Eigen::Isometry3d::Identity();
  double score = 0.0;
};

struct PairBoardGraphEdge {
  int pair_index = -1;
  int board_id = -1;
  bool cam0_valid = false;
  bool cam1_valid = false;
  bool shared_stereo = false;
  int cam0_outer_point_count = 0;
  int cam1_outer_point_count = 0;
  double cam0_rmse = std::numeric_limits<double>::infinity();
  double cam1_rmse = std::numeric_limits<double>::infinity();
  Eigen::Isometry3d T_cam0_board = Eigen::Isometry3d::Identity();
  Eigen::Isometry3d T_cam1_board = Eigen::Isometry3d::Identity();
};

struct BootstrapTransformCandidate {
  Eigen::Isometry3d transform = Eigen::Isometry3d::Identity();
  bool is_shared_stereo = false;
  int outer_point_count = 0;
  double pose_fit_rmse = std::numeric_limits<double>::infinity();
  int source_pair_index = -1;
  int source_board_id = -1;
  int source_camera_index = -1;
};

struct RuntimeCounters {
  int symmetric_refit_call_count = 0;
  int symmetric_refit_improved_count = 0;
  int symmetric_refit_fallback_count = 0;
  int graph_propagation_iteration_count = 0;
  int runtime_guard_trigger_count = 0;
};

template <typename GeometryT>
using CameraDv = aslam::backend::CameraDesignVariable<GeometryT>;

struct StereoPoseVariableState {
  sm::kinematics::Transformation transform;
  boost::shared_ptr<aslam::backend::MappedRotationQuaternion> rotation_dv;
  boost::shared_ptr<aslam::backend::MappedEuclideanPoint> translation_dv;
  aslam::backend::TransformationExpression expression;
};

using StereoPoseVariableMap =
    std::map<int, StereoPoseVariableState, std::less<int>,
             Eigen::aligned_allocator<
                 std::pair<const int, StereoPoseVariableState> > >;
using StereoPoseMatrixMap =
    std::map<int, Eigen::Matrix4d, std::less<int>,
             Eigen::aligned_allocator<
                 std::pair<const int, Eigen::Matrix4d> > >;

StereoPoseVariableState& GetOrCreateStereoPoseVariable(
    StereoPoseVariableMap* variables,
    int id) {
  auto it = variables->find(id);
  if (it != variables->end()) {
    return it->second;
  }
  return variables
      ->emplace(std::piecewise_construct, std::forward_as_tuple(id),
                std::forward_as_tuple())
      .first->second;
}

void InitializeStereoPoseVariable(StereoPoseVariableState* variable,
                                  const Eigen::Matrix4d& transform_matrix,
                                  bool active) {
  if (variable == nullptr) {
    return;
  }
  variable->transform = sm::kinematics::Transformation(transform_matrix);
  variable->expression = aslam::backend::transformationToExpression(
      variable->transform, variable->rotation_dv, variable->translation_dv);
  variable->rotation_dv->setActive(active);
  variable->translation_dv->setActive(active);
}

void AddStereoPoseVariableDvs(
    const StereoPoseVariableState& variable,
    std::size_t group_id,
    const boost::shared_ptr<CalibrationBatch>& batch) {
  batch->addDesignVariable(variable.rotation_dv, group_id);
  batch->addDesignVariable(variable.translation_dv, group_id);
}

void SetStereoPoseVariableFromMatrix(StereoPoseVariableState* variable,
                                     const Eigen::Matrix4d& matrix) {
  if (variable == nullptr) {
    return;
  }
  variable->transform.set(matrix);
  variable->rotation_dv->set(variable->transform.q());
  variable->translation_dv->set(variable->transform.t());
}

bool IsStereoPoseVariableFinite(const StereoPoseVariableState& variable) {
  const Eigen::Vector4d& q = variable.transform.q();
  const Eigen::Vector3d& t = variable.transform.t();
  return q.allFinite() && t.allFinite() &&
         std::abs(q.norm() - 1.0) < 1e-3;
}

int CountStereoObservationPointsForKeys(
    const StereoMeasurementDataset& dataset,
    const std::set<PairBoardKey>& keys) {
  int count = 0;
  for (const StereoObservation& observation : dataset.observations) {
    if (observation.used_in_solver &&
        keys.count(PairBoardKey(observation.pair_index,
                                observation.board_id)) > 0) {
      ++count;
    }
  }
  return count;
}

struct StereoViewSelectionScore {
  int shared_board_count = 0;
  int shared_outer_point_count = 0;
  double pose_fit_rmse = std::numeric_limits<double>::infinity();
  int single_camera_only_board_count = 0;
};

using CoObsLocalPoseKey = std::tuple<int, int, int>;

bool PairBoardSelected(const StereoPairSelectionSummary& selection_summary,
                       int pair_index,
                       int board_id) {
  return selection_summary.selected_pair_board_keys.empty() ||
         selection_summary.selected_pair_board_keys.count(
             PairBoardKey(pair_index, board_id)) > 0;
}

struct CoObsLocalPoseMeasurement {
  int pair_index = -1;
  int camera_index = -1;
  int board_id = -1;
  bool success = false;
  int corner_count = 0;
  double local_rmse = std::numeric_limits<double>::infinity();
  Eigen::Isometry3d T_camera_board = Eigen::Isometry3d::Identity();
};

struct CoObsFactorBuildStats {
  int stereo_factor_count = 0;
  int layout_factor_count = 0;
  int local_pose_attempt_count = 0;
  int local_pose_valid_count = 0;
  int local_pose_rejected_count = 0;
  int layout_pair_candidate_count = 0;
  double stereo_rot_sum_rad = 0.0;
  double stereo_rot_max_rad = 0.0;
  double stereo_trans_sum_m = 0.0;
  double stereo_trans_max_m = 0.0;
  double layout_rot_sum_rad = 0.0;
  double layout_rot_max_rad = 0.0;
  double layout_trans_sum_m = 0.0;
  double layout_trans_max_m = 0.0;
};

void ApplySelectionCoObsFactorBaOptions(StereoExtrinsicSolverOptions* options) {
  if (options == nullptr || !options->selection_coobs_factor_ba_enable) {
    return;
  }
  options->coobs_factor_ba_apply_stereo_factor =
      options->selection_coobs_factor_ba_apply_stereo_factor;
  options->coobs_factor_ba_apply_layout_factor =
      options->selection_coobs_factor_ba_apply_layout_factor;
  options->coobs_factor_ba_current_stereo_weight =
      options->selection_coobs_factor_ba_stereo_weight;
  options->coobs_factor_ba_current_layout_weight =
      options->selection_coobs_factor_ba_layout_weight;
}

void AccumulateCoObsFactorResidual(const Eigen::Isometry3d& predicted,
                                   const Eigen::Isometry3d& measured,
                                   double* rot_sum_rad,
                                   double* rot_max_rad,
                                   double* trans_sum_m,
                                   double* trans_max_m) {
  const double rot = RotationDistanceRadians(predicted, measured);
  const double trans =
      (predicted.translation() - measured.translation()).norm();
  if (rot_sum_rad != nullptr) {
    *rot_sum_rad += rot;
  }
  if (rot_max_rad != nullptr) {
    *rot_max_rad = std::max(*rot_max_rad, rot);
  }
  if (trans_sum_m != nullptr) {
    *trans_sum_m += trans;
  }
  if (trans_max_m != nullptr) {
    *trans_max_m = std::max(*trans_max_m, trans);
  }
}

void CopyCoObsTrialDiagnostics(const StereoGlobalSparseBaSummary& summary,
                               StereoPairTrialSelectionDecision* decision) {
  if (decision == nullptr) {
    return;
  }
  decision->trial_coobs_stereo_factor_count =
      summary.coobs_stereo_factor_count;
  decision->trial_coobs_layout_factor_count =
      summary.coobs_layout_factor_count;
  decision->trial_coobs_stereo_initial_rot_mean_deg =
      summary.coobs_stereo_initial_rot_mean_deg;
  decision->trial_coobs_stereo_initial_rot_max_deg =
      summary.coobs_stereo_initial_rot_max_deg;
  decision->trial_coobs_stereo_initial_trans_mean_m =
      summary.coobs_stereo_initial_trans_mean_m;
  decision->trial_coobs_stereo_initial_trans_max_m =
      summary.coobs_stereo_initial_trans_max_m;
  decision->trial_coobs_layout_initial_rot_mean_deg =
      summary.coobs_layout_initial_rot_mean_deg;
  decision->trial_coobs_layout_initial_rot_max_deg =
      summary.coobs_layout_initial_rot_max_deg;
  decision->trial_coobs_layout_initial_trans_mean_m =
      summary.coobs_layout_initial_trans_mean_m;
  decision->trial_coobs_layout_initial_trans_max_m =
      summary.coobs_layout_initial_trans_max_m;
}

void CopyCoObsTrialDiagnostics(const StereoGlobalSparseBaSummary& summary,
                               StereoPairBoardTrialSelectionDecision* decision) {
  if (decision == nullptr) {
    return;
  }
  decision->trial_coobs_stereo_factor_count =
      summary.coobs_stereo_factor_count;
  decision->trial_coobs_layout_factor_count =
      summary.coobs_layout_factor_count;
  decision->trial_coobs_stereo_initial_rot_mean_deg =
      summary.coobs_stereo_initial_rot_mean_deg;
  decision->trial_coobs_stereo_initial_rot_max_deg =
      summary.coobs_stereo_initial_rot_max_deg;
  decision->trial_coobs_stereo_initial_trans_mean_m =
      summary.coobs_stereo_initial_trans_mean_m;
  decision->trial_coobs_stereo_initial_trans_max_m =
      summary.coobs_stereo_initial_trans_max_m;
  decision->trial_coobs_layout_initial_rot_mean_deg =
      summary.coobs_layout_initial_rot_mean_deg;
  decision->trial_coobs_layout_initial_rot_max_deg =
      summary.coobs_layout_initial_rot_max_deg;
  decision->trial_coobs_layout_initial_trans_mean_m =
      summary.coobs_layout_initial_trans_mean_m;
  decision->trial_coobs_layout_initial_trans_max_m =
      summary.coobs_layout_initial_trans_max_m;
}

double ComputeCoObsAwareAcceptanceScore(
    const StereoGlobalSparseBaSummary& summary,
    const StereoExtrinsicSolverOptions& options) {
  if (!options.coobs_aware_acceptance_enable) {
    return 0.0;
  }
  double score = 0.0;
  if (summary.coobs_stereo_factor_count > 0) {
    const double rot_scale =
        std::max(1e-12, options.coobs_aware_acceptance_stereo_rot_scale_deg);
    const double trans_scale =
        std::max(1e-12, options.coobs_aware_acceptance_stereo_trans_scale_m);
    score +=
        std::min(2.0, summary.coobs_stereo_initial_rot_mean_deg / rot_scale);
    score += std::min(
        2.0, summary.coobs_stereo_initial_trans_mean_m / trans_scale);
  }
  if (summary.coobs_layout_factor_count > 0) {
    const double rot_scale =
        std::max(1e-12, options.coobs_aware_acceptance_layout_rot_scale_deg);
    const double trans_scale =
        std::max(1e-12, options.coobs_aware_acceptance_layout_trans_scale_m);
    score +=
        std::min(2.0, summary.coobs_layout_initial_rot_mean_deg / rot_scale);
    score += std::min(
        2.0, summary.coobs_layout_initial_trans_mean_m / trans_scale);
  }
  return score;
}

template <typename DecisionT>
bool EvaluateCoObsAwareBalanceGuard(const StereoExtrinsicSolverOptions& options,
                                    DecisionT* decision) {
  if (decision == nullptr) {
    return false;
  }
  const double cam0_over = std::max(0.0, decision->cam0_rmse_delta);
  const double cam1_over = std::max(0.0, decision->cam1_rmse_delta);
  decision->coobs_acceptance_camera_delta_imbalance =
      std::abs(cam0_over - cam1_over);
  const double smaller = std::max(1e-12, std::min(cam0_over, cam1_over));
  decision->coobs_acceptance_camera_delta_ratio =
      std::max(cam0_over, cam1_over) / smaller;
  if (!options.coobs_aware_acceptance_balance_guard_enable) {
    decision->coobs_acceptance_balance_pass = true;
    return true;
  }
  decision->coobs_acceptance_balance_pass =
      decision->coobs_acceptance_camera_delta_imbalance <=
          options.coobs_aware_acceptance_max_camera_delta_imbalance &&
      decision->coobs_acceptance_camera_delta_ratio <=
          options.coobs_aware_acceptance_max_camera_delta_ratio;
  return decision->coobs_acceptance_balance_pass;
}

bool TryAcceptByCoObsAwareScore(
    const StereoGlobalSparseBaSummary& summary,
    const StereoExtrinsicSolverOptions& options,
    bool hard_validity_pass,
    bool pair_completion_candidate,
    StereoPairBoardTrialSelectionDecision* decision) {
  if (decision == nullptr || !options.coobs_aware_acceptance_enable ||
      !hard_validity_pass) {
    return false;
  }
  decision->coobs_acceptance_health_pass =
      decision->total_rmse_delta <=
          options.coobs_aware_acceptance_max_total_rmse_delta &&
      decision->cam0_rmse_delta <=
          options.coobs_aware_acceptance_max_camera_rmse_delta &&
      decision->cam1_rmse_delta <=
          options.coobs_aware_acceptance_max_camera_rmse_delta;
  decision->coobs_acceptance_structure_pass =
      !options.coobs_aware_acceptance_require_pair_completion ||
      pair_completion_candidate;
  EvaluateCoObsAwareBalanceGuard(options, decision);
  if (!decision->coobs_acceptance_health_pass ||
      !decision->coobs_acceptance_structure_pass ||
      !decision->coobs_acceptance_balance_pass) {
    return false;
  }
  decision->coobs_acceptance_score =
      ComputeCoObsAwareAcceptanceScore(summary, options);
  if (decision->coobs_acceptance_score <
      options.coobs_aware_acceptance_min_score) {
    return false;
  }
  decision->accepted = true;
  decision->accepted_by_coobs_aware_acceptance = true;
  decision->accepted_by_batch_acceptance = true;
  decision->accept_reason = "coobs_aware_acceptance";
  decision->reject_reason.clear();
  return true;
}

bool TryAcceptByCoObsAwareScore(
    const StereoGlobalSparseBaSummary& summary,
    const StereoExtrinsicSolverOptions& options,
    bool solution_valid,
    bool pair_completion_candidate,
    Stage6IncrementalBatchResult* batch_result,
    StereoPairBoardTrialSelectionDecision* decision) {
  if (decision == nullptr || batch_result == nullptr ||
      !options.coobs_aware_acceptance_enable || !solution_valid) {
    return false;
  }
  decision->coobs_acceptance_health_pass =
      decision->total_rmse_delta <=
          options.coobs_aware_acceptance_max_total_rmse_delta &&
      decision->cam0_rmse_delta <=
          options.coobs_aware_acceptance_max_camera_rmse_delta &&
      decision->cam1_rmse_delta <=
          options.coobs_aware_acceptance_max_camera_rmse_delta;
  decision->coobs_acceptance_structure_pass =
      !options.coobs_aware_acceptance_require_pair_completion ||
      pair_completion_candidate;
  EvaluateCoObsAwareBalanceGuard(options, decision);
  if (!decision->coobs_acceptance_health_pass ||
      !decision->coobs_acceptance_structure_pass ||
      !decision->coobs_acceptance_balance_pass) {
    return false;
  }
  decision->coobs_acceptance_score =
      ComputeCoObsAwareAcceptanceScore(summary, options);
  if (decision->coobs_acceptance_score <
      options.coobs_aware_acceptance_min_score) {
    return false;
  }
  batch_result->batchAccepted = true;
  batch_result->accept_reason = "coobs_aware_acceptance";
  batch_result->reject_reason.clear();
  batch_result->committed_or_rollback = "committed";
  decision->accepted_by_coobs_aware_acceptance = true;
  return true;
}

bool TryAcceptByCoObsAwareScore(
    const StereoGlobalSparseBaSummary& summary,
    const StereoExtrinsicSolverOptions& options,
    bool solution_valid,
    Stage6IncrementalBatchResult* batch_result,
    StereoPairTrialSelectionDecision* decision) {
  if (decision == nullptr || batch_result == nullptr ||
      !options.coobs_aware_acceptance_enable || !solution_valid) {
    return false;
  }
  decision->coobs_acceptance_health_pass =
      decision->total_rmse_delta <=
          options.coobs_aware_acceptance_max_total_rmse_delta &&
      decision->cam0_rmse_delta <=
          options.coobs_aware_acceptance_max_camera_rmse_delta &&
      decision->cam1_rmse_delta <=
          options.coobs_aware_acceptance_max_camera_rmse_delta;
  decision->coobs_acceptance_structure_pass = true;
  EvaluateCoObsAwareBalanceGuard(options, decision);
  if (!decision->coobs_acceptance_health_pass ||
      !decision->coobs_acceptance_balance_pass) {
    return false;
  }
  decision->coobs_acceptance_score =
      ComputeCoObsAwareAcceptanceScore(summary, options);
  if (decision->coobs_acceptance_score <
      options.coobs_aware_acceptance_min_score) {
    return false;
  }
  batch_result->batchAccepted = true;
  batch_result->accept_reason = "coobs_aware_acceptance";
  batch_result->reject_reason.clear();
  batch_result->committed_or_rollback = "committed";
  decision->accepted_by_coobs_aware_acceptance = true;
  return true;
}

CoObsLocalPoseKey MakeCoObsLocalPoseKey(int pair_index,
                                        int camera_index,
                                        int board_id) {
  return std::make_tuple(pair_index, camera_index, board_id);
}

std::map<CoObsLocalPoseKey, CoObsLocalPoseMeasurement>
BuildCoObsLocalPoseMeasurements(
    const StereoMeasurementDataset& dataset,
    const StereoSceneState& scene_state,
    const StereoPairSelectionSummary& selection_summary,
    const StereoExtrinsicSolverOptions& options,
    CoObsFactorBuildStats* stats) {
  std::map<CoObsLocalPoseKey, CoObsLocalPoseMeasurement> measurements;
  const int min_corners =
      std::max(4, options.coobs_factor_ba_min_corners_per_cam_board);
  const double max_rmse =
      std::max(0.0, options.coobs_factor_ba_max_local_pose_rmse);
  const auto count_total_group_corners =
      [&](int pair_index, int camera_index, int board_id) {
        int count = 0;
        for (const StereoObservation& observation : dataset.observations) {
          if (observation.pair_index == pair_index &&
              observation.camera_index == camera_index &&
              observation.board_id == board_id &&
              observation.used_in_solver) {
            ++count;
          }
        }
        return count;
      };
  for (int pair_index : selection_summary.selected_pair_indices) {
    const auto boards_it = dataset.training_pair_board_ids.find(pair_index);
    if (boards_it == dataset.training_pair_board_ids.end()) {
      continue;
    }
    for (int board_id : boards_it->second) {
      if (!PairBoardSelected(selection_summary, pair_index, board_id)) {
        continue;
      }
      for (int camera_index = 0; camera_index <= 1; ++camera_index) {
        CoObsLocalPoseMeasurement measurement;
        measurement.pair_index = pair_index;
        measurement.camera_index = camera_index;
        measurement.board_id = board_id;
        measurement.corner_count =
            count_total_group_corners(pair_index, camera_index, board_id);
        const int outer_corner_count =
            CountOuterObservationsForCamera(dataset, pair_index, camera_index,
                                            board_id);
        if (measurement.corner_count < min_corners) {
          if (stats != nullptr) {
            ++stats->local_pose_rejected_count;
          }
          measurements[MakeCoObsLocalPoseKey(pair_index, camera_index,
                                             board_id)] = measurement;
          continue;
        }
        if (outer_corner_count < 4) {
          if (stats != nullptr) {
            ++stats->local_pose_rejected_count;
          }
          measurements[MakeCoObsLocalPoseKey(pair_index, camera_index,
                                             board_id)] = measurement;
          continue;
        }
        if (stats != nullptr) {
          ++stats->local_pose_attempt_count;
        }
        const StereoCameraFixedCalibration& calibration =
            camera_index == 0 ? scene_state.cam0 : scene_state.cam1;
        measurement.success = EstimateCameraBoardPoseWithRmse(
            dataset, calibration, pair_index, camera_index, board_id,
            &measurement.T_camera_board, &measurement.local_rmse);
        if (measurement.success &&
            std::isfinite(measurement.local_rmse) &&
            measurement.local_rmse <= max_rmse) {
          if (stats != nullptr) {
            ++stats->local_pose_valid_count;
          }
        } else {
          measurement.success = false;
          if (stats != nullptr) {
            ++stats->local_pose_rejected_count;
          }
        }
        measurements[MakeCoObsLocalPoseKey(pair_index, camera_index,
                                           board_id)] = measurement;
      }
    }
  }
  return measurements;
}

const CoObsLocalPoseMeasurement* FindCoObsMeasurement(
    const std::map<CoObsLocalPoseKey, CoObsLocalPoseMeasurement>& measurements,
    int pair_index,
    int camera_index,
    int board_id) {
  const auto it =
      measurements.find(MakeCoObsLocalPoseKey(pair_index, camera_index,
                                              board_id));
  if (it == measurements.end() || !it->second.success) {
    return nullptr;
  }
  return &it->second;
}

template <typename GeometryT0, typename GeometryT1>
void ComputeRigProjectionEquivalenceDiagnostics(
    const StereoMeasurementDataset& dataset,
    const StereoPairSelectionSummary& selection_summary,
    const StereoSceneState& scene_state,
    const GeometryT0& cam0_geometry,
    const GeometryT1& cam1_geometry,
    const Eigen::Isometry3d& T_cam0_rig,
    const Eigen::Isometry3d& T_cam1_rig,
    const std::map<int, StereoPoseVariableState>& pair_variables,
    const std::map<int, StereoPoseVariableState>& board_variables,
    double* max_pixel_diff,
    double* max_angular_diff_rad) {
  if (max_pixel_diff != nullptr) {
    *max_pixel_diff = 0.0;
  }
  if (max_angular_diff_rad != nullptr) {
    *max_angular_diff_rad = 0.0;
  }
  const Eigen::Isometry3d T_cam1_cam0 = ToIsometry3d(scene_state.T_cam1_cam0);
  for (const StereoObservation& observation : dataset.observations) {
    if (!observation.used_in_solver ||
        selection_summary.selected_pair_indices.count(observation.pair_index) ==
            0 ||
        !PairBoardSelected(selection_summary, observation.pair_index,
                           observation.board_id)) {
      continue;
    }
    const auto pair_it = pair_variables.find(observation.pair_index);
    if (pair_it == pair_variables.end()) {
      continue;
    }
    Eigen::Isometry3d T_world_board = Eigen::Isometry3d::Identity();
    if (observation.board_id != scene_state.gauge_fixed_board_id) {
      const auto board_it = board_variables.find(observation.board_id);
      if (board_it == board_variables.end()) {
        continue;
      }
      T_world_board = ToIsometry3d(board_it->second.transform.T());
    }
    const Eigen::Isometry3d T_cam0_world =
        ToIsometry3d(pair_it->second.transform.T());
    const Eigen::Vector4d X_board_h(observation.target_point_board.x(),
                                    observation.target_point_board.y(),
                                    observation.target_point_board.z(), 1.0);
    const Eigen::Vector4d X_world = T_world_board.matrix() * X_board_h;
    const Eigen::Vector4d X_cam0_legacy = T_cam0_world.matrix() * X_world;
    const Eigen::Vector4d X_cam_legacy =
        observation.camera_index == 0
            ? X_cam0_legacy
            : T_cam1_cam0.matrix() * X_cam0_legacy;
    const Eigen::Vector4d X_rig = T_cam0_world.matrix() * X_world;
    const Eigen::Vector4d X_cam_rig =
        observation.camera_index == 0 ? T_cam0_rig.matrix() * X_rig
                                      : T_cam1_rig.matrix() * X_rig;
    Eigen::Vector2d uv_legacy = Eigen::Vector2d::Zero();
    Eigen::Vector2d uv_rig = Eigen::Vector2d::Zero();
    bool ok_legacy = false;
    bool ok_rig = false;
    if (observation.camera_index == 0) {
      ok_legacy = cam0_geometry.homogeneousToKeypoint(X_cam_legacy, uv_legacy);
      ok_rig = cam0_geometry.homogeneousToKeypoint(X_cam_rig, uv_rig);
    } else {
      ok_legacy = cam1_geometry.homogeneousToKeypoint(X_cam_legacy, uv_legacy);
      ok_rig = cam1_geometry.homogeneousToKeypoint(X_cam_rig, uv_rig);
    }
    if (ok_legacy && ok_rig && uv_legacy.allFinite() && uv_rig.allFinite() &&
        max_pixel_diff != nullptr) {
      *max_pixel_diff = std::max(*max_pixel_diff, (uv_legacy - uv_rig).norm());
    }
    const Eigen::Vector3d legacy_head = X_cam_legacy.head<3>();
    const Eigen::Vector3d rig_head = X_cam_rig.head<3>();
    const double legacy_norm = legacy_head.norm();
    const double rig_norm = rig_head.norm();
    if (legacy_norm > 1e-12 && rig_norm > 1e-12 &&
        legacy_head.allFinite() && rig_head.allFinite() &&
        max_angular_diff_rad != nullptr) {
      const double dot = std::max(
          -1.0,
          std::min(1.0, legacy_head.normalized().dot(rig_head.normalized())));
      *max_angular_diff_rad =
          std::max(*max_angular_diff_rad, std::acos(dot));
    }
  }
}

template <typename GeometryT>
class StereoCameraReprojectionError : public aslam::backend::ErrorTermFs<2> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  using camera_dv_t = aslam::backend::CameraDesignVariable<GeometryT>;

  StereoCameraReprojectionError(const Eigen::Vector2d& measurement,
                                double weight,
                                const aslam::backend::HomogeneousExpression& point_camera,
                                const camera_dv_t& camera_dv,
                                double invalid_projection_penalty_pixels,
                                bool use_robust_loss = true)
      : measurement_(measurement),
        point_camera_(point_camera),
        camera_dv_(camera_dv),
        weight_(std::max(0.0, weight)),
        invalid_projection_penalty_pixels_(invalid_projection_penalty_pixels) {
    parent_t::setInvR(weight_ * Eigen::Matrix2d::Identity());
    aslam::backend::DesignVariable::set_t design_variables;
    point_camera_.getDesignVariables(design_variables);
    camera_dv_.getDesignVariables(design_variables);
    parent_t::setDesignVariablesIterator(design_variables.begin(),
                                         design_variables.end());
    if (use_robust_loss && weight_ > 0.0) {
      parent_t::setMEstimatorPolicy(
          boost::shared_ptr<aslam::backend::MEstimator>(
              new aslam::backend::HuberMEstimator(std::sqrt(weight_) * 2.0)));
    }
  }

 protected:
  double evaluateErrorImplementation() override {
    Eigen::Vector2d predicted = Eigen::Vector2d::Zero();
    bool valid = false;
    parent_t::setError(ComputeResidual(&predicted, &valid));
    return parent_t::evaluateChiSquaredError();
  }

  void evaluateJacobiansImplementation(
      aslam::backend::JacobianContainer& jacobians) const override {
    const Eigen::Vector4d point_homogeneous = point_camera_.toHomogeneous();
    typename GeometryT::jacobian_homogeneous_t projection_jacobian;
    Eigen::Vector2d predicted = Eigen::Vector2d::Zero();
    const bool valid_projection =
        camera_dv_.camera()->homogeneousToKeypoint(point_homogeneous, predicted,
                                                   projection_jacobian) &&
        predicted.allFinite() && projection_jacobian.allFinite();
    if (!valid_projection) {
      return;
    }
    point_camera_.evaluateJacobians(jacobians, -projection_jacobian);
    camera_dv_.evaluateJacobians(jacobians, point_homogeneous);
  }

 private:
  using parent_t = aslam::backend::ErrorTermFs<2>;

  Eigen::Vector2d ComputeResidual(Eigen::Vector2d* predicted,
                                  bool* valid_projection) const {
    const Eigen::Vector4d point_homogeneous = point_camera_.toHomogeneous();
    *predicted = Eigen::Vector2d::Zero();
    *valid_projection =
        camera_dv_.camera()->homogeneousToKeypoint(point_homogeneous, *predicted) &&
        predicted->allFinite();
    if (!(*valid_projection)) {
      *predicted = Eigen::Vector2d::Constant(
          std::numeric_limits<double>::quiet_NaN());
      return Eigen::Vector2d::Constant(invalid_projection_penalty_pixels_);
    }
    return measurement_ - *predicted;
  }

  Eigen::Vector2d measurement_;
  aslam::backend::HomogeneousExpression point_camera_;
  camera_dv_t camera_dv_;
  double weight_ = 1.0;
  double invalid_projection_penalty_pixels_ = 100.0;
};

template <typename GeometryT>
class StereoCameraAngularReprojectionError
    : public aslam::backend::ErrorTermFs<2> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  using camera_dv_t = aslam::backend::CameraDesignVariable<GeometryT>;

  StereoCameraAngularReprojectionError(
      const AngularObservationGeometry& observation_geometry,
      const Eigen::Vector2d& observed_image_xy,
      bool dynamic_observed_ray,
      const StereoCameraFixedCalibration& seed_calibration,
      bool use_normalize_jacobian,
      double weight,
      const aslam::backend::HomogeneousExpression& point_camera,
      const camera_dv_t& camera_dv,
      double invalid_projection_penalty_radians)
      : observation_geometry_(observation_geometry),
        observed_image_xy_(observed_image_xy),
        dynamic_observed_ray_(dynamic_observed_ray),
        seed_calibration_(seed_calibration),
        use_normalize_jacobian_(use_normalize_jacobian),
        point_camera_(point_camera),
        camera_dv_(camera_dv),
        weight_(std::max(0.0, weight)),
        invalid_projection_penalty_radians_(
            invalid_projection_penalty_radians) {
    parent_t::setInvR(weight_ * Eigen::Matrix2d::Identity());
    aslam::backend::DesignVariable::set_t design_variables;
    point_camera_.getDesignVariables(design_variables);
    camera_dv_.getDesignVariables(design_variables);
    parent_t::setDesignVariablesIterator(design_variables.begin(),
                                         design_variables.end());
    if (weight_ > 0.0) {
      parent_t::setMEstimatorPolicy(
          boost::shared_ptr<aslam::backend::MEstimator>(
              new aslam::backend::HuberMEstimator(
                  std::sqrt(weight_) * 0.01)));
    }
  }

  StereoCameraAngularReprojectionError(
      const AngularObservationGeometry& observation_geometry,
      const Eigen::Vector2d& observed_image_xy,
      bool dynamic_observed_ray,
      const StereoCameraFixedCalibration& seed_calibration,
      bool use_normalize_jacobian,
      double weight,
      const Eigen::Matrix2d& sqrt_information,
      const aslam::backend::HomogeneousExpression& point_camera,
      const camera_dv_t& camera_dv,
      double invalid_projection_penalty_radians)
      : observation_geometry_(observation_geometry),
        observed_image_xy_(observed_image_xy),
        dynamic_observed_ray_(dynamic_observed_ray),
        seed_calibration_(seed_calibration),
        use_normalize_jacobian_(use_normalize_jacobian),
        point_camera_(point_camera),
        camera_dv_(camera_dv),
        weight_(std::max(0.0, weight)),
        invalid_projection_penalty_radians_(
            invalid_projection_penalty_radians) {
    if (sqrt_information.allFinite()) {
      parent_t::setInvR(std::sqrt(weight_) * sqrt_information);
    } else {
      parent_t::setInvR(weight_ * Eigen::Matrix2d::Identity());
    }
    aslam::backend::DesignVariable::set_t design_variables;
    point_camera_.getDesignVariables(design_variables);
    camera_dv_.getDesignVariables(design_variables);
    parent_t::setDesignVariablesIterator(design_variables.begin(),
                                         design_variables.end());
    if (weight_ > 0.0) {
      parent_t::setMEstimatorPolicy(
          boost::shared_ptr<aslam::backend::MEstimator>(
              new aslam::backend::HuberMEstimator(
                  std::sqrt(weight_) * 0.01)));
    }
  }

 protected:
  double evaluateErrorImplementation() override {
    parent_t::setError(ComputeResidual());
    return parent_t::evaluateChiSquaredError();
  }

  void evaluateJacobiansImplementation(
      aslam::backend::JacobianContainer& jacobians) const override {
    AngularObservationGeometry observation_geometry;
    AngularPredictionGeometry prediction_geometry;
    bool valid_projection = false;
    const Eigen::Vector2d residual =
        ComputeResidual(&observation_geometry, &prediction_geometry,
                        &valid_projection);
    (void)residual;
    if (!valid_projection || !observation_geometry.success) {
      return;
    }
    const Eigen::Vector4d point_homogeneous = point_camera_.toHomogeneous();
    const Eigen::Vector3d point = point_homogeneous.head<3>();
    const double norm = point.norm();
    if (!(norm > 1e-12) || !std::isfinite(norm)) {
      return;
    }
    const Eigen::Vector3d unit = point / norm;
    Eigen::Matrix3d d_unit_d_point = Eigen::Matrix3d::Identity();
    if (use_normalize_jacobian_) {
      d_unit_d_point =
          (Eigen::Matrix3d::Identity() - unit * unit.transpose()) / norm;
    }
    const Eigen::Matrix<double, 2, 3> tangent_t =
        observation_geometry.tangent_basis.transpose();
    const Eigen::Matrix<double, 2, 3> d_residual_d_point =
        tangent_t * d_unit_d_point;
    const Eigen::Matrix<double, 2, 4> d_residual_d_homogeneous =
        (Eigen::Matrix<double, 2, 4>() <<
             d_residual_d_point(0, 0), d_residual_d_point(0, 1),
             d_residual_d_point(0, 2), 0.0,
             d_residual_d_point(1, 0), d_residual_d_point(1, 1),
            d_residual_d_point(1, 2), 0.0)
            .finished();
    point_camera_.evaluateJacobians(jacobians, d_residual_d_homogeneous);
    if (!dynamic_observed_ray_) {
      return;
    }
    auto add_finite_difference_camera_jacobian =
        [&](const auto& design_variable_adapter) {
          if (design_variable_adapter == nullptr ||
              !design_variable_adapter->isActive()) {
            return;
          }
          const Eigen::MatrixXd base_parameters =
              design_variable_adapter->getParameters();
          const int dimension = static_cast<int>(base_parameters.size());
          if (dimension <= 0) {
            return;
          }
          Eigen::MatrixXd camera_jacobian(2, dimension);
          camera_jacobian.setZero();
          for (int index = 0; index < dimension; ++index) {
            const Eigen::Index row =
                static_cast<Eigen::Index>(index % base_parameters.rows());
            const Eigen::Index col =
                static_cast<Eigen::Index>(index / base_parameters.rows());
            const double base_value = base_parameters(row, col);
            const double epsilon =
                std::max(1e-7,
                         1e-6 * std::max(1.0, std::fabs(base_value)));

            Eigen::MatrixXd positive = base_parameters;
            positive(row, col) = base_value + epsilon;
            design_variable_adapter->setParameters(positive);
            AngularObservationGeometry positive_observation_geometry;
            AngularPredictionGeometry positive_prediction_geometry;
            bool positive_valid = false;
            const Eigen::Vector2d positive_residual = ComputeResidual(
                &positive_observation_geometry, &positive_prediction_geometry,
                &positive_valid);

            Eigen::MatrixXd negative = base_parameters;
            negative(row, col) = base_value - epsilon;
            design_variable_adapter->setParameters(negative);
            AngularObservationGeometry negative_observation_geometry;
            AngularPredictionGeometry negative_prediction_geometry;
            bool negative_valid = false;
            const Eigen::Vector2d negative_residual = ComputeResidual(
                &negative_observation_geometry, &negative_prediction_geometry,
                &negative_valid);

            if (positive_valid && negative_valid &&
                positive_residual.allFinite() &&
                negative_residual.allFinite()) {
              camera_jacobian.col(index) =
                  (positive_residual - negative_residual) / (2.0 * epsilon);
            } else {
              camera_jacobian.col(index).setZero();
            }
          }
          design_variable_adapter->setParameters(base_parameters);
          jacobians.add(design_variable_adapter.get(), camera_jacobian);
        };
    add_finite_difference_camera_jacobian(
        const_cast<camera_dv_t&>(camera_dv_).projectionDesignVariable());
    add_finite_difference_camera_jacobian(
        const_cast<camera_dv_t&>(camera_dv_).distortionDesignVariable());
  }

 private:
  using parent_t = aslam::backend::ErrorTermFs<2>;

  Eigen::Vector2d ComputeResidual() const {
    AngularObservationGeometry observation_geometry;
    AngularPredictionGeometry prediction_geometry;
    bool valid_projection = false;
    return ComputeResidual(&observation_geometry, &prediction_geometry,
                           &valid_projection);
  }

  Eigen::Vector2d ComputeResidual(
      AngularObservationGeometry* observation_geometry,
      AngularPredictionGeometry* prediction_geometry,
      bool* valid_projection) const {
    if (observation_geometry == nullptr || prediction_geometry == nullptr ||
        valid_projection == nullptr) {
      throw std::runtime_error(
          "StereoCameraAngularReprojectionError requires valid output pointers.");
    }
    const Eigen::Vector4d point_homogeneous = point_camera_.toHomogeneous();
    const Eigen::Vector3d point = point_homogeneous.head<3>();
    *observation_geometry = observation_geometry_;
    if (dynamic_observed_ray_) {
      const DoubleSphereCameraModel residual_camera =
          DoubleSphereCameraModel::FromConfig(MakeCameraConfigFromGeometry(
              *camera_dv_.camera(), seed_calibration_));
      if (!ComputeAngularObservationGeometry(
              residual_camera, observed_image_xy_, observation_geometry) ||
          !ComputeAngularPredictionGeometry(
              residual_camera, point, prediction_geometry)) {
        *valid_projection = false;
        return Eigen::Vector2d::Constant(invalid_projection_penalty_radians_);
      }
      *valid_projection = true;
      return ComputeAngularResidualTangent(*observation_geometry,
                                           *prediction_geometry);
    }
    *prediction_geometry = AngularPredictionGeometry{};
    if (!observation_geometry->success) {
      *valid_projection = false;
      return Eigen::Vector2d::Constant(invalid_projection_penalty_radians_);
    }
    const double norm = point.norm();
    if (!(norm > 1e-12) || !std::isfinite(norm)) {
      *valid_projection = false;
      return Eigen::Vector2d::Constant(invalid_projection_penalty_radians_);
    }
    const Eigen::Vector3d predicted_ray = point / norm;
    if (!predicted_ray.allFinite()) {
      *valid_projection = false;
      return Eigen::Vector2d::Constant(invalid_projection_penalty_radians_);
    }
    prediction_geometry->predicted_ray = predicted_ray;
    prediction_geometry->valid_projection = true;
    *valid_projection = true;
    return observation_geometry->tangent_basis.transpose() *
           (predicted_ray - observation_geometry->observed_ray);
  }

  AngularObservationGeometry observation_geometry_;
  Eigen::Vector2d observed_image_xy_ = Eigen::Vector2d::Zero();
  bool dynamic_observed_ray_ = false;
  StereoCameraFixedCalibration seed_calibration_;
  bool use_normalize_jacobian_ = true;
  aslam::backend::HomogeneousExpression point_camera_;
  camera_dv_t camera_dv_;
  double weight_ = 1.0;
  double invalid_projection_penalty_radians_ = 0.35;
};

class StereoCameraSphericalChordalError
    : public aslam::backend::ErrorTermFs<3> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  StereoCameraSphericalChordalError(
      const Eigen::Vector3d& observed_ray,
      double weight,
      const aslam::backend::HomogeneousExpression& point_camera,
      double invalid_projection_penalty)
      : observed_ray_(observed_ray),
        point_camera_(point_camera),
        weight_(std::max(0.0, weight)),
        invalid_projection_penalty_(invalid_projection_penalty) {
    parent_t::setInvR(weight_ * Eigen::Matrix3d::Identity());
    aslam::backend::DesignVariable::set_t design_variables;
    point_camera_.getDesignVariables(design_variables);
    parent_t::setDesignVariablesIterator(design_variables.begin(),
                                         design_variables.end());
    if (weight_ > 0.0) {
      parent_t::setMEstimatorPolicy(
          boost::shared_ptr<aslam::backend::MEstimator>(
              new aslam::backend::HuberMEstimator(
                  std::sqrt(weight_) * 0.01)));
    }
  }

 protected:
  double evaluateErrorImplementation() override {
    parent_t::setError(ComputeResidual());
    return parent_t::evaluateChiSquaredError();
  }

  void evaluateJacobiansImplementation(
      aslam::backend::JacobianContainer& jacobians) const override {
    const Eigen::Vector4d point_homogeneous = point_camera_.toHomogeneous();
    const Eigen::Vector3d point = point_homogeneous.head<3>();
    const double norm = point.norm();
    if (!(norm > 1e-12) || !std::isfinite(norm)) {
      return;
    }
    const Eigen::Vector3d unit = point / norm;
    const Eigen::Matrix3d d_unit_d_point =
        (Eigen::Matrix3d::Identity() - unit * unit.transpose()) / norm;
    Eigen::Matrix<double, 3, 4> d_residual_d_homogeneous =
        Eigen::Matrix<double, 3, 4>::Zero();
    d_residual_d_homogeneous.block<3, 3>(0, 0) = d_unit_d_point;
    point_camera_.evaluateJacobians(jacobians, d_residual_d_homogeneous);
  }

 private:
  using parent_t = aslam::backend::ErrorTermFs<3>;

  Eigen::Vector3d ComputeResidual() const {
    const Eigen::Vector3d point = point_camera_.toHomogeneous().head<3>();
    const double norm = point.norm();
    if (!(norm > 1e-12) || !std::isfinite(norm) ||
        !observed_ray_.allFinite()) {
      return Eigen::Vector3d::Constant(invalid_projection_penalty_);
    }
    const Eigen::Vector3d predicted_ray = point / norm;
    if (!predicted_ray.allFinite()) {
      return Eigen::Vector3d::Constant(invalid_projection_penalty_);
    }
    return predicted_ray - observed_ray_;
  }

  Eigen::Vector3d observed_ray_ = Eigen::Vector3d::Zero();
  aslam::backend::HomogeneousExpression point_camera_;
  double weight_ = 1.0;
  double invalid_projection_penalty_ = 0.35;
};

void AppendUniqueWarning(const std::string& warning,
                         std::vector<std::string>* warnings) {
  if (warnings == nullptr || warning.empty()) {
    return;
  }
  if (std::find(warnings->begin(), warnings->end(), warning) == warnings->end()) {
    warnings->push_back(warning);
  }
}

struct StereoRefitDiagnostics {
  bool pose_success = false;
  bool used_symmetric_refit = false;
  bool refit_fell_back_to_seed = false;
  bool improved = false;
};

double ElapsedSeconds(const Clock::time_point& start) {
  return std::chrono::duration<double>(Clock::now() - start).count();
}

boost::shared_ptr<DsGeometry> MakeDsGeometry(
    const StereoCameraFixedCalibration& calibration) {
  const OuterBootstrapCameraIntrinsics intrinsics =
      MakeOuterBootstrapCameraIntrinsics(calibration);
  DsProjection projection(intrinsics.xi, intrinsics.alpha, intrinsics.fu,
                          intrinsics.fv, intrinsics.cu, intrinsics.cv,
                          intrinsics.resolution.width, intrinsics.resolution.height);
  return boost::shared_ptr<DsGeometry>(
      new DsGeometry(projection, aslam::cameras::GlobalShutter(),
                     aslam::cameras::NoMask()));
}

boost::shared_ptr<EucmGeometry> MakeEucmGeometry(
    const StereoCameraFixedCalibration& calibration) {
  const OuterBootstrapCameraIntrinsics intrinsics =
      MakeOuterBootstrapCameraIntrinsics(calibration);
  EucmProjection projection(intrinsics.alpha, intrinsics.beta, intrinsics.fu,
                            intrinsics.fv, intrinsics.cu, intrinsics.cv,
                            intrinsics.resolution.width, intrinsics.resolution.height);
  return boost::shared_ptr<EucmGeometry>(
      new EucmGeometry(projection, aslam::cameras::GlobalShutter(),
                       aslam::cameras::NoMask()));
}

boost::shared_ptr<PinholeEquiGeometry> MakePinholeEquiGeometry(
    const StereoCameraFixedCalibration& calibration) {
  const OuterBootstrapCameraIntrinsics intrinsics =
      MakeOuterBootstrapCameraIntrinsics(calibration);
  std::vector<double> distortion = intrinsics.DistortionVector();
  if (distortion.size() != 4) {
    distortion.resize(4, 0.0);
  }
  PinholeEquiProjection projection(
      intrinsics.fu, intrinsics.fv, intrinsics.cu, intrinsics.cv,
      intrinsics.resolution.width, intrinsics.resolution.height,
      aslam::cameras::EquidistantDistortion(distortion[0], distortion[1],
                                            distortion[2], distortion[3]));
  return boost::shared_ptr<PinholeEquiGeometry>(
      new PinholeEquiGeometry(projection, aslam::cameras::GlobalShutter(),
                              aslam::cameras::NoMask()));
}

boost::shared_ptr<OmniGeometry> MakeOmniGeometry(
    const StereoCameraFixedCalibration& calibration) {
  const OuterBootstrapCameraIntrinsics intrinsics =
      MakeOuterBootstrapCameraIntrinsics(calibration);
  OmniProjection projection(
      intrinsics.xi, intrinsics.fu, intrinsics.fv, intrinsics.cu,
      intrinsics.cv, intrinsics.resolution.width, intrinsics.resolution.height);
  return boost::shared_ptr<OmniGeometry>(
      new OmniGeometry(projection, aslam::cameras::GlobalShutter(),
                       aslam::cameras::NoMask()));
}

boost::shared_ptr<OmniRadtanGeometry> MakeOmniRadtanGeometry(
    const StereoCameraFixedCalibration& calibration) {
  const OuterBootstrapCameraIntrinsics intrinsics =
      MakeOuterBootstrapCameraIntrinsics(calibration);
  std::vector<double> distortion = intrinsics.DistortionVector();
  distortion.resize(4, 0.0);
  OmniRadtanProjection projection(
      intrinsics.xi, intrinsics.fu, intrinsics.fv, intrinsics.cu,
      intrinsics.cv, intrinsics.resolution.width, intrinsics.resolution.height,
      aslam::cameras::RadialTangentialDistortion(
          distortion[0], distortion[1], distortion[2], distortion[3]));
  return boost::shared_ptr<OmniRadtanGeometry>(
      new OmniRadtanGeometry(projection, aslam::cameras::GlobalShutter(),
                             aslam::cameras::NoMask()));
}

template <typename GeometryT>
boost::shared_ptr<GeometryT> MakeTypedStereoGeometry(
    const StereoCameraFixedCalibration& calibration);

template <>
boost::shared_ptr<DsGeometry> MakeTypedStereoGeometry<DsGeometry>(
    const StereoCameraFixedCalibration& calibration) {
  return MakeDsGeometry(calibration);
}

template <>
boost::shared_ptr<EucmGeometry> MakeTypedStereoGeometry<EucmGeometry>(
    const StereoCameraFixedCalibration& calibration) {
  return MakeEucmGeometry(calibration);
}

template <>
boost::shared_ptr<PinholeEquiGeometry> MakeTypedStereoGeometry<PinholeEquiGeometry>(
    const StereoCameraFixedCalibration& calibration) {
  return MakePinholeEquiGeometry(calibration);
}

template <>
boost::shared_ptr<OmniGeometry> MakeTypedStereoGeometry<OmniGeometry>(
    const StereoCameraFixedCalibration& calibration) {
  return MakeOmniGeometry(calibration);
}

template <>
boost::shared_ptr<OmniRadtanGeometry>
MakeTypedStereoGeometry<OmniRadtanGeometry>(
    const StereoCameraFixedCalibration& calibration) {
  return MakeOmniRadtanGeometry(calibration);
}

template <>
IntermediateCameraConfig MakeCameraConfigFromGeometry<DsGeometry>(
    const DsGeometry& geometry,
    const StereoCameraFixedCalibration& seed_calibration) {
  IntermediateCameraConfig config;
  config.camera_yaml.clear();
  config.camera_model = "ds";
  config.distortion_model = "none";
  config.intrinsics = {geometry.projection().xi(), geometry.projection().alpha(),
                       geometry.projection().fu(), geometry.projection().fv(),
                       geometry.projection().cu(), geometry.projection().cv()};
  config.distortion_coeffs.clear();
  config.resolution = {geometry.projection().width(),
                       geometry.projection().height()};
  return config;
}

template <>
IntermediateCameraConfig MakeCameraConfigFromGeometry<EucmGeometry>(
    const EucmGeometry& geometry,
    const StereoCameraFixedCalibration& seed_calibration) {
  IntermediateCameraConfig config;
  config.camera_yaml.clear();
  config.camera_model = "eucm";
  config.distortion_model = "none";
  config.intrinsics = {geometry.projection().alpha(), geometry.projection().beta(),
                       geometry.projection().fu(), geometry.projection().fv(),
                       geometry.projection().cu(), geometry.projection().cv()};
  config.distortion_coeffs.clear();
  config.resolution = {geometry.projection().width(),
                       geometry.projection().height()};
  return config;
}

template <>
IntermediateCameraConfig MakeCameraConfigFromGeometry<PinholeEquiGeometry>(
    const PinholeEquiGeometry& geometry,
    const StereoCameraFixedCalibration& seed_calibration) {
  IntermediateCameraConfig config;
  config.camera_yaml.clear();
  config.camera_model = "pinhole";
  config.distortion_model = "equi";
  config.intrinsics = {geometry.projection().fu(), geometry.projection().fv(),
                       geometry.projection().cu(), geometry.projection().cv()};
  Eigen::MatrixXd distortion_parameters;
  geometry.projection().distortion().getParameters(distortion_parameters);
  config.distortion_coeffs.resize(
      static_cast<std::size_t>(distortion_parameters.rows()), 0.0);
  for (Eigen::Index index = 0; index < distortion_parameters.rows(); ++index) {
    config.distortion_coeffs[static_cast<std::size_t>(index)] =
        distortion_parameters(index, 0);
  }
  config.resolution = {geometry.projection().width(),
                       geometry.projection().height()};
  return config;
}

template <>
IntermediateCameraConfig MakeCameraConfigFromGeometry<OmniGeometry>(
    const OmniGeometry& geometry,
    const StereoCameraFixedCalibration& seed_calibration) {
  IntermediateCameraConfig config;
  config.camera_yaml.clear();
  config.camera_model = "omni";
  config.distortion_model = "none";
  config.intrinsics = {
      geometry.projection().xi(), geometry.projection().fu(),
      geometry.projection().fv(), geometry.projection().cu(),
      geometry.projection().cv()};
  config.distortion_coeffs.clear();
  config.resolution = {geometry.projection().width(),
                       geometry.projection().height()};
  return config;
}

template <>
IntermediateCameraConfig MakeCameraConfigFromGeometry<OmniRadtanGeometry>(
    const OmniRadtanGeometry& geometry,
    const StereoCameraFixedCalibration& seed_calibration) {
  IntermediateCameraConfig config;
  config.camera_yaml.clear();
  config.camera_model = "omni";
  config.distortion_model = "radtan";
  config.intrinsics = {
      geometry.projection().xi(), geometry.projection().fu(),
      geometry.projection().fv(), geometry.projection().cu(),
      geometry.projection().cv()};
  Eigen::MatrixXd parameters;
  geometry.projection().distortion().getParameters(parameters);
  config.distortion_coeffs.resize(
      static_cast<std::size_t>(parameters.rows()), 0.0);
  for (Eigen::Index index = 0; index < parameters.rows(); ++index) {
    config.distortion_coeffs[static_cast<std::size_t>(index)] =
        parameters(index, 0);
  }
  config.resolution = {geometry.projection().width(),
                       geometry.projection().height()};
  return config;
}

template <typename GeometryT>
StereoCameraFixedCalibration UpdateStereoCalibrationFromGeometry(
    const StereoCameraFixedCalibration& seed_calibration,
    const GeometryT& geometry) {
  const IntermediateCameraConfig config =
      MakeCameraConfigFromGeometry<GeometryT>(geometry, seed_calibration);
  StereoCameraFixedCalibration calibration = seed_calibration;
  calibration.camera_model = config.camera_model;
  calibration.distortion_model = config.distortion_model;
  calibration.camera_model_family =
      config.camera_model + "-" + config.distortion_model;
  calibration.intrinsics = config.intrinsics;
  calibration.distortion_coeffs = config.distortion_coeffs;
  calibration.resolution = config.resolution;
  return calibration;
}

bool IsStereoCalibrationModelValid(
    const StereoCameraFixedCalibration& calibration) {
  try {
    return DoubleSphereCameraModel::FromConfig(
               MakeCameraConfig(calibration))
        .IsValid();
  } catch (const std::exception&) {
    return false;
  }
}

struct Stage6PersistentResidualStats {
  int point_count = 0;
  int cam0_point_count = 0;
  int cam1_point_count = 0;
  int invalid_projection_count = 0;
  double squared_error = 0.0;
  double cam0_squared_error = 0.0;
  double cam1_squared_error = 0.0;

  double Rmse() const {
    return point_count > 0
               ? std::sqrt(squared_error / static_cast<double>(point_count))
               : std::numeric_limits<double>::infinity();
  }
  double Cam0Rmse() const {
    return cam0_point_count > 0
               ? std::sqrt(cam0_squared_error /
                           static_cast<double>(cam0_point_count))
               : std::numeric_limits<double>::infinity();
  }
  double Cam1Rmse() const {
    return cam1_point_count > 0
               ? std::sqrt(cam1_squared_error /
                           static_cast<double>(cam1_point_count))
               : std::numeric_limits<double>::infinity();
  }
};

enum class Stage6PersistentResidualMetric {
  Pixel,
  TangentPlaneAngular,
  PixelTangentHybrid,
};

Stage6PersistentResidualMetric PersistentResidualMetricForMode(
    StereoFinalBaResidualMode mode) {
  switch (mode) {
    case StereoFinalBaResidualMode::Pixel:
      return Stage6PersistentResidualMetric::Pixel;
    case StereoFinalBaResidualMode::SphericalTangent:
      return Stage6PersistentResidualMetric::TangentPlaneAngular;
    case StereoFinalBaResidualMode::HybridPixelSpherical:
      return Stage6PersistentResidualMetric::PixelTangentHybrid;
    case StereoFinalBaResidualMode::SphericalChordal:
      break;
  }
  throw std::runtime_error(
      "Stage6 persistent incremental selection does not support this residual mode.");
}

const char* ToString(Stage6PersistentResidualMetric metric) {
  switch (metric) {
    case Stage6PersistentResidualMetric::Pixel:
      return "pixel_px";
    case Stage6PersistentResidualMetric::TangentPlaneAngular:
      return "tangent_plane_rad";
    case Stage6PersistentResidualMetric::PixelTangentHybrid:
      return "pixel_tangent_px_equivalent";
  }
  return "unknown";
}

const char* PersistentResidualMetricUnits(
    Stage6PersistentResidualMetric metric) {
  switch (metric) {
    case Stage6PersistentResidualMetric::Pixel:
      return "px";
    case Stage6PersistentResidualMetric::TangentPlaneAngular:
      return "rad";
    case Stage6PersistentResidualMetric::PixelTangentHybrid:
      return "px_equivalent";
  }
  return "unknown";
}

bool UsesTangentPlaneAngularResidual(
    Stage6PersistentResidualMetric metric) {
  return metric == Stage6PersistentResidualMetric::TangentPlaneAngular ||
         metric == Stage6PersistentResidualMetric::PixelTangentHybrid;
}

bool UsesPixelResidual(Stage6PersistentResidualMetric metric) {
  return metric == Stage6PersistentResidualMetric::Pixel ||
         metric == Stage6PersistentResidualMetric::PixelTangentHybrid;
}

void FillPersistentResidualDiagnostics(
    const Stage6PersistentResidualStats& before,
    const Stage6PersistentResidualStats& after,
    StereoPairTrialSelectionDecision* decision) {
  if (decision == nullptr) {
    return;
  }
  decision->initial_total_rmse = before.Rmse();
  decision->trial_total_rmse = after.Rmse();
  decision->total_rmse_delta =
      decision->trial_total_rmse - decision->initial_total_rmse;
  decision->cam0_rmse_delta = after.Cam0Rmse() - before.Cam0Rmse();
  decision->cam1_rmse_delta = after.Cam1Rmse() - before.Cam1Rmse();
}

void FillPersistentResidualDiagnostics(
    const Stage6PersistentResidualStats& before,
    const Stage6PersistentResidualStats& after,
    StereoPairBoardTrialSelectionDecision* decision) {
  if (decision == nullptr) {
    return;
  }
  decision->initial_total_rmse = before.Rmse();
  decision->trial_total_rmse = after.Rmse();
  decision->total_rmse_delta =
      decision->trial_total_rmse - decision->initial_total_rmse;
  decision->cam0_rmse_delta = after.Cam0Rmse() - before.Cam0Rmse();
  decision->cam1_rmse_delta = after.Cam1Rmse() - before.Cam1Rmse();
}

Eigen::VectorXd MakeProjectionPriorSigmas(
    const Eigen::MatrixXd& anchor,
    const StereoExtrinsicSolverOptions& options) {
  Eigen::VectorXd sigma(anchor.size());
  sigma.setConstant(std::max(
      1e-9, options.persistent_incremental_projection_prior_shape_sigma));
  const int dimension = static_cast<int>(anchor.size());
  int focal_begin = 0;
  int principal_begin = 2;
  if (dimension == 5) {
    focal_begin = 1;
    principal_begin = 3;
  } else if (dimension == 6) {
    focal_begin = 2;
    principal_begin = 4;
  }
  const double focal_relative_sigma = std::max(
      1e-9,
      options.persistent_incremental_projection_prior_focal_relative_sigma);
  for (int index = focal_begin;
       index < std::min(dimension, focal_begin + 2); ++index) {
    sigma[index] = std::max(
        1e-6, focal_relative_sigma * std::max(1.0, std::abs(anchor(index))));
  }
  const double principal_sigma = std::max(
      1e-6, options.persistent_incremental_projection_prior_principal_sigma_px);
  for (int index = principal_begin;
       index < std::min(dimension, principal_begin + 2); ++index) {
    sigma[index] = principal_sigma;
  }
  return sigma;
}

template <typename ProjectionT>
class StereoProjectionPriorError : public aslam::backend::ErrorTermDs {
 public:
  using ProjectionDv = aslam::backend::DesignVariableAdapter<ProjectionT>;

  StereoProjectionPriorError(
      const boost::shared_ptr<ProjectionDv>& projection_dv,
      const Eigen::MatrixXd& anchor,
      const Eigen::VectorXd& sigma)
      : aslam::backend::ErrorTermDs(static_cast<int>(anchor.size())),
        projection_dv_(projection_dv),
        anchor_(Eigen::Map<const Eigen::VectorXd>(anchor.data(), anchor.size())),
        inverse_sigma_(sigma.cwiseInverse()) {
    if (!projection_dv_ || anchor_.size() <= 0 ||
        sigma.size() != anchor_.size() || !anchor_.allFinite() ||
        !sigma.allFinite() || (sigma.array() <= 0.0).any()) {
      throw std::runtime_error("Invalid Stage6 projection prior parameters.");
    }
    this->setInvR(Eigen::MatrixXd::Identity(anchor_.size(), anchor_.size()));
    this->setDesignVariables(projection_dv_.get());
  }

 protected:
  double evaluateErrorImplementation() override {
    const Eigen::MatrixXd parameters = projection_dv_->getParameters();
    if (parameters.size() != anchor_.size()) {
      throw std::runtime_error("Stage6 projection prior dimension changed.");
    }
    const Eigen::Map<const Eigen::VectorXd> current(parameters.data(),
                                                     parameters.size());
    this->setError((current - anchor_).cwiseProduct(inverse_sigma_));
    return this->evaluateChiSquaredError();
  }

  void evaluateJacobiansImplementation(
      aslam::backend::JacobianContainer& jacobians) const override {
    const Eigen::MatrixXd jacobian = inverse_sigma_.asDiagonal();
    jacobians.add(projection_dv_.get(), jacobian);
  }

 private:
  boost::shared_ptr<ProjectionDv> projection_dv_;
  Eigen::VectorXd anchor_;
  Eigen::VectorXd inverse_sigma_;
};

// Kept separate from the projection prior so that a model with no distortion
// parameters never acquires a synthetic distortion degree of freedom.
template <typename DistortionT>
class StereoDistortionPriorError : public aslam::backend::ErrorTermDs {
 public:
  using DistortionDv = aslam::backend::DesignVariableAdapter<DistortionT>;

  StereoDistortionPriorError(
      const boost::shared_ptr<DistortionDv>& distortion_dv,
      const Eigen::MatrixXd& anchor,
      const Eigen::VectorXd& sigma)
      : aslam::backend::ErrorTermDs(static_cast<int>(anchor.size())),
        distortion_dv_(distortion_dv),
        anchor_(Eigen::Map<const Eigen::VectorXd>(anchor.data(), anchor.size())),
        inverse_sigma_(sigma.cwiseInverse()) {
    if (!distortion_dv_ || anchor_.size() <= 0 ||
        sigma.size() != anchor_.size() || !anchor_.allFinite() ||
        !sigma.allFinite() || (sigma.array() <= 0.0).any()) {
      throw std::runtime_error("Invalid Stage6 distortion prior parameters.");
    }
    this->setInvR(Eigen::MatrixXd::Identity(anchor_.size(), anchor_.size()));
    this->setDesignVariables(distortion_dv_.get());
  }

 protected:
  double evaluateErrorImplementation() override {
    const Eigen::MatrixXd parameters = distortion_dv_->getParameters();
    if (parameters.size() != anchor_.size()) {
      throw std::runtime_error("Stage6 distortion prior dimension changed.");
    }
    const Eigen::Map<const Eigen::VectorXd> current(parameters.data(),
                                                     parameters.size());
    this->setError((current - anchor_).cwiseProduct(inverse_sigma_));
    return this->evaluateChiSquaredError();
  }

  void evaluateJacobiansImplementation(
      aslam::backend::JacobianContainer& jacobians) const override {
    const Eigen::MatrixXd jacobian = inverse_sigma_.asDiagonal();
    jacobians.add(distortion_dv_.get(), jacobian);
  }

 private:
  boost::shared_ptr<DistortionDv> distortion_dv_;
  Eigen::VectorXd anchor_;
  Eigen::VectorXd inverse_sigma_;
};

template <typename GeometryT0, typename GeometryT1>
class Stage6PersistentStereoProblemBuilder {
 public:
  Stage6PersistentStereoProblemBuilder(
      const StereoMeasurementDataset& dataset,
      const StereoExtrinsicSolverOptions& options,
      const StereoSceneState& seed_scene)
      : dataset_(dataset),
        options_(options),
        seed_cam0_(seed_scene.cam0),
        seed_cam1_(seed_scene.cam1),
        cam0_geometry_(MakeTypedStereoGeometry<GeometryT0>(seed_scene.cam0)),
        cam1_geometry_(MakeTypedStereoGeometry<GeometryT1>(seed_scene.cam1)),
        cam0_dv_(cam0_geometry_),
        cam1_dv_(cam1_geometry_),
        intrinsics_policy_(EvaluateStereoIntrinsicsPolicy(dataset, options)),
        gauge_fixed_board_id_(seed_scene.gauge_fixed_board_id),
        baseline_seed_(seed_scene.T_cam1_cam0) {
    cam0_dv_.setActive(false, false, false);
    cam1_dv_.setActive(false, false, false);
    cam0_geometry_->projection().getParameters(cam0_projection_seed_);
    cam1_geometry_->projection().getParameters(cam1_projection_seed_);
    cam0_geometry_->projection().distortion().getParameters(
        cam0_distortion_seed_);
    cam1_geometry_->projection().distortion().getParameters(
        cam1_distortion_seed_);
    seed_scene_pair_poses_.insert(seed_scene.T_cam0_world_by_pair.begin(),
                                  seed_scene.T_cam0_world_by_pair.end());
    InitializeStereoPoseVariable(&baseline_variable_, seed_scene.T_cam1_cam0,
                                 true);
    BuildBoardVariables(seed_scene);
  }

  struct StateSnapshot {
    Eigen::Matrix4d baseline = Eigen::Matrix4d::Identity();
    Eigen::MatrixXd cam0_projection;
    Eigen::MatrixXd cam1_projection;
    Eigen::MatrixXd cam0_distortion;
    Eigen::MatrixXd cam1_distortion;
    StereoPoseMatrixMap pair_poses;
    StereoPoseMatrixMap board_poses;
    std::map<PairBoardKey, Eigen::Matrix4d> local_pair_board_poses;
  };

  boost::shared_ptr<CalibrationBatch> BuildBatch(
      const std::set<PairBoardKey>& keys,
      bool force_add_pair_variables,
      const StateSnapshot* anchor_state,
      bool add_static_projection_prior) {
    boost::shared_ptr<CalibrationBatch> batch =
        boost::make_shared<CalibrationBatch>();
    AddCameraVariables(batch);
    MaybeAddProjectionPrior(add_static_projection_prior, batch);
    MaybeAddDistortionPrior(add_static_projection_prior, batch);
    AddBaselineVariable(batch);
    MaybeAddBaselineAnchorPrior(anchor_state, batch);
    AddBoardVariables(batch);
    AddPairVariables(keys, force_add_pair_variables, batch);
    AddResiduals(keys, batch);
    return batch;
  }

  StateSnapshot CaptureState() const {
    StateSnapshot snapshot;
    snapshot.baseline = baseline_variable_.transform.T();
    cam0_geometry_->projection().getParameters(snapshot.cam0_projection);
    cam1_geometry_->projection().getParameters(snapshot.cam1_projection);
    cam0_geometry_->projection().distortion().getParameters(
        snapshot.cam0_distortion);
    cam1_geometry_->projection().distortion().getParameters(
        snapshot.cam1_distortion);
    for (const auto& entry : pair_variables_) {
      snapshot.pair_poses[entry.first] = entry.second.transform.T();
    }
    for (const auto& entry : board_variables_) {
      snapshot.board_poses[entry.first] = entry.second.transform.T();
    }
    for (const auto& entry : local_pair_board_variables_) {
      snapshot.local_pair_board_poses[entry.first] = entry.second.transform.T();
    }
    return snapshot;
  }

  void RestoreState(const StateSnapshot& snapshot) {
    SetStereoPoseVariableFromMatrix(&baseline_variable_, snapshot.baseline);
    if (snapshot.cam0_projection.size() > 0) {
      cam0_geometry_->projection().setParameters(snapshot.cam0_projection);
    }
    if (snapshot.cam1_projection.size() > 0) {
      cam1_geometry_->projection().setParameters(snapshot.cam1_projection);
    }
    if (snapshot.cam0_distortion.size() > 0) {
      cam0_geometry_->projection().distortion().setParameters(
          snapshot.cam0_distortion);
    }
    if (snapshot.cam1_distortion.size() > 0) {
      cam1_geometry_->projection().distortion().setParameters(
          snapshot.cam1_distortion);
    }
    for (auto it = pair_variables_.begin(); it != pair_variables_.end();) {
      if (snapshot.pair_poses.count(it->first) == 0) {
        it = pair_variables_.erase(it);
      } else {
        ++it;
      }
    }
    for (const auto& entry : snapshot.pair_poses) {
      auto variable_it = pair_variables_.find(entry.first);
      if (variable_it != pair_variables_.end()) {
        SetStereoPoseVariableFromMatrix(&variable_it->second, entry.second);
      }
    }
    for (const auto& entry : snapshot.board_poses) {
      auto variable_it = board_variables_.find(entry.first);
      if (variable_it != board_variables_.end()) {
        SetStereoPoseVariableFromMatrix(&variable_it->second, entry.second);
      }
    }
    for (auto it = local_pair_board_variables_.begin();
         it != local_pair_board_variables_.end();) {
      if (snapshot.local_pair_board_poses.count(it->first) == 0) {
        it = local_pair_board_variables_.erase(it);
      } else {
        ++it;
      }
    }
    for (const auto& entry : snapshot.local_pair_board_poses) {
      auto variable_it = local_pair_board_variables_.find(entry.first);
      if (variable_it != local_pair_board_variables_.end()) {
        SetStereoPoseVariableFromMatrix(&variable_it->second, entry.second);
      }
    }
  }

  bool CurrentStateFinite(std::string* reason) const {
    const StereoCameraFixedCalibration current_cam0 =
        UpdateStereoCalibrationFromGeometry(seed_cam0_, *cam0_geometry_);
    const StereoCameraFixedCalibration current_cam1 =
        UpdateStereoCalibrationFromGeometry(seed_cam1_, *cam1_geometry_);
    if (!IsStereoCalibrationModelValid(current_cam0) ||
        !IsStereoCalibrationModelValid(current_cam1)) {
      if (reason != nullptr) {
        *reason = "invalid_camera_projection_parameters";
      }
      return false;
    }
    if (!IsStereoPoseVariableFinite(baseline_variable_)) {
      if (reason != nullptr) {
        *reason = "nonfinite_stereo_baseline";
      }
      return false;
    }
    for (const auto& entry : pair_variables_) {
      if (!IsStereoPoseVariableFinite(entry.second)) {
        if (reason != nullptr) {
          *reason = "nonfinite_pair_pose_" + std::to_string(entry.first);
        }
        return false;
      }
    }
    for (const auto& entry : board_variables_) {
      if (!IsStereoPoseVariableFinite(entry.second)) {
        if (reason != nullptr) {
          *reason = "nonfinite_board_pose_" + std::to_string(entry.first);
        }
        return false;
      }
    }
    for (const auto& entry : local_pair_board_variables_) {
      if (!IsStereoPoseVariableFinite(entry.second)) {
        if (reason != nullptr) {
          *reason = "nonfinite_local_pair_board_pose_" +
                    std::to_string(entry.first.first) + "_" +
                    std::to_string(entry.first.second);
        }
        return false;
      }
    }
    if (reason != nullptr) {
      reason->clear();
    }
    return true;
  }

  StereoSceneState BuildSceneState(const StereoSceneState& scene_template) const {
    StereoSceneState scene = scene_template;
    scene.cam0 = UpdateStereoCalibrationFromGeometry(
        scene_template.cam0, *cam0_geometry_);
    scene.cam1 = UpdateStereoCalibrationFromGeometry(
        scene_template.cam1, *cam1_geometry_);
    scene.T_cam1_cam0 = baseline_variable_.transform.T();
    for (const auto& entry : pair_variables_) {
      scene.T_cam0_world_by_pair[entry.first] = entry.second.transform.T();
    }
    for (const auto& entry : board_variables_) {
      scene.T_world_board_by_id[entry.first] = entry.second.transform.T();
    }
    scene.T_cam0_board_by_pair_board.clear();
    if (options_.persistent_pose_structure ==
        StereoPersistentPoseStructure::IndependentPairBoard) {
      for (const auto& entry : local_pair_board_variables_) {
        scene.T_cam0_board_by_pair_board[entry.first] =
            entry.second.transform.T();
      }
    }
    scene.success = true;
    scene.failure_reason.clear();
    return scene;
  }

  Stage6PersistentResidualStats EvaluateAccepted(
      const std::set<PairBoardKey>& accepted_keys,
      Stage6PersistentResidualMetric metric) const {
    Stage6PersistentResidualStats stats;
    const bool use_pixel = UsesPixelResidual(metric);
    const bool use_tangent = UsesTangentPlaneAngularResidual(metric);
    DoubleSphereCameraModel cam0_residual_camera;
    DoubleSphereCameraModel cam1_residual_camera;
    if (use_tangent) {
      try {
        cam0_residual_camera = DoubleSphereCameraModel::FromConfig(
            MakeCameraConfigFromGeometry(*cam0_geometry_, seed_cam0_));
        cam1_residual_camera = DoubleSphereCameraModel::FromConfig(
            MakeCameraConfigFromGeometry(*cam1_geometry_, seed_cam1_));
      } catch (const std::exception&) {
        stats.invalid_projection_count =
            std::numeric_limits<int>::max();
        return stats;
      }
    }
    const double focal_reference_px = std::sqrt(
        std::max(1.0, std::abs(seed_cam0_.intrinsics.size() > 3
                                    ? seed_cam0_.intrinsics[2]
                                    : 1.0)) *
        std::max(1.0, std::abs(seed_cam0_.intrinsics.size() > 3
                                    ? seed_cam0_.intrinsics[3]
                                    : 1.0)));
    const Eigen::Matrix4d T_cam1_cam0 = baseline_variable_.transform.T();
    for (const StereoObservation& observation : dataset_.observations) {
      if (!observation.used_in_solver ||
          accepted_keys.count(PairBoardKey(observation.pair_index,
                                           observation.board_id)) == 0) {
        continue;
      }
      Eigen::Matrix4d T_cam0_board = Eigen::Matrix4d::Identity();
      if (options_.persistent_pose_structure ==
          StereoPersistentPoseStructure::IndependentPairBoard) {
        const StereoPoseVariableState* local_variable =
            FindLocalPairBoardVariable(
                PairBoardKey(observation.pair_index, observation.board_id));
        if (local_variable == nullptr) {
          ++stats.invalid_projection_count;
          continue;
        }
        T_cam0_board = local_variable->transform.T();
      } else {
        const StereoPoseVariableState* pair_variable =
            FindPairVariable(observation.pair_index);
        if (pair_variable == nullptr) {
          ++stats.invalid_projection_count;
          continue;
        }
        Eigen::Matrix4d T_world_board = Eigen::Matrix4d::Identity();
        if (observation.board_id != gauge_fixed_board_id_) {
          const StereoPoseVariableState* board_variable =
              FindBoardVariable(observation.board_id);
          if (board_variable == nullptr) {
            ++stats.invalid_projection_count;
            continue;
          }
          T_world_board = board_variable->transform.T();
        }
        T_cam0_board = pair_variable->transform.T() * T_world_board;
      }
      const Eigen::Vector4d point_board(observation.target_point_board.x(),
                                        observation.target_point_board.y(),
                                        observation.target_point_board.z(),
                                        1.0);
      Eigen::Vector4d point_camera = T_cam0_board * point_board;
      if (observation.camera_index == 1) {
        point_camera = T_cam1_cam0 * point_camera;
      }
      double squared_error = 0.0;
      if (use_pixel) {
        Eigen::Vector2d predicted = Eigen::Vector2d::Zero();
        const bool ok =
            observation.camera_index == 0
                ? cam0_geometry_->homogeneousToKeypoint(point_camera, predicted)
                : cam1_geometry_->homogeneousToKeypoint(point_camera, predicted);
        if (!ok || !predicted.allFinite()) {
          ++stats.invalid_projection_count;
          continue;
        }
        squared_error +=
            (predicted - observation.observed_image_xy).squaredNorm();
      }
      if (use_tangent) {
        AngularObservationGeometry observation_geometry;
        const DoubleSphereCameraModel& residual_camera =
            observation.camera_index == 0 ? cam0_residual_camera
                                          : cam1_residual_camera;
        const double point_norm = point_camera.head<3>().norm();
        if (!(point_norm > 1e-12) || !std::isfinite(point_norm) ||
            !ComputeAngularObservationGeometry(
                residual_camera, observation.observed_image_xy,
                &observation_geometry)) {
          ++stats.invalid_projection_count;
          continue;
        }
        const Eigen::Vector3d predicted_ray =
            point_camera.head<3>() / point_norm;
        const double tangent_squared_error =
            (observation_geometry.tangent_basis.transpose() *
             (predicted_ray - observation_geometry.observed_ray))
                .squaredNorm();
        if (!std::isfinite(tangent_squared_error)) {
          ++stats.invalid_projection_count;
          continue;
        }
        // Hybrid diagnostics are reported in a fixed focal-length-equivalent
        // scale. Pure tangent diagnostics remain in radians.
        squared_error += metric == Stage6PersistentResidualMetric::
                                     PixelTangentHybrid
                             ? focal_reference_px * focal_reference_px *
                                   tangent_squared_error
                             : tangent_squared_error;
      }
      ++stats.point_count;
      stats.squared_error += squared_error;
      if (observation.camera_index == 0) {
        ++stats.cam0_point_count;
        stats.cam0_squared_error += squared_error;
      } else {
        ++stats.cam1_point_count;
        stats.cam1_squared_error += squared_error;
      }
    }
    return stats;
  }

 private:
  void AddCameraVariables(const boost::shared_ptr<CalibrationBatch>& batch) {
    const bool optimize_projection = intrinsics_policy_.projection_active;
    constexpr bool kCam0HasDistortion =
        GeometryT0::projection_t::distortion_t::DesignVariableDimension > 0;
    constexpr bool kCam1HasDistortion =
        GeometryT1::projection_t::distortion_t::DesignVariableDimension > 0;
    const bool optimize_cam0_distortion =
        intrinsics_policy_.distortion_active && kCam0HasDistortion;
    const bool optimize_cam1_distortion =
        intrinsics_policy_.distortion_active && kCam1HasDistortion;
    cam0_dv_.setActive(optimize_projection, optimize_cam0_distortion, false);
    cam1_dv_.setActive(optimize_projection, optimize_cam1_distortion, false);
    const std::size_t group_id =
        (optimize_projection || optimize_cam0_distortion ||
         optimize_cam1_distortion)
            ? kStage6StereoExtrinsicInformationGroupId
            : kStage6PairPoseGroupId;
    batch->addDesignVariable(cam0_dv_.projectionDesignVariable(),
                             group_id);
    batch->addDesignVariable(cam0_dv_.distortionDesignVariable(),
                             optimize_cam0_distortion
                                 ? kStage6StereoExtrinsicInformationGroupId
                                 : kStage6PairPoseGroupId);
    batch->addDesignVariable(cam0_dv_.shutterDesignVariable(),
                             kStage6PairPoseGroupId);
    batch->addDesignVariable(cam1_dv_.projectionDesignVariable(),
                             group_id);
    batch->addDesignVariable(cam1_dv_.distortionDesignVariable(),
                             optimize_cam1_distortion
                                 ? kStage6StereoExtrinsicInformationGroupId
                                 : kStage6PairPoseGroupId);
    batch->addDesignVariable(cam1_dv_.shutterDesignVariable(),
                             kStage6PairPoseGroupId);
  }

  void MaybeAddProjectionPrior(
      bool add_static_projection_prior,
      const boost::shared_ptr<CalibrationBatch>& batch) {
    if (!add_static_projection_prior ||
        !intrinsics_policy_.projection_prior_enabled) {
      return;
    }
    const Eigen::VectorXd cam0_sigma =
        MakeProjectionPriorSigmas(cam0_projection_seed_, options_);
    const Eigen::VectorXd cam1_sigma =
        MakeProjectionPriorSigmas(cam1_projection_seed_, options_);
    using ProjectionT0 = typename GeometryT0::projection_t;
    using ProjectionT1 = typename GeometryT1::projection_t;
    batch->addErrorTerm(boost::shared_ptr<aslam::backend::ErrorTerm>(
        new StereoProjectionPriorError<ProjectionT0>(
            cam0_dv_.projectionDesignVariable(), cam0_projection_seed_,
            cam0_sigma)));
    batch->addErrorTerm(boost::shared_ptr<aslam::backend::ErrorTerm>(
        new StereoProjectionPriorError<ProjectionT1>(
            cam1_dv_.projectionDesignVariable(), cam1_projection_seed_,
            cam1_sigma)));
  }

  template <typename GeometryT>
  void MaybeAddDistortionPriorForCamera(
      CameraDv<GeometryT>* camera_dv,
      const Eigen::MatrixXd& anchor,
      const boost::shared_ptr<CalibrationBatch>& batch,
      std::true_type) {
    if (camera_dv == nullptr || anchor.size() <= 0 || !anchor.allFinite()) {
      return;
    }
    const double sigma_value = std::max(
        1e-9, options_.persistent_incremental_distortion_prior_sigma);
    const Eigen::VectorXd sigma =
        Eigen::VectorXd::Constant(anchor.size(), sigma_value);
    using DistortionT = typename GeometryT::projection_t::distortion_t;
    batch->addErrorTerm(boost::shared_ptr<aslam::backend::ErrorTerm>(
        new StereoDistortionPriorError<DistortionT>(
            camera_dv->distortionDesignVariable(), anchor, sigma)));
  }

  template <typename GeometryT>
  void MaybeAddDistortionPriorForCamera(
      CameraDv<GeometryT>* /*camera_dv*/,
      const Eigen::MatrixXd& /*anchor*/,
      const boost::shared_ptr<CalibrationBatch>& /*batch*/,
      std::false_type) {}

  void MaybeAddDistortionPrior(
      bool add_static_distortion_prior,
      const boost::shared_ptr<CalibrationBatch>& batch) {
    if (!add_static_distortion_prior ||
        !intrinsics_policy_.distortion_prior_enabled) {
      return;
    }
    using Cam0DistortionT = typename GeometryT0::projection_t::distortion_t;
    using Cam1DistortionT = typename GeometryT1::projection_t::distortion_t;
    MaybeAddDistortionPriorForCamera<GeometryT0>(
        &cam0_dv_,
        cam0_distortion_seed_, batch,
        std::integral_constant<bool,
                               (Cam0DistortionT::DesignVariableDimension > 0)>());
    MaybeAddDistortionPriorForCamera<GeometryT1>(
        &cam1_dv_,
        cam1_distortion_seed_, batch,
        std::integral_constant<bool,
                               (Cam1DistortionT::DesignVariableDimension > 0)>());
  }

  void AddBaselineVariable(const boost::shared_ptr<CalibrationBatch>& batch) {
    baseline_variable_.rotation_dv->setActive(true);
    baseline_variable_.translation_dv->setActive(true);
    AddStereoPoseVariableDvs(baseline_variable_,
                             kStage6StereoExtrinsicInformationGroupId, batch);
  }

  void MaybeAddBaselineAnchorPrior(
      const StateSnapshot* anchor_state,
      const boost::shared_ptr<CalibrationBatch>& batch) {
    const double translation_weight = std::max(
        0.0, options_.persistent_incremental_baseline_prior_translation_weight);
    const double rotation_weight = std::max(
        0.0, options_.persistent_incremental_baseline_prior_rotation_weight);
    if (translation_weight <= 0.0 && rotation_weight <= 0.0) {
      return;
    }
    Eigen::Matrix4d anchor = baseline_seed_;
    if (anchor_state != nullptr) {
      anchor = anchor_state->baseline;
    }
    batch->addErrorTerm(boost::shared_ptr<aslam::backend::ErrorTerm>(
        new aslam::backend::ErrorTermTransformation(
            baseline_variable_.expression, sm::kinematics::Transformation(anchor),
            rotation_weight, translation_weight)));
  }

  void BuildBoardVariables(const StereoSceneState& seed_scene) {
    for (const auto& entry : seed_scene.T_world_board_by_id) {
      if (entry.first == gauge_fixed_board_id_) {
        continue;
      }
      StereoPoseVariableState& variable =
          GetOrCreateStereoPoseVariable(&board_variables_, entry.first);
      InitializeStereoPoseVariable(&variable, entry.second, true);
    }
  }

  void AddBoardVariables(const boost::shared_ptr<CalibrationBatch>& batch) {
    if (options_.persistent_pose_structure ==
        StereoPersistentPoseStructure::IndependentPairBoard) {
      return;
    }
    for (auto& entry : board_variables_) {
      entry.second.rotation_dv->setActive(true);
      entry.second.translation_dv->setActive(true);
      AddStereoPoseVariableDvs(entry.second, kStage6BoardLayoutGroupId, batch);
    }
  }

  void AddPairVariables(const std::set<PairBoardKey>& keys,
                        bool force_add_pair_variables,
                        const boost::shared_ptr<CalibrationBatch>& batch) {
    if (options_.persistent_pose_structure ==
        StereoPersistentPoseStructure::IndependentPairBoard) {
      AddLocalPairBoardVariables(keys, force_add_pair_variables, batch);
      return;
    }
    std::set<int> pairs;
    for (const PairBoardKey& key : keys) {
      pairs.insert(key.first);
    }
    for (int pair_index : pairs) {
      auto pair_it = pair_variables_.find(pair_index);
      if (pair_it == pair_variables_.end()) {
        const auto pose_it = seed_scene_pair_poses_.find(pair_index);
        if (pose_it == seed_scene_pair_poses_.end()) {
          throw std::runtime_error(
              "Stage6 persistent estimator missing initialized pair pose.");
        }
        StereoPoseVariableState& variable =
            GetOrCreateStereoPoseVariable(&pair_variables_, pair_index);
        InitializeStereoPoseVariable(&variable, pose_it->second, true);
        pair_it = pair_variables_.find(pair_index);
      } else if (!force_add_pair_variables) {
        continue;
      }
      pair_it->second.rotation_dv->setActive(true);
      pair_it->second.translation_dv->setActive(true);
      AddStereoPoseVariableDvs(pair_it->second, kStage6PairPoseGroupId, batch);
    }
  }

  void AddLocalPairBoardVariables(
      const std::set<PairBoardKey>& keys,
      bool force_add_variables,
      const boost::shared_ptr<CalibrationBatch>& batch) {
    for (const PairBoardKey& key : keys) {
      auto variable_it = local_pair_board_variables_.find(key);
      if (variable_it == local_pair_board_variables_.end()) {
        const auto pair_pose_it = seed_scene_pair_poses_.find(key.first);
        if (pair_pose_it == seed_scene_pair_poses_.end()) {
          throw std::runtime_error(
              "Stage6 persistent estimator missing pair pose for local board.");
        }
        Eigen::Matrix4d T_world_board = Eigen::Matrix4d::Identity();
        if (key.second != gauge_fixed_board_id_) {
          const auto board_pose_it = board_variables_.find(key.second);
          if (board_pose_it == board_variables_.end()) {
            throw std::runtime_error(
                "Stage6 persistent estimator missing board pose for local board.");
          }
          T_world_board = board_pose_it->second.transform.T();
        }
        StereoPoseVariableState& variable = local_pair_board_variables_[key];
        InitializeStereoPoseVariable(
            &variable, pair_pose_it->second * T_world_board, true);
        variable_it = local_pair_board_variables_.find(key);
      } else if (!force_add_variables) {
        continue;
      }
      variable_it->second.rotation_dv->setActive(true);
      variable_it->second.translation_dv->setActive(true);
      AddStereoPoseVariableDvs(variable_it->second, kStage6PairPoseGroupId,
                               batch);
    }
  }

  void AddResiduals(const std::set<PairBoardKey>& keys,
                    const boost::shared_ptr<CalibrationBatch>& batch) {
    const Stage6PersistentResidualMetric residual_metric =
        PersistentResidualMetricForMode(options_.selection_ba_residual_mode);
    const bool use_pixel = UsesPixelResidual(residual_metric);
    const bool use_tangent = UsesTangentPlaneAngularResidual(residual_metric);
    const bool dynamic_observed_ray =
        intrinsics_policy_.projection_active || intrinsics_policy_.distortion_active;
    DoubleSphereCameraModel cam0_residual_camera;
    DoubleSphereCameraModel cam1_residual_camera;
    if (use_tangent) {
      cam0_residual_camera = DoubleSphereCameraModel::FromConfig(
          MakeCameraConfigFromGeometry(*cam0_geometry_, seed_cam0_));
      cam1_residual_camera = DoubleSphereCameraModel::FromConfig(
          MakeCameraConfigFromGeometry(*cam1_geometry_, seed_cam1_));
    }
    const aslam::backend::TransformationExpression identity_transform(
        Eigen::Matrix4d::Identity());
    for (const StereoObservation& observation : dataset_.observations) {
      if (!observation.used_in_solver ||
          keys.count(PairBoardKey(observation.pair_index,
                                  observation.board_id)) == 0) {
        continue;
      }
      aslam::backend::TransformationExpression T_cam0_board_expression =
          identity_transform;
      if (options_.persistent_pose_structure ==
          StereoPersistentPoseStructure::IndependentPairBoard) {
        const StereoPoseVariableState* local_variable =
            FindLocalPairBoardVariable(
                PairBoardKey(observation.pair_index, observation.board_id));
        if (local_variable == nullptr) {
          continue;
        }
        T_cam0_board_expression = local_variable->expression;
      } else {
        const StereoPoseVariableState* pair_variable =
            FindPairVariable(observation.pair_index);
        if (pair_variable == nullptr) {
          continue;
        }
        aslam::backend::TransformationExpression board_expression =
            identity_transform;
        if (observation.board_id != gauge_fixed_board_id_) {
          const StereoPoseVariableState* board_variable =
              FindBoardVariable(observation.board_id);
          if (board_variable == nullptr) {
            continue;
          }
          board_expression = board_variable->expression;
        }
        T_cam0_board_expression =
            pair_variable->expression * board_expression;
      }
      const aslam::backend::HomogeneousExpression point_board(
          observation.target_point_board);
      aslam::backend::HomogeneousExpression point_camera =
          T_cam0_board_expression * point_board;
      if (observation.camera_index == 1) {
        point_camera = baseline_variable_.expression * point_camera;
      }
      const double weight = std::max(0.0, observation.weight) *
                            options_.ba_shared_observation_weight_scale;
      if (weight <= 0.0) {
        continue;
      }
      if (observation.camera_index == 0) {
        if (use_pixel) {
        batch->addErrorTerm(boost::shared_ptr<aslam::backend::ErrorTerm>(
            new StereoCameraReprojectionError<GeometryT0>(
                observation.observed_image_xy, weight, point_camera, cam0_dv_,
                options_.persistent_incremental_invalid_projection_penalty_px)));
        }
        if (use_tangent) {
          AngularObservationGeometry observation_geometry;
          if (!ComputeAngularObservationGeometry(
                  cam0_residual_camera, observation.observed_image_xy,
                  &observation_geometry)) {
            continue;
          }
          double angular_weight =
              std::max(0.0, options_.spherical_weight) * std::sqrt(weight);
          if (options_.spherical_polar_weighting) {
            angular_weight *= ComputePolarContinuousAngularWeight(
                observation_geometry.polar_angle_deg,
                options_.spherical_min_polar_deg, 5.0);
          }
          if (angular_weight <= 0.0) {
            continue;
          }
          batch->addErrorTerm(boost::shared_ptr<aslam::backend::ErrorTerm>(
              new StereoCameraAngularReprojectionError<GeometryT0>(
                  observation_geometry, observation.observed_image_xy,
                  dynamic_observed_ray, seed_cam0_,
                  options_.spherical_use_normalize_jacobian, angular_weight,
                  point_camera, cam0_dv_, 0.35)));
        }
      } else {
        if (use_pixel) {
        batch->addErrorTerm(boost::shared_ptr<aslam::backend::ErrorTerm>(
            new StereoCameraReprojectionError<GeometryT1>(
                observation.observed_image_xy, weight, point_camera, cam1_dv_,
                options_.persistent_incremental_invalid_projection_penalty_px)));
        }
        if (use_tangent) {
          AngularObservationGeometry observation_geometry;
          if (!ComputeAngularObservationGeometry(
                  cam1_residual_camera, observation.observed_image_xy,
                  &observation_geometry)) {
            continue;
          }
          double angular_weight =
              std::max(0.0, options_.spherical_weight) * std::sqrt(weight);
          if (options_.spherical_polar_weighting) {
            angular_weight *= ComputePolarContinuousAngularWeight(
                observation_geometry.polar_angle_deg,
                options_.spherical_min_polar_deg, 5.0);
          }
          if (angular_weight <= 0.0) {
            continue;
          }
          batch->addErrorTerm(boost::shared_ptr<aslam::backend::ErrorTerm>(
              new StereoCameraAngularReprojectionError<GeometryT1>(
                  observation_geometry, observation.observed_image_xy,
                  dynamic_observed_ray, seed_cam1_,
                  options_.spherical_use_normalize_jacobian, angular_weight,
                  point_camera, cam1_dv_, 0.35)));
        }
      }
    }
  }

  const StereoPoseVariableState* FindPairVariable(int pair_index) const {
    const auto it = pair_variables_.find(pair_index);
    return it == pair_variables_.end() ? nullptr : &it->second;
  }

  const StereoPoseVariableState* FindBoardVariable(int board_id) const {
    const auto it = board_variables_.find(board_id);
    return it == board_variables_.end() ? nullptr : &it->second;
  }

  const StereoPoseVariableState* FindLocalPairBoardVariable(
      const PairBoardKey& key) const {
    const auto it = local_pair_board_variables_.find(key);
    return it == local_pair_board_variables_.end() ? nullptr : &it->second;
  }

  const StereoMeasurementDataset& dataset_;
  const StereoExtrinsicSolverOptions& options_;
  StereoCameraFixedCalibration seed_cam0_;
  StereoCameraFixedCalibration seed_cam1_;
  boost::shared_ptr<GeometryT0> cam0_geometry_;
  boost::shared_ptr<GeometryT1> cam1_geometry_;
  CameraDv<GeometryT0> cam0_dv_;
  CameraDv<GeometryT1> cam1_dv_;
  StereoIntrinsicsPolicyDecision intrinsics_policy_;
  Eigen::MatrixXd cam0_projection_seed_;
  Eigen::MatrixXd cam1_projection_seed_;
  Eigen::MatrixXd cam0_distortion_seed_;
  Eigen::MatrixXd cam1_distortion_seed_;
  int gauge_fixed_board_id_ = 1;
  Eigen::Matrix4d baseline_seed_ = Eigen::Matrix4d::Identity();
  StereoPoseVariableState baseline_variable_;
  StereoPoseVariableMap pair_variables_;
  StereoPoseVariableMap board_variables_;
  std::map<PairBoardKey, StereoPoseVariableState> local_pair_board_variables_;
  StereoPoseMatrixMap seed_scene_pair_poses_;
};

double EvaluateTotalProblemObjective(aslam::backend::OptimizationProblem* problem) {
  if (problem == nullptr) {
    throw std::runtime_error("EvaluateTotalProblemObjective requires a valid problem.");
  }
  double total_cost = 0.0;
  for (std::size_t index = 0; index < problem->numErrorTerms(); ++index) {
    total_cost += problem->errorTerm(index)->evaluateError();
  }
  return total_cost;
}

double DegreesToRadians(double degrees) {
  return degrees * M_PI / 180.0;
}

bool CandidateConsistent(const Eigen::Isometry3d& lhs,
                         const Eigen::Isometry3d& rhs,
                         double max_rotation_radians,
                         double max_translation_m) {
  return RotationDistanceRadians(lhs, rhs) <= max_rotation_radians &&
         (lhs.translation() - rhs.translation()).norm() <= max_translation_m;
}

std::vector<TransformCandidate> FilterConsistentCandidates(
    const std::vector<TransformCandidate>& candidates,
    double max_rotation_radians,
    double max_translation_m,
    int* rejected_count) {
  if (rejected_count != nullptr) {
    *rejected_count = 0;
  }
  if (candidates.size() <= 1) {
    return candidates;
  }

  int best_index = -1;
  int best_support = -1;
  double best_score = std::numeric_limits<double>::infinity();
  for (std::size_t i = 0; i < candidates.size(); ++i) {
    int support = 0;
    double score = 0.0;
    for (std::size_t j = 0; j < candidates.size(); ++j) {
      if (CandidateConsistent(candidates[i].transform, candidates[j].transform,
                              max_rotation_radians, max_translation_m)) {
        ++support;
        score += TransformDistanceScore(candidates[i].transform,
                                        candidates[j].transform);
      }
    }
    if (support > best_support || (support == best_support && score < best_score)) {
      best_support = support;
      best_score = score;
      best_index = static_cast<int>(i);
    }
  }

  if (best_index < 0 || best_support <= 0) {
    if (rejected_count != nullptr) {
      *rejected_count = static_cast<int>(candidates.size());
    }
    return std::vector<TransformCandidate>();
  }

  std::vector<TransformCandidate> filtered;
  for (const TransformCandidate& candidate : candidates) {
    if (CandidateConsistent(candidate.transform,
                            candidates[static_cast<std::size_t>(best_index)].transform,
                            max_rotation_radians, max_translation_m)) {
      filtered.push_back(candidate);
    }
  }
  if (rejected_count != nullptr) {
    *rejected_count =
        static_cast<int>(candidates.size() - filtered.size());
  }
  return filtered;
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

const char* StereoPairPoseRefitModeToString(StereoPairPoseRefitMode mode) {
  switch (mode) {
    case StereoPairPoseRefitMode::Cam0Only:
      return "cam0_only";
    case StereoPairPoseRefitMode::StereoSymmetric:
      return "stereo_symmetric";
  }
  return "unknown";
}

void DrawStereoVisualizationLegend(cv::Mat* image,
                                   const std::string& status,
                                   const std::string& extra_line) {
  if (image == nullptr || image->empty()) {
    return;
  }
  const int x = 18;
  int y = 58;
  const double scale = 0.48;
  const int thickness = 1;
  cv::rectangle(*image, cv::Rect(10, 38, 760, 140), cv::Scalar(0, 0, 0), -1);
  cv::rectangle(*image, cv::Rect(10, 38, 760, 140), cv::Scalar(80, 80, 80), 1);
  cv::putText(*image, "status: " + status, cv::Point(x, y),
              cv::FONT_HERSHEY_SIMPLEX, scale, cv::Scalar(255, 255, 255),
              thickness);
  y += 20;
  cv::putText(*image,
              "OBS outer: cyan filled circle | OBS internal: green cross",
              cv::Point(x, y), cv::FONT_HERSHEY_SIMPLEX, scale,
              cv::Scalar(255, 255, 255), thickness);
  y += 20;
  cv::putText(*image,
              "PROJ outer: magenta square | PROJ internal: red tilted cross",
              cv::Point(x, y), cv::FONT_HERSHEY_SIMPLEX, scale,
              cv::Scalar(255, 255, 255), thickness);
  y += 20;
  cv::putText(*image, "yellow arrow = residual vector from observed to projected",
              cv::Point(x, y), cv::FONT_HERSHEY_SIMPLEX, scale,
              cv::Scalar(0, 255, 255), thickness);
  y += 20;
  cv::putText(*image,
              "outer subpix window: blue box | close-edge boosted: orange box",
              cv::Point(x, y), cv::FONT_HERSHEY_SIMPLEX, scale,
              cv::Scalar(255, 255, 255), thickness);
  if (!extra_line.empty()) {
    y += 20;
    cv::putText(*image, extra_line, cv::Point(x, y),
                cv::FONT_HERSHEY_SIMPLEX, scale, cv::Scalar(210, 210, 255),
                thickness);
  }
}

void DrawOuterSubpixWindow(cv::Mat* image,
                           const StereoObservation& observation,
                           const cv::Point& observed_pt) {
  if (image == nullptr || image->empty() ||
      observation.point_type != JointPointType::Outer ||
      observation.outer_subpix_window_radius <= 0) {
    return;
  }
  const int radius = std::max(1, observation.outer_subpix_window_radius);
  const int x0 = std::max(0, observed_pt.x - radius);
  const int y0 = std::max(0, observed_pt.y - radius);
  const int x1 = std::min(image->cols - 1, observed_pt.x + radius);
  const int y1 = std::min(image->rows - 1, observed_pt.y + radius);
  if (x1 <= x0 || y1 <= y0) {
    return;
  }
  const cv::Scalar normal_color(255, 128, 0);
  const cv::Scalar boosted_color(0, 165, 255);
  const cv::Scalar color = observation.outer_close_edge_subpix_boost_applied
                               ? boosted_color
                               : normal_color;
  const int thickness = observation.outer_close_edge_subpix_boost_applied ? 3 : 1;
  cv::rectangle(*image, cv::Rect(cv::Point(x0, y0), cv::Point(x1, y1)),
                color, thickness, cv::LINE_AA);
  std::ostringstream label;
  label << "r=" << observation.outer_subpix_window_radius;
  if (observation.outer_close_edge_subpix_boost_applied) {
    label << " boost";
  }
  cv::putText(*image, label.str(), cv::Point(x0, std::max(12, y0 - 4)),
              cv::FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv::LINE_AA);
}

std::string SafeVisualizationToken(const std::string& text) {
  std::string token;
  token.reserve(text.size());
  for (const unsigned char ch : text) {
    if (std::isalnum(ch)) {
      token.push_back(static_cast<char>(ch));
    } else if (ch == '_' || ch == '-' || ch == '.') {
      token.push_back(static_cast<char>(ch));
    } else {
      token.push_back('_');
    }
  }
  if (token.empty()) {
    return "unknown";
  }
  if (token.size() > 64) {
    token.resize(64);
  }
  return token;
}

std::string MakeStereoVisualizationPrefix(int rank,
                                          const StereoPairResidualSummary& pair_summary,
                                          const StereoFramePair& pair) {
  std::ostringstream stream;
  stream << "rank_" << std::setw(3) << std::setfill('0') << rank
         << "_pair_" << pair_summary.pair_index
         << "_" << SafeVisualizationToken(pair.left_frame_label)
         << "_" << SafeVisualizationToken(pair.right_frame_label);
  return stream.str();
}

void DrawStereoResidualObservation(cv::Mat* image,
                                   const cv::Point& observed_pt,
                                   const cv::Point& predicted_pt,
                                   bool is_internal) {
  if (image == nullptr || image->empty()) {
    return;
  }
  const int marker_size = 8;
  const int marker_thickness = 1;
  const int line_thickness = 1;
  const int outer_radius = is_internal ? 3 : 4;
  const int box_radius = 5;

  const auto inside_image = [&](const cv::Point& pt) {
    return pt.x >= 0 && pt.y >= 0 && pt.x < image->cols && pt.y < image->rows;
  };
  const auto clamp_to_image = [&](const cv::Point& pt) {
    return cv::Point(std::min(std::max(pt.x, 0), image->cols - 1),
                     std::min(std::max(pt.y, 0), image->rows - 1));
  };

  const cv::Point clipped_observed = clamp_to_image(observed_pt);
  const cv::Point clipped_predicted = clamp_to_image(predicted_pt);
  const bool predicted_inside = inside_image(predicted_pt);
  const cv::Scalar residual_color(0, 255, 255);
  const cv::Scalar observed_outer_color(255, 255, 0);
  const cv::Scalar observed_internal_color(0, 255, 0);
  const cv::Scalar projected_outer_color(255, 0, 255);
  const cv::Scalar projected_internal_color(0, 0, 255);

  cv::arrowedLine(*image, clipped_observed, clipped_predicted, residual_color,
                  line_thickness,
                  cv::LINE_AA, 0, 0.18);
  if (is_internal) {
    cv::circle(*image, clipped_observed, outer_radius,
               observed_internal_color, 2, cv::LINE_AA);
    cv::drawMarker(*image, clipped_predicted, projected_internal_color,
                   cv::MARKER_CROSS, marker_size, marker_thickness,
                   cv::LINE_AA);
  } else {
    cv::circle(*image, clipped_observed, outer_radius,
               observed_outer_color, 2, cv::LINE_AA);
    cv::rectangle(*image,
                  cv::Rect(clipped_predicted.x - box_radius, clipped_predicted.y - box_radius,
                           box_radius * 2, box_radius * 2),
                  projected_outer_color, 1, cv::LINE_AA);
  }
  if (!predicted_inside) {
    cv::putText(*image, "PROJ_OUT_OF_IMAGE", clipped_predicted + cv::Point(10, -10),
                cv::FONT_HERSHEY_SIMPLEX, 0.45, projected_internal_color,
                line_thickness, cv::LINE_AA);
  }
}

struct StereoBackendBoardSource {
  bool has_decision = false;
  bool seed = false;
  bool trial_accepted = false;
  bool pair_cohesion_accepted = false;
  bool cap_gate_relaxed = false;
};

std::string StereoBackendBoardSourceLabel(
    const StereoBackendBoardSource& source) {
  if (source.seed) {
    return "seed";
  }
  if (source.pair_cohesion_accepted) {
    return source.cap_gate_relaxed ? "cohesion_cap" : "cohesion";
  }
  if (source.trial_accepted) {
    return "trial";
  }
  return source.has_decision ? "decision" : "unknown";
}

cv::Scalar StereoBackendBoardSourceColor(
    const StereoBackendBoardSource& source) {
  if (source.seed) {
    return cv::Scalar(60, 230, 60);
  }
  if (source.pair_cohesion_accepted) {
    return cv::Scalar(255, 0, 255);
  }
  if (source.trial_accepted) {
    return cv::Scalar(0, 165, 255);
  }
  return cv::Scalar(180, 180, 180);
}

std::map<PairBoardKey, StereoBackendBoardSource> BuildStereoBackendBoardSources(
    const StereoPairBoardTrialSelectionSummary& selection_summary) {
  std::map<PairBoardKey, StereoBackendBoardSource> sources;
  for (const StereoPairBoardTrialSelectionDecision& decision :
       selection_summary.decisions) {
    if (!decision.accepted && !decision.seed) {
      continue;
    }
    const PairBoardKey key(decision.pair_index, decision.board_id);
    StereoBackendBoardSource& source = sources[key];
    source.has_decision = true;
    source.seed = source.seed || decision.seed;
    source.pair_cohesion_accepted =
        source.pair_cohesion_accepted ||
        (decision.pair_cohesion_candidate && decision.accepted);
    source.trial_accepted =
        source.trial_accepted ||
        (!decision.seed && !decision.pair_cohesion_candidate &&
         decision.accepted);
    source.cap_gate_relaxed =
        source.cap_gate_relaxed || decision.pair_cohesion_cap_gate_relaxed;
  }
  return sources;
}

void DrawStereoBackendObservation(
    cv::Mat* image,
    const cv::Point& observed_pt,
    const cv::Point& predicted_pt,
    bool is_internal,
    const cv::Scalar& board_color,
    bool has_residual,
    double residual_norm) {
  if (image == nullptr || image->empty()) {
    return;
  }
  const auto clamp_to_image = [&](const cv::Point& pt) {
    return cv::Point(std::min(std::max(pt.x, 0), image->cols - 1),
                     std::min(std::max(pt.y, 0), image->rows - 1));
  };
  const cv::Point clipped_observed = clamp_to_image(observed_pt);
  const cv::Point clipped_predicted = clamp_to_image(predicted_pt);
  const cv::Scalar residual_color =
      !has_residual
          ? cv::Scalar(255, 120, 0)
          : (residual_norm <= 3.0 ? cv::Scalar(60, 230, 60)
                                  : cv::Scalar(0, 215, 255));
  cv::line(*image, clipped_observed, clipped_predicted, residual_color, 1,
           cv::LINE_AA);
  if (is_internal) {
    cv::circle(*image, clipped_observed, 4, residual_color, cv::FILLED,
               cv::LINE_AA);
    cv::circle(*image, clipped_observed, 6, cv::Scalar(255, 255, 255), 1,
               cv::LINE_AA);
  } else {
    cv::circle(*image, clipped_observed, 5, board_color, 2, cv::LINE_AA);
    cv::circle(*image, clipped_observed, 2, board_color, cv::FILLED,
               cv::LINE_AA);
  }
  cv::drawMarker(*image, clipped_predicted, cv::Scalar(255, 0, 255),
                 cv::MARKER_TILTED_CROSS, 10, 1, cv::LINE_AA);
}

std::vector<int> SortedBoardIds(const StereoMeasurementDataset& dataset,
                                int pair_index) {
  std::set<int> board_ids;
  const auto train_it = dataset.training_pair_board_ids.find(pair_index);
  if (train_it != dataset.training_pair_board_ids.end()) {
    board_ids = train_it->second;
  }
  const auto holdout_it = dataset.holdout_pair_board_ids.find(pair_index);
  if (holdout_it != dataset.holdout_pair_board_ids.end()) {
    board_ids.insert(holdout_it->second.begin(), holdout_it->second.end());
  }
  return std::vector<int>(board_ids.begin(), board_ids.end());
}

bool ContainsBoard(const std::map<int, std::set<int> >& boards_by_pair,
                   int pair_index,
                   int board_id) {
  const auto it = boards_by_pair.find(pair_index);
  return it != boards_by_pair.end() && it->second.count(board_id) > 0;
}

bool PairHasBoardVisibleInCamera(const StereoMeasurementDataset& dataset,
                                 int pair_index,
                                 int camera_index,
                                 int board_id) {
  for (const StereoObservation& observation : dataset.observations) {
    if (observation.pair_index == pair_index &&
        observation.camera_index == camera_index &&
        observation.board_id == board_id &&
        observation.point_type == JointPointType::Outer &&
        observation.used_in_solver) {
      return true;
    }
  }
  return false;
}

std::string JoinInts(const std::vector<int>& values, const char delimiter) {
  std::ostringstream stream;
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (i > 0) {
      stream << delimiter;
    }
    stream << values[i];
  }
  return stream.str();
}

int CountOuterObservations(const StereoMeasurementDataset& dataset,
                           int pair_index,
                           int camera_index,
                           int board_id) {
  int count = 0;
  for (const StereoObservation& observation : dataset.observations) {
    if (observation.pair_index == pair_index &&
        observation.camera_index == camera_index &&
        observation.board_id == board_id &&
        observation.point_type == JointPointType::Outer &&
        observation.used_in_solver) {
      ++count;
    }
  }
  return count;
}

struct OuterSubpixWindowSummary {
  int outer_count = 0;
  int boosted_outer_count = 0;
  int max_window_radius = 0;
  int max_pre_boost_window_radius = 0;
  int max_boosted_raw_window_radius = 0;
  double max_area_ratio = 0.0;
  double max_polar_deg = 0.0;
};

OuterSubpixWindowSummary SummarizeOuterSubpixWindows(
    const StereoMeasurementDataset& dataset,
    int pair_index,
    int board_id) {
  OuterSubpixWindowSummary summary;
  for (const StereoObservation& observation : dataset.observations) {
    if (observation.pair_index != pair_index ||
        observation.board_id != board_id ||
        observation.point_type != JointPointType::Outer ||
        !observation.used_in_solver) {
      continue;
    }
    ++summary.outer_count;
    summary.max_window_radius =
        std::max(summary.max_window_radius,
                 observation.outer_subpix_window_radius);
    summary.max_pre_boost_window_radius =
        std::max(summary.max_pre_boost_window_radius,
                 observation.outer_pre_boost_subpix_window_radius);
    summary.max_boosted_raw_window_radius =
        std::max(summary.max_boosted_raw_window_radius,
                 observation.outer_boosted_raw_subpix_window_radius);
    summary.max_area_ratio =
        std::max(summary.max_area_ratio,
                 observation.outer_close_edge_subpix_area_ratio);
    summary.max_polar_deg =
        std::max(summary.max_polar_deg,
                 observation.outer_close_edge_subpix_max_polar_deg);
    if (observation.outer_close_edge_subpix_boost_applied) {
      ++summary.boosted_outer_count;
    }
  }
  return summary;
}

std::vector<int> CollectTrainingBoardIds(const StereoMeasurementDataset& dataset) {
  std::set<int> board_ids;
  for (const auto& pair_boards : dataset.training_pair_board_ids) {
    board_ids.insert(pair_boards.second.begin(), pair_boards.second.end());
  }
  return std::vector<int>(board_ids.begin(), board_ids.end());
}

bool BuildPairBoardGraphEdge(const StereoMeasurementDataset& dataset,
                             const StereoCameraFixedCalibration& cam0,
                             const StereoCameraFixedCalibration& cam1,
                             const StereoExtrinsicSolverOptions& options,
                             int pair_index,
                             int board_id,
                             PairBoardGraphEdge* edge) {
  if (edge == nullptr) {
    throw std::runtime_error("BuildPairBoardGraphEdge requires output edge.");
  }
  edge->pair_index = pair_index;
  edge->board_id = board_id;
  edge->cam0_outer_point_count =
      CountOuterObservations(dataset, pair_index, 0, board_id);
  edge->cam1_outer_point_count =
      CountOuterObservations(dataset, pair_index, 1, board_id);
  edge->cam0_valid = EstimateCameraBoardPoseWithRmse(
      dataset, cam0, pair_index, 0, board_id,
      &edge->T_cam0_board, &edge->cam0_rmse) &&
      edge->cam0_rmse <= SharedBoardOuterRmseThreshold(options);
  edge->cam1_valid = EstimateCameraBoardPoseWithRmse(
      dataset, cam1, pair_index, 1, board_id,
      &edge->T_cam1_board, &edge->cam1_rmse) &&
      edge->cam1_rmse <= SharedBoardOuterRmseThreshold(options);
  edge->shared_stereo = edge->cam0_valid && edge->cam1_valid;
  return edge->cam0_valid || edge->cam1_valid;
}

bool IsCandidateBetter(const BootstrapTransformCandidate& lhs,
                       const BootstrapTransformCandidate& rhs) {
  if (lhs.is_shared_stereo != rhs.is_shared_stereo) {
    return lhs.is_shared_stereo && !rhs.is_shared_stereo;
  }
  if (lhs.outer_point_count != rhs.outer_point_count) {
    return lhs.outer_point_count > rhs.outer_point_count;
  }
  if (std::abs(lhs.pose_fit_rmse - rhs.pose_fit_rmse) > 1e-12) {
    return lhs.pose_fit_rmse < rhs.pose_fit_rmse;
  }
  if (lhs.source_pair_index != rhs.source_pair_index) {
    return lhs.source_pair_index < rhs.source_pair_index;
  }
  if (lhs.source_board_id != rhs.source_board_id) {
    return lhs.source_board_id < rhs.source_board_id;
  }
  return lhs.source_camera_index < rhs.source_camera_index;
}

bool SelectBootstrapCandidate(
    const std::vector<BootstrapTransformCandidate>& candidates,
    const StereoExtrinsicSolverOptions& options,
    int* rejected_for_consistency,
    BootstrapTransformCandidate* selected_candidate) {
  if (selected_candidate == nullptr) {
    throw std::runtime_error("SelectBootstrapCandidate requires output candidate.");
  }
  if (rejected_for_consistency != nullptr) {
    *rejected_for_consistency = 0;
  }
  if (candidates.empty()) {
    return false;
  }

  std::vector<TransformCandidate> transforms;
  transforms.reserve(candidates.size());
  for (const BootstrapTransformCandidate& candidate : candidates) {
    TransformCandidate transform_candidate;
    transform_candidate.transform = candidate.transform;
    transform_candidate.weight = 1.0;
    transforms.push_back(transform_candidate);
  }

  int local_rejected = 0;
  const std::vector<TransformCandidate> filtered_transforms =
      FilterConsistentCandidates(
          transforms,
          DegreesToRadians(options.candidate_consistency_max_rotation_deg),
          options.candidate_consistency_max_translation_m,
          &local_rejected);
  if (rejected_for_consistency != nullptr) {
    *rejected_for_consistency = local_rejected;
  }
  if (filtered_transforms.empty()) {
    return false;
  }

  std::vector<BootstrapTransformCandidate> filtered_candidates;
  filtered_candidates.reserve(filtered_transforms.size());
  for (std::size_t i = 0; i < candidates.size(); ++i) {
    bool accepted = false;
    for (const TransformCandidate& filtered : filtered_transforms) {
      if (TransformDistanceScore(filtered.transform, candidates[i].transform) < 1e-12) {
        accepted = true;
        break;
      }
    }
    if (accepted) {
      filtered_candidates.push_back(candidates[i]);
    }
  }
  if (filtered_candidates.empty()) {
    return false;
  }

  BootstrapTransformCandidate best = filtered_candidates.front();
  for (std::size_t i = 1; i < filtered_candidates.size(); ++i) {
    if (IsCandidateBetter(filtered_candidates[i], best)) {
      best = filtered_candidates[i];
    }
  }
  *selected_candidate = best;
  return true;
}

bool PairHasSharedBoardCandidate(const StereoMeasurementDataset& dataset,
                                 int pair_index,
                                 int min_shared_boards) {
  int shared_count = 0;
  for (int board_id : SortedBoardIds(dataset, pair_index)) {
    if (PairHasBoardVisibleInCamera(dataset, pair_index, 0, board_id) &&
        PairHasBoardVisibleInCamera(dataset, pair_index, 1, board_id)) {
      ++shared_count;
    }
  }
  return shared_count >= std::max(1, min_shared_boards);
}

double SharedBoardOuterRmseThreshold(
    const StereoExtrinsicSolverOptions& options) {
  if (!options.enable_shared_board_quality_hard_gate) {
    return options.pose_fit_guard_threshold_px;
  }
  return std::min(options.pose_fit_guard_threshold_px,
                  options.shared_board_quality_max_outer_rmse_px);
}

double SharedBoardQualityAuditRmseThreshold(
    const StereoExtrinsicSolverOptions& options) {
  if (!options.enable_shared_board_quality_gate &&
      !options.enable_shared_board_quality_hard_gate) {
    return options.pose_fit_guard_threshold_px;
  }
  return std::min(options.pose_fit_guard_threshold_px,
                  options.shared_board_quality_max_outer_rmse_px);
}

bool SharedBoardQualityHardGateEnabled(
    const StereoExtrinsicSolverOptions& options) {
  return options.enable_shared_board_quality_hard_gate;
}

struct StereoSharedBoardQuality {
  bool have_cam0 = false;
  bool have_cam1 = false;
  bool pass = false;
  int cam0_outer_point_count = 0;
  int cam1_outer_point_count = 0;
  double cam0_outer_rmse = std::numeric_limits<double>::infinity();
  double cam1_outer_rmse = std::numeric_limits<double>::infinity();
  Eigen::Isometry3d T_cam0_board = Eigen::Isometry3d::Identity();
  Eigen::Isometry3d T_cam1_board = Eigen::Isometry3d::Identity();
};

bool EstimateCameraBoardPose(const StereoMeasurementDataset& dataset,
                             const StereoCameraFixedCalibration& calibration,
                             int pair_index,
                             int camera_index,
                             int board_id,
                             Eigen::Isometry3d* T_camera_board) {
  if (T_camera_board == nullptr) {
    throw std::runtime_error("EstimateCameraBoardPose requires output transform.");
  }
  std::vector<Eigen::Vector3d> target_points;
  std::vector<cv::Point2f> image_points;
  for (const StereoObservation& observation : dataset.observations) {
    if (observation.pair_index != pair_index ||
        observation.camera_index != camera_index ||
        observation.board_id != board_id ||
        observation.point_type != JointPointType::Outer ||
        !observation.used_in_solver) {
      continue;
    }
    target_points.push_back(observation.target_point_board);
    image_points.push_back(cv::Point2f(
        static_cast<float>(observation.observed_image_xy.x()),
        static_cast<float>(observation.observed_image_xy.y())));
  }
  if (target_points.size() < 4) {
    return false;
  }
  const DoubleSphereCameraModel camera =
      DoubleSphereCameraModel::FromConfig(MakeCameraConfig(calibration));
  cv::Mat rvec;
  cv::Mat tvec;
  if (!camera.estimateTransformation(BuildObjectPoints(target_points), image_points,
                                     &rvec, &tvec)) {
    return false;
  }
  *T_camera_board = MakePose(rvec, tvec);
  return true;
}

bool EstimateCameraBoardPoseWithRmse(const StereoMeasurementDataset& dataset,
                                     const StereoCameraFixedCalibration& calibration,
                                     int pair_index,
                                     int camera_index,
                                     int board_id,
                                     Eigen::Isometry3d* T_camera_board,
                                     double* rmse) {
  if (T_camera_board == nullptr || rmse == nullptr) {
    throw std::runtime_error(
        "EstimateCameraBoardPoseWithRmse requires valid output pointers.");
  }
  std::vector<Eigen::Vector3d> target_points;
  std::vector<cv::Point2f> image_points;
  for (const StereoObservation& observation : dataset.observations) {
    if (observation.pair_index != pair_index ||
        observation.camera_index != camera_index ||
        observation.board_id != board_id ||
        observation.point_type != JointPointType::Outer ||
        !observation.used_in_solver) {
      continue;
    }
    target_points.push_back(observation.target_point_board);
    image_points.push_back(cv::Point2f(
        static_cast<float>(observation.observed_image_xy.x()),
        static_cast<float>(observation.observed_image_xy.y())));
  }
  if (target_points.size() < 4) {
    return false;
  }
  const DoubleSphereCameraModel camera =
      DoubleSphereCameraModel::FromConfig(MakeCameraConfig(calibration));
  cv::Mat rvec;
  cv::Mat tvec;
  if (!camera.estimateTransformation(BuildObjectPoints(target_points), image_points,
                                     &rvec, &tvec)) {
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
    const Eigen::Vector2d residual =
        predicted - Eigen::Vector2d(image_points[i].x, image_points[i].y);
    squared_error_sum += residual.squaredNorm();
    ++count;
  }
  if (count <= 0) {
    return false;
  }
  *T_camera_board = pose;
  *rmse = std::sqrt(squared_error_sum / static_cast<double>(count));
  return true;
}

int CountOuterObservationsForCamera(const StereoMeasurementDataset& dataset,
                                    int pair_index,
                                    int camera_index,
                                    int board_id) {
  int count = 0;
  for (const StereoObservation& observation : dataset.observations) {
    if (observation.pair_index == pair_index &&
        observation.camera_index == camera_index &&
        observation.board_id == board_id &&
        observation.point_type == JointPointType::Outer &&
        observation.used_in_solver) {
      ++count;
    }
  }
  return count;
}

double EvaluateGlobalOuterRmseForPairBoard(
    const StereoMeasurementDataset& dataset,
    const StereoSceneState& scene_state,
    int pair_index,
    int board_id,
    int* point_count) {
  if (point_count != nullptr) {
    *point_count = 0;
  }
  const auto board_pose_it = scene_state.T_world_board_by_id.find(board_id);
  const auto pair_pose_it = scene_state.T_cam0_world_by_pair.find(pair_index);
  if (pair_pose_it == scene_state.T_cam0_world_by_pair.end() ||
      board_pose_it == scene_state.T_world_board_by_id.end()) {
    return std::numeric_limits<double>::infinity();
  }
  const Eigen::Isometry3d T_cam0_world = ToIsometry3d(pair_pose_it->second);
  const Eigen::Isometry3d T_world_board = ToIsometry3d(board_pose_it->second);
  const Eigen::Isometry3d T_cam1_cam0 = ToIsometry3d(scene_state.T_cam1_cam0);
  const DoubleSphereCameraModel cam0 =
      DoubleSphereCameraModel::FromConfig(MakeCameraConfig(scene_state.cam0));
  const DoubleSphereCameraModel cam1 =
      DoubleSphereCameraModel::FromConfig(MakeCameraConfig(scene_state.cam1));
  if (!cam0.IsValid() || !cam1.IsValid()) {
    return std::numeric_limits<double>::infinity();
  }
  double squared_error_sum = 0.0;
  int count = 0;
  for (const StereoObservation& observation : dataset.observations) {
    if (observation.pair_index != pair_index ||
        observation.board_id != board_id ||
        observation.point_type != JointPointType::Outer ||
        !observation.used_in_solver) {
      continue;
    }
    const Eigen::Vector3d point_cam0 =
        T_cam0_world * (T_world_board * observation.target_point_board);
    Eigen::Vector2d predicted = Eigen::Vector2d::Zero();
    bool ok = false;
    if (observation.camera_index == 0) {
      ok = cam0.vsEuclideanToKeypoint(point_cam0, &predicted);
    } else {
      ok = cam1.vsEuclideanToKeypoint(T_cam1_cam0 * point_cam0, &predicted);
    }
    if (!ok) {
      continue;
    }
    const Eigen::Vector2d residual = predicted - observation.observed_image_xy;
    squared_error_sum += residual.squaredNorm();
    ++count;
  }
  if (point_count != nullptr) {
    *point_count = count;
  }
  if (count <= 0) {
    return std::numeric_limits<double>::infinity();
  }
  return std::sqrt(squared_error_sum / static_cast<double>(count));
}

double EvaluateOuterRmseForPairBoardWithPose(
    const StereoMeasurementDataset& dataset,
    const StereoSceneState& scene_state,
    int pair_index,
    int board_id,
    const Eigen::Isometry3d& T_cam0_world,
    int* point_count) {
  if (point_count != nullptr) {
    *point_count = 0;
  }
  const auto board_pose_it = scene_state.T_world_board_by_id.find(board_id);
  if (board_pose_it == scene_state.T_world_board_by_id.end()) {
    return std::numeric_limits<double>::infinity();
  }
  const DoubleSphereCameraModel cam0 =
      DoubleSphereCameraModel::FromConfig(MakeCameraConfig(scene_state.cam0));
  const DoubleSphereCameraModel cam1 =
      DoubleSphereCameraModel::FromConfig(MakeCameraConfig(scene_state.cam1));
  if (!cam0.IsValid() || !cam1.IsValid()) {
    return std::numeric_limits<double>::infinity();
  }
  const Eigen::Isometry3d T_world_board = ToIsometry3d(board_pose_it->second);
  const Eigen::Isometry3d T_cam1_cam0 = ToIsometry3d(scene_state.T_cam1_cam0);
  double squared_error_sum = 0.0;
  int count = 0;
  for (const StereoObservation& observation : dataset.observations) {
    if (observation.pair_index != pair_index ||
        observation.board_id != board_id ||
        observation.point_type != JointPointType::Outer ||
        !observation.used_in_solver) {
      continue;
    }
    const Eigen::Vector3d point_cam0 =
        T_cam0_world * (T_world_board * observation.target_point_board);
    Eigen::Vector2d predicted = Eigen::Vector2d::Zero();
    bool ok = false;
    if (observation.camera_index == 0) {
      ok = cam0.vsEuclideanToKeypoint(point_cam0, &predicted);
    } else {
      ok = cam1.vsEuclideanToKeypoint(T_cam1_cam0 * point_cam0, &predicted);
    }
    if (!ok) {
      continue;
    }
    const Eigen::Vector2d residual = predicted - observation.observed_image_xy;
    squared_error_sum += residual.squaredNorm();
    ++count;
  }
  if (point_count != nullptr) {
    *point_count = count;
  }
  if (count <= 0) {
    return std::numeric_limits<double>::infinity();
  }
  return std::sqrt(squared_error_sum / static_cast<double>(count));
}

double EvaluateCameraOuterRmseForBoardWithPose(
    const StereoMeasurementDataset& dataset,
    const StereoCameraFixedCalibration& calibration,
    int pair_index,
    int camera_index,
    int board_id,
    const Eigen::Isometry3d& T_camera_board,
    int* point_count) {
  if (point_count != nullptr) {
    *point_count = 0;
  }
  const DoubleSphereCameraModel camera =
      DoubleSphereCameraModel::FromConfig(MakeCameraConfig(calibration));
  if (!camera.IsValid()) {
    return std::numeric_limits<double>::infinity();
  }
  double squared_error_sum = 0.0;
  int count = 0;
  for (const StereoObservation& observation : dataset.observations) {
    if (observation.pair_index != pair_index ||
        observation.camera_index != camera_index ||
        observation.board_id != board_id ||
        observation.point_type != JointPointType::Outer ||
        !observation.used_in_solver) {
      continue;
    }
    Eigen::Vector2d predicted = Eigen::Vector2d::Zero();
    if (!camera.vsEuclideanToKeypoint(
            T_camera_board * observation.target_point_board, &predicted)) {
      continue;
    }
    const Eigen::Vector2d residual = predicted - observation.observed_image_xy;
    squared_error_sum += residual.squaredNorm();
    ++count;
  }
  if (point_count != nullptr) {
    *point_count = count;
  }
  if (count <= 0) {
    return std::numeric_limits<double>::infinity();
  }
  return std::sqrt(squared_error_sum / static_cast<double>(count));
}

double SafeMinFinite(double lhs, double rhs) {
  const bool lhs_finite = std::isfinite(lhs);
  const bool rhs_finite = std::isfinite(rhs);
  if (lhs_finite && rhs_finite) {
    return std::min(lhs, rhs);
  }
  if (lhs_finite) {
    return lhs;
  }
  if (rhs_finite) {
    return rhs;
  }
  return std::numeric_limits<double>::infinity();
}

bool RefitPairPoseFromStereoOuterObservations(const StereoMeasurementDataset& dataset,
                                              const StereoSceneState& scene_state,
                                              int pair_index,
                                              const StereoExtrinsicSolverOptions& options,
                                              RuntimeCounters* runtime_counters,
                                              StereoRefitDiagnostics* diagnostics,
                                              Eigen::Matrix4d* T_cam0_world);

StereoPairBoardConsistencySummary BuildPairBoardConsistencySummary(
    const StereoMeasurementDataset& dataset,
    const StereoSceneState& scene_state,
    const StereoExtrinsicSolverOptions& options) {
  StereoPairBoardConsistencySummary summary;
  summary.enabled = options.export_pair_board_consistency_audit ||
                    options.enable_pair_board_consistency_gate;
  summary.gate_enabled = options.enable_pair_board_consistency_gate;
  summary.local_good_max_outer_rmse_px =
      options.pair_board_consistency_local_good_max_outer_rmse_px;
  summary.global_bad_min_outer_rmse_px =
      options.pair_board_consistency_global_bad_min_outer_rmse_px;
  if (!summary.enabled) {
    return summary;
  }

  auto add_rows_for_split = [&](const std::vector<int>& pair_indices,
                                const std::string& split) {
    for (int pair_index : pair_indices) {
      const StereoFramePair* pair = FindPair(dataset, pair_index);
      Eigen::Isometry3d audit_T_cam0_world = Eigen::Isometry3d::Identity();
      bool have_audit_pair_pose = false;
      const auto audit_pair_pose_it =
          scene_state.T_cam0_world_by_pair.find(pair_index);
      if (audit_pair_pose_it != scene_state.T_cam0_world_by_pair.end()) {
        audit_T_cam0_world = ToIsometry3d(audit_pair_pose_it->second);
        have_audit_pair_pose = true;
      } else if (split == "holdout") {
        // Holdout pairs are not design variables in the optimized scene. Use the
        // same stereo-outer refit pose as the holdout evaluator so the audit is
        // comparable to extrinsic-only holdout RMSE instead of reporting
        // infinity.
        RuntimeCounters runtime_counters;
        StereoRefitDiagnostics refit_diagnostics;
        Eigen::Matrix4d refit_pose = Eigen::Matrix4d::Identity();
        if (RefitPairPoseFromStereoOuterObservations(
                dataset, scene_state, pair_index, options, &runtime_counters,
                &refit_diagnostics, &refit_pose)) {
          audit_T_cam0_world = ToIsometry3d(refit_pose);
          have_audit_pair_pose = true;
        }
      }
      for (int board_id : SortedBoardIds(dataset, pair_index)) {
        StereoPairBoardConsistencyRow row;
        row.split = split;
        row.pair_index = pair_index;
        row.board_id = board_id;
        if (pair != nullptr) {
          row.left_frame_label = pair->left_frame_label;
          row.right_frame_label = pair->right_frame_label;
          row.is_training = pair->is_training;
        }
        row.shared_board =
            ContainsBoard(dataset.pair_shared_board_ids, pair_index, board_id);
        row.cam0_only_board =
            ContainsBoard(dataset.pair_cam0_only_board_ids, pair_index, board_id);
        row.cam1_only_board =
            ContainsBoard(dataset.pair_cam1_only_board_ids, pair_index, board_id);
        row.cam0_outer_point_count =
            CountOuterObservationsForCamera(dataset, pair_index, 0, board_id);
        row.cam1_outer_point_count =
            CountOuterObservationsForCamera(dataset, pair_index, 1, board_id);
        row.global_outer_rmse =
            have_audit_pair_pose
                ? EvaluateOuterRmseForPairBoardWithPose(
                      dataset, scene_state, pair_index, board_id,
                      audit_T_cam0_world, &row.global_outer_point_count)
                : EvaluateGlobalOuterRmseForPairBoard(
                      dataset, scene_state, pair_index, board_id,
                      &row.global_outer_point_count);

        Eigen::Isometry3d T_cam0_board = Eigen::Isometry3d::Identity();
        Eigen::Isometry3d T_cam1_board = Eigen::Isometry3d::Identity();
        row.cam0_local_success = EstimateCameraBoardPoseWithRmse(
            dataset, scene_state.cam0, pair_index, 0, board_id, &T_cam0_board,
            &row.cam0_local_outer_rmse);
        row.cam1_local_success = EstimateCameraBoardPoseWithRmse(
            dataset, scene_state.cam1, pair_index, 1, board_id, &T_cam1_board,
            &row.cam1_local_outer_rmse);
        row.local_outer_rmse =
            SafeMinFinite(row.cam0_local_outer_rmse, row.cam1_local_outer_rmse);
        if (row.cam0_local_success && row.cam1_local_success) {
          const Eigen::Isometry3d T_cam1_cam0 =
              ToIsometry3d(scene_state.T_cam1_cam0);
          const Eigen::Isometry3d T_cam1_board_from_cam0 =
              T_cam1_cam0 * T_cam0_board;
          row.stereo_local_pose_delta_rotation_deg =
              RotationDistanceRadians(T_cam1_board,
                                      T_cam1_board_from_cam0) *
              180.0 / M_PI;
          row.stereo_local_pose_delta_translation_m =
              (T_cam1_board.translation() -
               T_cam1_board_from_cam0.translation()).norm();
          int projected_point_count = 0;
          row.cam1_outer_rmse_from_cam0_pose =
              EvaluateCameraOuterRmseForBoardWithPose(
                  dataset, scene_state.cam1, pair_index, 1, board_id,
                  T_cam1_board_from_cam0, &projected_point_count);
          if (std::isfinite(row.cam0_local_outer_rmse) &&
              std::isfinite(row.cam1_outer_rmse_from_cam0_pose) &&
              row.cam0_outer_point_count + row.cam1_outer_point_count > 0) {
            const double squared_sum =
                static_cast<double>(row.cam0_outer_point_count) *
                    row.cam0_local_outer_rmse * row.cam0_local_outer_rmse +
                static_cast<double>(row.cam1_outer_point_count) *
                    row.cam1_outer_rmse_from_cam0_pose *
                    row.cam1_outer_rmse_from_cam0_pose;
            row.stereo_outer_rmse_from_cam0_pose =
                std::sqrt(squared_sum /
                          static_cast<double>(row.cam0_outer_point_count +
                                              row.cam1_outer_point_count));
          }

          const Eigen::Isometry3d T_cam0_board_from_cam1 =
              T_cam1_cam0.inverse() * T_cam1_board;
          row.cam0_outer_rmse_from_cam1_pose =
              EvaluateCameraOuterRmseForBoardWithPose(
                  dataset, scene_state.cam0, pair_index, 0, board_id,
                  T_cam0_board_from_cam1, &projected_point_count);
          if (std::isfinite(row.cam1_local_outer_rmse) &&
              std::isfinite(row.cam0_outer_rmse_from_cam1_pose) &&
              row.cam0_outer_point_count + row.cam1_outer_point_count > 0) {
            const double squared_sum =
                static_cast<double>(row.cam0_outer_point_count) *
                    row.cam0_outer_rmse_from_cam1_pose *
                    row.cam0_outer_rmse_from_cam1_pose +
                static_cast<double>(row.cam1_outer_point_count) *
                    row.cam1_local_outer_rmse * row.cam1_local_outer_rmse;
            row.stereo_outer_rmse_from_cam1_pose =
                std::sqrt(squared_sum /
                          static_cast<double>(row.cam0_outer_point_count +
                                              row.cam1_outer_point_count));
          }

          // Direction sanity check: if these inverse-direction RMSE values are
          // much smaller, the camchain convention is likely being interpreted
          // backwards somewhere.
          const Eigen::Isometry3d inverse_direction_T_cam1_board =
              T_cam1_cam0.inverse() * T_cam0_board;
          row.cam1_outer_rmse_from_cam0_pose_inverse_extrinsic =
              EvaluateCameraOuterRmseForBoardWithPose(
                  dataset, scene_state.cam1, pair_index, 1, board_id,
                  inverse_direction_T_cam1_board, &projected_point_count);
          const Eigen::Isometry3d inverse_direction_T_cam0_board =
              T_cam1_cam0 * T_cam1_board;
          row.cam0_outer_rmse_from_cam1_pose_inverse_extrinsic =
              EvaluateCameraOuterRmseForBoardWithPose(
                  dataset, scene_state.cam0, pair_index, 0, board_id,
                  inverse_direction_T_cam0_board, &projected_point_count);
        }

        const auto board_pose_it = scene_state.T_world_board_by_id.find(board_id);
        if (have_audit_pair_pose &&
            board_pose_it != scene_state.T_world_board_by_id.end()) {
          const Eigen::Isometry3d global_T_cam0_board =
              audit_T_cam0_world * ToIsometry3d(board_pose_it->second);
          if (row.cam0_local_success) {
            row.cam0_pose_delta_rotation_deg =
                RotationDistanceRadians(T_cam0_board, global_T_cam0_board) *
                180.0 / M_PI;
            row.cam0_pose_delta_translation_m =
                (T_cam0_board.translation() -
                 global_T_cam0_board.translation()).norm();
          }
          if (row.cam1_local_success) {
            const Eigen::Isometry3d global_T_cam1_board =
                ToIsometry3d(scene_state.T_cam1_cam0) * global_T_cam0_board;
            row.cam1_pose_delta_rotation_deg =
                RotationDistanceRadians(T_cam1_board, global_T_cam1_board) *
                180.0 / M_PI;
            row.cam1_pose_delta_translation_m =
                (T_cam1_board.translation() -
                 global_T_cam1_board.translation()).norm();
          }
        }

        row.local_good_global_bad =
            std::isfinite(row.local_outer_rmse) &&
            std::isfinite(row.global_outer_rmse) &&
            row.local_outer_rmse <= summary.local_good_max_outer_rmse_px &&
            row.global_outer_rmse >= summary.global_bad_min_outer_rmse_px;
        row.rejected_by_consistency_gate =
            summary.gate_enabled && row.local_good_global_bad &&
            row.is_training;
        if (row.local_good_global_bad) {
          row.diagnosis_label = "local_good_global_bad";
          ++summary.local_good_global_bad_count;
        } else if (std::isfinite(row.local_outer_rmse) &&
                   row.local_outer_rmse >
                       summary.local_good_max_outer_rmse_px) {
          row.diagnosis_label = "local_bad";
        } else if (std::isfinite(row.global_outer_rmse) &&
                   row.global_outer_rmse <
                       summary.global_bad_min_outer_rmse_px) {
          row.diagnosis_label = "global_consistent";
        } else {
          row.diagnosis_label = "insufficient_data";
        }
        if (row.rejected_by_consistency_gate) {
          summary.gate_rejected_pair_boards.insert(
              std::make_pair(row.pair_index, row.board_id));
        }
        if (split == "training") {
          ++summary.training_row_count;
        } else {
          ++summary.holdout_row_count;
        }
        summary.rows.push_back(row);
      }
    }
  };

  add_rows_for_split(dataset.training_pair_indices, "training");
  add_rows_for_split(dataset.holdout_pair_indices, "holdout");
  summary.row_count = static_cast<int>(summary.rows.size());
  summary.gate_rejected_pair_board_count =
      static_cast<int>(summary.gate_rejected_pair_boards.size());
  return summary;
}

StereoSharedBoardQuality EvaluateSharedBoardQuality(
    const StereoMeasurementDataset& dataset,
    const StereoSceneState& scene_state,
    const StereoExtrinsicSolverOptions& options,
    int pair_index,
    int board_id) {
  StereoSharedBoardQuality quality;
  quality.cam0_outer_point_count =
      CountOuterObservations(dataset, pair_index, 0, board_id);
  quality.cam1_outer_point_count =
      CountOuterObservations(dataset, pair_index, 1, board_id);
  quality.have_cam0 =
      quality.cam0_outer_point_count >=
          options.shared_board_quality_min_outer_points_per_camera &&
      EstimateCameraBoardPoseWithRmse(dataset, scene_state.cam0, pair_index, 0,
                                      board_id, &quality.T_cam0_board,
                                      &quality.cam0_outer_rmse);
  quality.have_cam1 =
      quality.cam1_outer_point_count >=
          options.shared_board_quality_min_outer_points_per_camera &&
      EstimateCameraBoardPoseWithRmse(dataset, scene_state.cam1, pair_index, 1,
                                      board_id, &quality.T_cam1_board,
                                      &quality.cam1_outer_rmse);
  const double rmse_threshold = SharedBoardQualityAuditRmseThreshold(options);
  quality.pass = quality.have_cam0 && quality.have_cam1 &&
                 quality.cam0_outer_rmse <= rmse_threshold &&
                 quality.cam1_outer_rmse <= rmse_threshold;
  return quality;
}

int CountQualitySharedBoards(const StereoMeasurementDataset& dataset,
                             const StereoSceneState& scene_state,
                             const StereoExtrinsicSolverOptions& options,
                             int pair_index) {
  int count = 0;
  for (int board_id : SortedBoardIds(dataset, pair_index)) {
    if (EvaluateSharedBoardQuality(dataset, scene_state, options, pair_index,
                                   board_id)
            .pass) {
      ++count;
    }
  }
  return count;
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
  const OuterBootstrapCameraIntrinsics intrinsics =
      MakeOuterBootstrapCameraIntrinsics(scene_state.cam0);
  Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
  double rmse = 0.0;
  if (!EstimatePoseFromObjectPoints(intrinsics, object_points, image_points, &pose, &rmse)) {
    return false;
  }
  *T_cam0_world = ToMatrix4d(pose);
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

std::vector<StereoBoardPoseObservation> CollectStereoBoardPoseObservations(
    const StereoMeasurementDataset& dataset,
    int pair_index,
    int board_id) {
  std::vector<StereoBoardPoseObservation> observations;
  for (const StereoObservation& observation : dataset.observations) {
    if (observation.pair_index != pair_index ||
        observation.board_id != board_id ||
        observation.point_type != JointPointType::Outer ||
        !observation.used_in_solver) {
      continue;
    }
    StereoBoardPoseObservation stereo_observation;
    stereo_observation.camera_index = observation.camera_index;
    stereo_observation.object_point_board = observation.target_point_board;
    stereo_observation.observed_image_xy = observation.observed_image_xy;
    observations.push_back(stereo_observation);
  }
  return observations;
}

double EvaluateStereoBoardPoseRmse(
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
      valid_projection =
          cam1.vsEuclideanToKeypoint(T_cam1_cam0 * point_cam0, &predicted);
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

bool RefitStereoBoardPoseForVisualization(
    const StereoMeasurementDataset& dataset,
    const StereoSceneState& scene_state,
    int pair_index,
    int board_id,
    int max_iterations,
    double step,
    Eigen::Matrix4d* T_cam0_board) {
  if (T_cam0_board == nullptr) {
    throw std::runtime_error(
        "RefitStereoBoardPoseForVisualization requires output pose.");
  }
  Eigen::Isometry3d seed_pose = Eigen::Isometry3d::Identity();
  double seed_rmse = 0.0;
  if (!EstimateCameraBoardPoseWithRmse(dataset, scene_state.cam0, pair_index, 0,
                                      board_id, &seed_pose, &seed_rmse)) {
    return false;
  }

  const std::vector<StereoBoardPoseObservation> observations =
      CollectStereoBoardPoseObservations(dataset, pair_index, board_id);
  if (observations.size() < 8) {
    *T_cam0_board = ToMatrix4d(seed_pose);
    return true;
  }

  Eigen::Isometry3d current_pose = seed_pose;
  double current_rmse =
      EvaluateStereoBoardPoseRmse(observations, scene_state, current_pose);
  if (!std::isfinite(current_rmse)) {
    *T_cam0_board = ToMatrix4d(seed_pose);
    return true;
  }

  for (int iteration = 0; iteration < max_iterations; ++iteration) {
    Eigen::Matrix<double, 6, 1> gradient = Eigen::Matrix<double, 6, 1>::Zero();
    Eigen::Matrix<double, 6, 6> hessian = Eigen::Matrix<double, 6, 6>::Zero();
    for (int axis = 0; axis < 6; ++axis) {
      Eigen::Matrix<double, 6, 1> plus_delta = Eigen::Matrix<double, 6, 1>::Zero();
      Eigen::Matrix<double, 6, 1> minus_delta = Eigen::Matrix<double, 6, 1>::Zero();
      plus_delta(axis) = step;
      minus_delta(axis) = -step;
      const double plus_rmse = EvaluateStereoBoardPoseRmse(
          observations, scene_state, ApplyPoseDelta(current_pose, plus_delta));
      const double minus_rmse = EvaluateStereoBoardPoseRmse(
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
    const double candidate_rmse =
        EvaluateStereoBoardPoseRmse(observations, scene_state, candidate_pose);
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

bool RefitPairPoseFromStereoOuterObservations(const StereoMeasurementDataset& dataset,
                                              const StereoSceneState& scene_state,
                                              int pair_index,
                                              const StereoExtrinsicSolverOptions& options,
                                              RuntimeCounters* runtime_counters,
                                              StereoRefitDiagnostics* diagnostics,
                                              Eigen::Matrix4d* T_cam0_world) {
  if (T_cam0_world == nullptr) {
    throw std::runtime_error(
        "RefitPairPoseFromStereoOuterObservations requires output pose.");
  }
  if (runtime_counters == nullptr || diagnostics == nullptr) {
    throw std::runtime_error(
        "RefitPairPoseFromStereoOuterObservations requires diagnostics.");
  }
  diagnostics->used_symmetric_refit = true;
  ++runtime_counters->symmetric_refit_call_count;

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
    diagnostics->pose_success = have_seed;
    diagnostics->refit_fell_back_to_seed = true;
    ++runtime_counters->symmetric_refit_fallback_count;
    ++runtime_counters->runtime_guard_trigger_count;
    return true;
  }

  Eigen::Isometry3d current_pose = ToIsometry3d(seed_pose);
  double current_rmse =
      EvaluateStereoOuterPoseRmse(stereo_observations, scene_state, current_pose);
  if (!std::isfinite(current_rmse)) {
    *T_cam0_world = seed_pose;
    diagnostics->pose_success = have_seed;
    diagnostics->refit_fell_back_to_seed = true;
    ++runtime_counters->symmetric_refit_fallback_count;
    ++runtime_counters->runtime_guard_trigger_count;
    return true;
  }

  const double step = options.symmetric_refit_step;
  bool refined = false;
  for (int iteration = 0; iteration < options.symmetric_refit_max_iterations; ++iteration) {
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
    diagnostics->pose_success = true;
    diagnostics->improved = true;
    ++runtime_counters->symmetric_refit_improved_count;
    return true;
  }

  *T_cam0_world = seed_pose;
  diagnostics->pose_success = have_seed;
  diagnostics->refit_fell_back_to_seed = true;
  ++runtime_counters->symmetric_refit_fallback_count;
  return have_seed;
}

StereoPairOnlyBaInitSummary RunPairOnlyStereoBaInitialization(
    const StereoMeasurementDataset& dataset,
    const StereoExtrinsicSolverOptions& options,
    StereoSceneState* scene_state);

StereoInitializationDiagnostics InitializeStereoScene(
    const StereoMeasurementDataset& dataset,
    const StereoExtrinsicSolverOptions& options,
    StereoPairOnlyBaInitSummary* pair_init_summary,
    StereoSceneState* scene_state) {
  if (scene_state == nullptr) {
    throw std::runtime_error("InitializeStereoScene requires scene_state.");
  }
  StereoInitializationDiagnostics diagnostics;
  std::vector<TransformCandidateWithMeta> candidates;

  std::cout << "[Stage6] initialization: collecting shared-board stereo extrinsic candidates..."
            << std::endl;

  for (int pair_index : dataset.training_pair_indices) {
    const int good_shared_board_count =
        SharedBoardQualityHardGateEnabled(options)
            ? CountQualitySharedBoards(dataset, *scene_state, options, pair_index)
            : 0;
    const bool has_enough_shared_boards =
        SharedBoardQualityHardGateEnabled(options)
            ? good_shared_board_count >=
                  std::max(options.min_shared_boards_for_extrinsic_candidate,
                           options.shared_board_quality_min_good_shared_boards)
            : PairHasSharedBoardCandidate(
                  dataset, pair_index,
                  options.min_shared_boards_for_extrinsic_candidate);
    if (!has_enough_shared_boards) {
      ++diagnostics.excluded_candidate_count;
      diagnostics.excluded_candidate_reasons.push_back(
          "pair=" + std::to_string(pair_index) +
          (SharedBoardQualityHardGateEnabled(options)
               ? " no_quality_shared_board_candidate"
               : " no_shared_board_candidate"));
      continue;
    }
    const std::vector<int> board_ids = SortedBoardIds(dataset, pair_index);
    for (int board_id : board_ids) {
      const StereoSharedBoardQuality quality =
          EvaluateSharedBoardQuality(dataset, *scene_state, options, pair_index,
                                     board_id);
      if (!quality.have_cam0 || !quality.have_cam1) {
        ++diagnostics.excluded_candidate_count;
        diagnostics.excluded_candidate_reasons.push_back(
            "pair=" + std::to_string(pair_index) + " board=" +
            std::to_string(board_id) + " no_shared_board_candidate");
        continue;
      }
      if (SharedBoardQualityHardGateEnabled(options) && !quality.pass) {
        ++diagnostics.candidate_rejected_pose_fit_count;
        continue;
      }
      TransformCandidateWithMeta candidate;
      candidate.pair_index = pair_index;
      candidate.board_id = board_id;
      candidate.transform =
          quality.T_cam1_board * quality.T_cam0_board.inverse();
      candidates.push_back(candidate);
    }
  }

  diagnostics.candidate_count = static_cast<int>(candidates.size());
  std::cout << "[Stage6] initialization: raw candidate_count="
            << diagnostics.candidate_count
            << ", excluded_candidate_count="
            << diagnostics.excluded_candidate_count << std::endl;
  if (candidates.empty()) {
    diagnostics.failure_reason =
        "No valid stereo extrinsic candidates from shared board observations.";
    return diagnostics;
  }

  std::vector<TransformCandidate> extrinsic_candidates;
  extrinsic_candidates.reserve(candidates.size());
  for (const TransformCandidateWithMeta& candidate : candidates) {
    TransformCandidate transform_candidate;
    transform_candidate.transform = candidate.transform;
    transform_candidate.weight = 1.0;
    extrinsic_candidates.push_back(transform_candidate);
  }
  const double consistency_rotation_radians =
      DegreesToRadians(options.candidate_consistency_max_rotation_deg);
  int rejected_for_consistency = 0;
  const std::vector<TransformCandidate> filtered_extrinsic_candidates =
      FilterConsistentCandidates(extrinsic_candidates,
                                 consistency_rotation_radians,
                                 options.candidate_consistency_max_translation_m,
                                 &rejected_for_consistency);
  diagnostics.candidate_rejected_consistency_count += rejected_for_consistency;
  if (filtered_extrinsic_candidates.empty()) {
    diagnostics.failure_reason =
        "No stereo extrinsic candidates survived SE(3) consistency guard.";
    return diagnostics;
  }

  int medoid_index = 0;
  double best_score = std::numeric_limits<double>::infinity();
  for (std::size_t i = 0; i < filtered_extrinsic_candidates.size(); ++i) {
    double score = 0.0;
    for (std::size_t j = 0; j < filtered_extrinsic_candidates.size(); ++j) {
      score += TransformDistanceScore(filtered_extrinsic_candidates[i].transform,
                                      filtered_extrinsic_candidates[j].transform);
    }
    if (score < best_score) {
      best_score = score;
      medoid_index = static_cast<int>(i);
    }
  }
  diagnostics.medoid_score = best_score;
  scene_state->T_cam1_cam0 =
      ToMatrix4d(filtered_extrinsic_candidates[static_cast<std::size_t>(medoid_index)].transform);
  std::cout << "[Stage6] initialization: selected extrinsic medoid from "
            << filtered_extrinsic_candidates.size()
            << " consistent candidates, medoid_score="
            << diagnostics.medoid_score << std::endl;
  if (options.enable_pair_only_stereo_ba_init) {
    if (pair_init_summary == nullptr) {
      throw std::runtime_error(
          "Pair-only stereo init enabled but summary output is null.");
    }
    *pair_init_summary =
        RunPairOnlyStereoBaInitialization(dataset, options, scene_state);
    if (!pair_init_summary->success) {
      diagnostics.warnings.push_back(
          "pair_only_stereo_ba_init_failed: " +
          pair_init_summary->failure_reason);
    } else {
      std::cout << "[Stage6] initialization: pair-only stereo init refined "
                << "baseline_length=" << pair_init_summary->pair_ba_baseline_length
                << ", before_rmse=" << pair_init_summary->before_shared_rmse
                << ", after_rmse=" << pair_init_summary->after_shared_rmse
                << std::endl;
    }
  }

  scene_state->T_world_board_by_id[scene_state->gauge_fixed_board_id] =
      Eigen::Matrix4d::Identity();
  const std::vector<int> training_board_ids = CollectTrainingBoardIds(dataset);
  std::map<std::pair<int, int>, PairBoardGraphEdge> graph_edges;
  std::map<int, std::vector<int> > pair_to_boards;
  std::map<int, std::vector<int> > board_to_pairs;
  std::set<int> all_pairs(dataset.training_pair_indices.begin(),
                          dataset.training_pair_indices.end());
  std::set<int> all_boards(training_board_ids.begin(), training_board_ids.end());

  for (int pair_index : dataset.training_pair_indices) {
    for (int board_id : SortedBoardIds(dataset, pair_index)) {
      PairBoardGraphEdge edge;
      if (!BuildPairBoardGraphEdge(dataset, scene_state->cam0, scene_state->cam1,
                                   options, pair_index, board_id, &edge)) {
        continue;
      }
      if (SharedBoardQualityHardGateEnabled(options) && !edge.shared_stereo) {
        continue;
      }
      graph_edges[std::make_pair(pair_index, board_id)] = edge;
      pair_to_boards[pair_index].push_back(board_id);
      board_to_pairs[board_id].push_back(pair_index);
    }
  }

  std::set<int> graph_reachable_pairs;
  std::set<int> graph_reachable_boards;
  std::set<int> visited_pairs_for_component;
  std::set<int> visited_boards_for_component;
  int next_component_id = 0;
  auto assign_component_from_pair = [&](int seed_pair) {
    if (visited_pairs_for_component.count(seed_pair) > 0) {
      return;
    }
    std::queue<std::pair<bool, int> > queue;
    queue.push(std::make_pair(true, seed_pair));
    visited_pairs_for_component.insert(seed_pair);
    diagnostics.pair_component_ids[seed_pair] = next_component_id;
    while (!queue.empty()) {
      const std::pair<bool, int> node = queue.front();
      queue.pop();
      if (node.first) {
        const auto pair_it = pair_to_boards.find(node.second);
        if (pair_it == pair_to_boards.end()) {
          continue;
        }
        for (int board_id : pair_it->second) {
          if (visited_boards_for_component.insert(board_id).second) {
            diagnostics.board_component_ids[board_id] = next_component_id;
            queue.push(std::make_pair(false, board_id));
          }
        }
      } else {
        const auto board_it = board_to_pairs.find(node.second);
        if (board_it == board_to_pairs.end()) {
          continue;
        }
        for (int pair_index : board_it->second) {
          if (visited_pairs_for_component.insert(pair_index).second) {
            diagnostics.pair_component_ids[pair_index] = next_component_id;
            queue.push(std::make_pair(true, pair_index));
          }
        }
      }
    }
    ++next_component_id;
  };

  for (int pair_index : dataset.training_pair_indices) {
    assign_component_from_pair(pair_index);
  }
  for (int board_id : training_board_ids) {
    if (visited_boards_for_component.count(board_id) == 0) {
      diagnostics.board_component_ids[board_id] = next_component_id++;
    }
  }
  diagnostics.connected_component_count = next_component_id;
  const auto gauge_component_it =
      diagnostics.board_component_ids.find(scene_state->gauge_fixed_board_id);
  if (gauge_component_it != diagnostics.board_component_ids.end()) {
    diagnostics.gauge_connected_component_id = gauge_component_it->second;
  }

  for (const auto& entry : diagnostics.pair_component_ids) {
    if (entry.second == diagnostics.gauge_connected_component_id) {
      ++diagnostics.gauge_connected_pair_count;
    }
  }
  for (const auto& entry : diagnostics.board_component_ids) {
    if (entry.second == diagnostics.gauge_connected_component_id) {
      ++diagnostics.gauge_connected_board_count;
    }
  }

  std::set<int> initialized_pairs;
  std::set<int> initialized_boards;
  initialized_boards.insert(scene_state->gauge_fixed_board_id);
  graph_reachable_boards.insert(scene_state->gauge_fixed_board_id);
  std::vector<int> pair_frontier;
  std::vector<int> board_frontier(1, scene_state->gauge_fixed_board_id);
  for (int pair_index : dataset.training_pair_indices) {
    const auto edge_it =
        graph_edges.find(std::make_pair(pair_index, scene_state->gauge_fixed_board_id));
    if (edge_it == graph_edges.end()) {
      continue;
    }
    const PairBoardGraphEdge& edge = edge_it->second;
    std::vector<BootstrapTransformCandidate> seed_candidates;
    if (edge.cam0_valid) {
      BootstrapTransformCandidate candidate;
      candidate.transform = edge.T_cam0_board;
      candidate.is_shared_stereo = edge.shared_stereo;
      candidate.outer_point_count = edge.cam0_outer_point_count;
      candidate.pose_fit_rmse = edge.cam0_rmse;
      candidate.source_pair_index = pair_index;
      candidate.source_board_id = scene_state->gauge_fixed_board_id;
      candidate.source_camera_index = 0;
      seed_candidates.push_back(candidate);
      ++diagnostics.pair_pose_candidate_count;
    } else if (edge.cam0_outer_point_count > 0) {
      ++diagnostics.candidate_rejected_pose_fit_count;
    }
    if (edge.cam1_valid) {
      BootstrapTransformCandidate candidate;
      candidate.transform =
          ToIsometry3d(scene_state->T_cam1_cam0).inverse() * edge.T_cam1_board;
      candidate.is_shared_stereo = edge.shared_stereo;
      candidate.outer_point_count = edge.cam1_outer_point_count;
      candidate.pose_fit_rmse = edge.cam1_rmse;
      candidate.source_pair_index = pair_index;
      candidate.source_board_id = scene_state->gauge_fixed_board_id;
      candidate.source_camera_index = 1;
      seed_candidates.push_back(candidate);
      ++diagnostics.pair_pose_candidate_count;
    } else if (edge.cam1_outer_point_count > 0) {
      ++diagnostics.candidate_rejected_pose_fit_count;
    }
    BootstrapTransformCandidate selected_seed;
    int rejected_seed = 0;
    if (!SelectBootstrapCandidate(seed_candidates, options, &rejected_seed,
                                  &selected_seed)) {
      diagnostics.candidate_rejected_consistency_count += rejected_seed;
      continue;
    }
    diagnostics.candidate_rejected_consistency_count += rejected_seed;
    if (initialized_pairs.insert(pair_index).second) {
      scene_state->T_cam0_world_by_pair[pair_index] =
          ToMatrix4d(selected_seed.transform);
      pair_frontier.push_back(pair_index);
      graph_reachable_pairs.insert(pair_index);
    }
  }
  diagnostics.graph_seed_pair_count = static_cast<int>(initialized_pairs.size());
  std::cout << "[Stage6] initialization: graph seed pairs="
            << diagnostics.graph_seed_pair_count << std::endl;

  int total_new_pairs = 0;
  int total_new_boards = 0;
  int propagation_iterations = 0;
  const Eigen::Isometry3d T_cam1_cam0 = ToIsometry3d(scene_state->T_cam1_cam0);
  for (; propagation_iterations < options.max_graph_propagation_iterations;
       ++propagation_iterations) {
    std::vector<int> next_pair_frontier;
    std::vector<int> next_board_frontier;
    int iteration_new_pairs = 0;
    int iteration_new_boards = 0;

    for (int board_id : board_frontier) {
      const auto board_pose_it = scene_state->T_world_board_by_id.find(board_id);
      if (board_pose_it == scene_state->T_world_board_by_id.end()) {
        continue;
      }
      const Eigen::Isometry3d T_world_board =
          ToIsometry3d(board_pose_it->second);
      const auto board_pairs_it = board_to_pairs.find(board_id);
      if (board_pairs_it == board_to_pairs.end()) {
        continue;
      }
      for (int pair_index : board_pairs_it->second) {
        graph_reachable_pairs.insert(pair_index);
        if (initialized_pairs.count(pair_index) > 0) {
          continue;
        }
        const PairBoardGraphEdge& edge =
            graph_edges[std::make_pair(pair_index, board_id)];
        std::vector<BootstrapTransformCandidate> pair_candidates;
        if (edge.cam0_valid) {
          BootstrapTransformCandidate candidate;
          candidate.transform = edge.T_cam0_board * T_world_board.inverse();
          candidate.is_shared_stereo = edge.shared_stereo;
          candidate.outer_point_count = edge.cam0_outer_point_count;
          candidate.pose_fit_rmse = edge.cam0_rmse;
          candidate.source_pair_index = pair_index;
          candidate.source_board_id = board_id;
          candidate.source_camera_index = 0;
          pair_candidates.push_back(candidate);
          ++diagnostics.pair_pose_candidate_count;
        } else if (edge.cam0_outer_point_count > 0) {
          ++diagnostics.candidate_rejected_pose_fit_count;
        }
        if (edge.cam1_valid) {
          BootstrapTransformCandidate candidate;
          candidate.transform =
              T_cam1_cam0.inverse() * edge.T_cam1_board * T_world_board.inverse();
          candidate.is_shared_stereo = edge.shared_stereo;
          candidate.outer_point_count = edge.cam1_outer_point_count;
          candidate.pose_fit_rmse = edge.cam1_rmse;
          candidate.source_pair_index = pair_index;
          candidate.source_board_id = board_id;
          candidate.source_camera_index = 1;
          pair_candidates.push_back(candidate);
          ++diagnostics.pair_pose_candidate_count;
        } else if (edge.cam1_outer_point_count > 0) {
          ++diagnostics.candidate_rejected_pose_fit_count;
        }
        BootstrapTransformCandidate selected_candidate;
        int rejected_count = 0;
        if (!SelectBootstrapCandidate(pair_candidates, options, &rejected_count,
                                      &selected_candidate)) {
          diagnostics.candidate_rejected_consistency_count += rejected_count;
          diagnostics.warnings.push_back(
              "pair=" + std::to_string(pair_index) + " board=" +
              std::to_string(board_id) + " pose_estimation_failed");
          continue;
        }
        diagnostics.candidate_rejected_consistency_count += rejected_count;
        scene_state->T_cam0_world_by_pair[pair_index] =
            ToMatrix4d(selected_candidate.transform);
        initialized_pairs.insert(pair_index);
        next_pair_frontier.push_back(pair_index);
        ++iteration_new_pairs;
      }
    }

    for (int pair_index : pair_frontier) {
      const auto pair_pose_it = scene_state->T_cam0_world_by_pair.find(pair_index);
      if (pair_pose_it == scene_state->T_cam0_world_by_pair.end()) {
        continue;
      }
      const Eigen::Isometry3d T_cam0_world = ToIsometry3d(pair_pose_it->second);
      const auto pair_boards_it = pair_to_boards.find(pair_index);
      if (pair_boards_it == pair_to_boards.end()) {
        continue;
      }
      for (int board_id : pair_boards_it->second) {
        graph_reachable_boards.insert(board_id);
        if (initialized_boards.count(board_id) > 0) {
          continue;
        }
        const PairBoardGraphEdge& edge =
            graph_edges[std::make_pair(pair_index, board_id)];
        std::vector<BootstrapTransformCandidate> board_candidates;
        if (edge.cam0_valid) {
          BootstrapTransformCandidate candidate;
          candidate.transform = T_cam0_world.inverse() * edge.T_cam0_board;
          candidate.is_shared_stereo = edge.shared_stereo;
          candidate.outer_point_count = edge.cam0_outer_point_count;
          candidate.pose_fit_rmse = edge.cam0_rmse;
          candidate.source_pair_index = pair_index;
          candidate.source_board_id = board_id;
          candidate.source_camera_index = 0;
          board_candidates.push_back(candidate);
          ++diagnostics.board_pose_candidate_count;
        } else if (edge.cam0_outer_point_count > 0) {
          ++diagnostics.candidate_rejected_pose_fit_count;
        }
        if (edge.cam1_valid) {
          BootstrapTransformCandidate candidate;
          candidate.transform =
              T_cam0_world.inverse() * T_cam1_cam0.inverse() * edge.T_cam1_board;
          candidate.is_shared_stereo = edge.shared_stereo;
          candidate.outer_point_count = edge.cam1_outer_point_count;
          candidate.pose_fit_rmse = edge.cam1_rmse;
          candidate.source_pair_index = pair_index;
          candidate.source_board_id = board_id;
          candidate.source_camera_index = 1;
          board_candidates.push_back(candidate);
          ++diagnostics.board_pose_candidate_count;
        } else if (edge.cam1_outer_point_count > 0) {
          ++diagnostics.candidate_rejected_pose_fit_count;
        }
        BootstrapTransformCandidate selected_candidate;
        int rejected_count = 0;
        if (!SelectBootstrapCandidate(board_candidates, options, &rejected_count,
                                      &selected_candidate)) {
          diagnostics.candidate_rejected_consistency_count += rejected_count;
          diagnostics.warnings.push_back(
              "pair=" + std::to_string(pair_index) + " board=" +
              std::to_string(board_id) + " pose_estimation_failed");
          continue;
        }
        diagnostics.candidate_rejected_consistency_count += rejected_count;
        scene_state->T_world_board_by_id[board_id] =
            ToMatrix4d(selected_candidate.transform);
        initialized_boards.insert(board_id);
        next_board_frontier.push_back(board_id);
        ++iteration_new_boards;
      }
    }

    total_new_pairs += iteration_new_pairs;
    total_new_boards += iteration_new_boards;
    std::cout << "[Stage6] graph propagation iteration "
              << (propagation_iterations + 1)
              << " new_pairs=" << iteration_new_pairs
              << " new_boards=" << iteration_new_boards
              << " total_initialized_pairs=" << initialized_pairs.size()
              << " total_initialized_boards=" << initialized_boards.size()
              << std::endl;

    if (iteration_new_pairs == 0 && iteration_new_boards == 0) {
      diagnostics.graph_propagation_stopped_by_no_progress = true;
      break;
    }
    pair_frontier.swap(next_pair_frontier);
    board_frontier.swap(next_board_frontier);
  }
  if (!diagnostics.graph_propagation_stopped_by_no_progress &&
      propagation_iterations >= options.max_graph_propagation_iterations) {
    diagnostics.graph_propagation_stopped_by_iteration_limit = true;
    diagnostics.warnings.push_back(
        "graph_propagation_reached_iteration_limit");
  }
  diagnostics.graph_propagation_iteration_count =
      std::min(propagation_iterations + 1,
               options.max_graph_propagation_iterations);
  diagnostics.graph_propagation_new_pair_count = total_new_pairs;
  diagnostics.graph_propagation_new_board_count = total_new_boards;

  for (int pair_index : dataset.training_pair_indices) {
    if (scene_state->T_cam0_world_by_pair.count(pair_index) == 0) {
      scene_state->excluded_training_pair_indices.insert(pair_index);
      if (graph_reachable_pairs.count(pair_index) > 0) {
        ++diagnostics.excluded_training_pair_count;
        diagnostics.warnings.push_back(
            "pair=" + std::to_string(pair_index) + " pose_estimation_failed");
      } else {
        diagnostics.warnings.push_back(
            "pair=" + std::to_string(pair_index) +
            " graph_unreachable_from_gauge");
      }
    }
  }

  for (int pair_index : dataset.training_pair_indices) {
    if (graph_reachable_pairs.count(pair_index) > 0) {
      diagnostics.reachable_training_pair_indices.push_back(pair_index);
    } else {
      diagnostics.unreachable_training_pair_indices.push_back(pair_index);
    }
  }
  for (int board_id : training_board_ids) {
    if (graph_reachable_boards.count(board_id) > 0) {
      diagnostics.reachable_board_ids.push_back(board_id);
    } else {
      diagnostics.unreachable_board_ids.push_back(board_id);
    }
  }
  std::sort(diagnostics.reachable_training_pair_indices.begin(),
            diagnostics.reachable_training_pair_indices.end());
  std::sort(diagnostics.unreachable_training_pair_indices.begin(),
            diagnostics.unreachable_training_pair_indices.end());
  std::sort(diagnostics.reachable_board_ids.begin(),
            diagnostics.reachable_board_ids.end());
  diagnostics.reachable_board_ids.erase(
      std::unique(diagnostics.reachable_board_ids.begin(),
                  diagnostics.reachable_board_ids.end()),
      diagnostics.reachable_board_ids.end());
  std::sort(diagnostics.unreachable_board_ids.begin(),
            diagnostics.unreachable_board_ids.end());
  diagnostics.unreachable_board_ids.erase(
      std::unique(diagnostics.unreachable_board_ids.begin(),
                  diagnostics.unreachable_board_ids.end()),
      diagnostics.unreachable_board_ids.end());
  diagnostics.reachable_training_pair_count =
      static_cast<int>(diagnostics.reachable_training_pair_indices.size());
  diagnostics.unreachable_training_pair_count =
      static_cast<int>(diagnostics.unreachable_training_pair_indices.size());
  diagnostics.initialized_training_pair_count =
      static_cast<int>(scene_state->T_cam0_world_by_pair.size());
  diagnostics.initialized_board_count =
      static_cast<int>(scene_state->T_world_board_by_id.size());
  diagnostics.uninitialized_board_count =
      static_cast<int>(all_boards.size()) - diagnostics.initialized_board_count;
  diagnostics.uninitialized_training_pair_count =
      static_cast<int>(scene_state->excluded_training_pair_indices.size());
  std::cout << "[Stage6] initialization: reachable_pairs="
            << diagnostics.reachable_training_pair_count
            << ", unreachable_pairs="
            << diagnostics.unreachable_training_pair_count
            << ", initialized_pairs="
            << diagnostics.initialized_training_pair_count
            << ", initialized_boards="
            << diagnostics.initialized_board_count << std::endl;
  diagnostics.success = diagnostics.initialized_training_pair_count > 0 &&
                        diagnostics.initialized_board_count > 0;
  if (!diagnostics.success && diagnostics.failure_reason.empty()) {
    diagnostics.failure_reason =
        "Stereo initialization could not seed training pair poses and board poses.";
  }
  return diagnostics;
}

bool OptimizeSceneAlternating(const StereoMeasurementDataset& dataset,
                              const StereoExtrinsicSolverOptions& options,
                              RuntimeCounters* runtime_counters,
                              StereoSceneState* scene_state) {
  if (scene_state == nullptr) {
    throw std::runtime_error("OptimizeSceneAlternating requires scene_state.");
  }
  if (runtime_counters == nullptr) {
    throw std::runtime_error("OptimizeSceneAlternating requires runtime counters.");
  }
  const double consistency_rotation_radians =
      DegreesToRadians(options.candidate_consistency_max_rotation_deg);
  StereoResidualEvaluator training_evaluator(
      StereoResidualEvaluationOptions{
          false,
          options.pair_pose_refit_mode,
          options.symmetric_refit_max_iterations,
          options.symmetric_refit_step,
          false});
  StereoResidualSummary previous_summary = training_evaluator.Evaluate(
      dataset, *scene_state,
      std::set<int>(dataset.training_pair_indices.begin(),
                    dataset.training_pair_indices.end()),
      "training");
  std::cout << "[Stage6] optimization: initial training RMSE="
            << previous_summary.total_stereo_rmse
            << " (cam0=" << previous_summary.cam0_rmse
            << ", cam1=" << previous_summary.cam1_rmse << ")"
            << std::endl;
  for (int iteration = 0; iteration < options.max_iterations; ++iteration) {
    const StereoSceneState previous_scene = *scene_state;
    bool updated = false;
    std::cout << "[Stage6] optimization: alternating iteration "
              << (iteration + 1) << "/" << options.max_iterations << "..."
              << std::endl;
    for (int pair_index : dataset.training_pair_indices) {
      if (scene_state->excluded_training_pair_indices.count(pair_index) > 0) {
        continue;
      }
      Eigen::Matrix4d refit_pose = Eigen::Matrix4d::Identity();
      StereoRefitDiagnostics diagnostics;
      const bool pose_ok =
          options.pair_pose_refit_mode == StereoPairPoseRefitMode::StereoSymmetric
              ? RefitPairPoseFromStereoOuterObservations(
                    dataset, *scene_state, pair_index, options, runtime_counters,
                    &diagnostics, &refit_pose)
              : RefitPairPoseFromOuterObservations(dataset, *scene_state,
                                                   pair_index, &refit_pose);
      if (pose_ok) {
        const double delta = ComputePoseDeltaNorm(
            ToIsometry3d(scene_state->T_cam0_world_by_pair[pair_index]),
            ToIsometry3d(refit_pose));
        scene_state->T_cam0_world_by_pair[pair_index] = refit_pose;
        updated = updated || (delta > options.convergence_threshold);
      }
    }

    for (const auto& pair_entry : scene_state->T_cam0_world_by_pair) {
      const int pair_index = pair_entry.first;
      const Eigen::Isometry3d T_cam0_world = ToIsometry3d(pair_entry.second);
      for (int board_id : SortedBoardIds(dataset, pair_index)) {
        if (board_id == scene_state->gauge_fixed_board_id) {
          continue;
        }
        std::vector<TransformCandidate> board_candidates;
        Eigen::Isometry3d T_cam0_board = Eigen::Isometry3d::Identity();
        double cam0_rmse = 0.0;
        if (EstimateCameraBoardPoseWithRmse(dataset, scene_state->cam0, pair_index, 0,
                                            board_id, &T_cam0_board, &cam0_rmse) &&
            cam0_rmse <= options.pose_fit_guard_threshold_px) {
          TransformCandidate candidate;
          candidate.transform = T_cam0_world.inverse() * T_cam0_board;
          candidate.weight = 1.0;
          board_candidates.push_back(candidate);
        }
        Eigen::Isometry3d T_cam1_board = Eigen::Isometry3d::Identity();
        double cam1_rmse = 0.0;
        if (EstimateCameraBoardPoseWithRmse(dataset, scene_state->cam1, pair_index, 1,
                                            board_id, &T_cam1_board, &cam1_rmse) &&
            cam1_rmse <= options.pose_fit_guard_threshold_px) {
          const Eigen::Isometry3d T_cam1_cam0 =
              ToIsometry3d(scene_state->T_cam1_cam0);
          TransformCandidate candidate;
          candidate.transform =
              T_cam0_world.inverse() * T_cam1_cam0.inverse() * T_cam1_board;
          candidate.weight = 1.0;
          board_candidates.push_back(candidate);
        }
        int rejected_board_candidates = 0;
        const std::vector<TransformCandidate> filtered_board_candidates =
            FilterConsistentCandidates(board_candidates,
                                       consistency_rotation_radians,
                                       options.candidate_consistency_max_translation_m,
                                       &rejected_board_candidates);
        if (filtered_board_candidates.empty()) {
          continue;
        }
        scene_state->T_world_board_by_id[board_id] =
            ToMatrix4d(AverageTransforms(filtered_board_candidates));
        updated = true;
      }
    }

    std::vector<TransformCandidateWithMeta> stereo_candidates;
    for (int pair_index : dataset.training_pair_indices) {
      if (scene_state->excluded_training_pair_indices.count(pair_index) > 0) {
        continue;
      }
      for (int board_id : SortedBoardIds(dataset, pair_index)) {
        Eigen::Isometry3d T_cam0_board = Eigen::Isometry3d::Identity();
        Eigen::Isometry3d T_cam1_board = Eigen::Isometry3d::Identity();
        if (!EstimateCameraBoardPose(dataset, scene_state->cam0, pair_index, 0,
                                     board_id, &T_cam0_board) ||
            !EstimateCameraBoardPose(dataset, scene_state->cam1, pair_index, 1,
                                     board_id, &T_cam1_board)) {
          continue;
        }
        TransformCandidateWithMeta candidate;
        candidate.transform = T_cam1_board * T_cam0_board.inverse();
        stereo_candidates.push_back(candidate);
      }
    }
    if (!stereo_candidates.empty()) {
      int medoid_index = 0;
      double best_score = std::numeric_limits<double>::infinity();
      for (std::size_t i = 0; i < stereo_candidates.size(); ++i) {
        double score = 0.0;
        for (std::size_t j = 0; j < stereo_candidates.size(); ++j) {
          score += TransformDistanceScore(
              stereo_candidates[i].transform, stereo_candidates[j].transform);
        }
        if (score < best_score) {
          best_score = score;
          medoid_index = static_cast<int>(i);
        }
      }
      const Eigen::Isometry3d previous = ToIsometry3d(scene_state->T_cam1_cam0);
      scene_state->T_cam1_cam0 = ToMatrix4d(stereo_candidates[medoid_index].transform);
      updated = updated || (TransformDistanceScore(
                                 previous,
                                 stereo_candidates[medoid_index].transform) >
                             options.convergence_threshold);
    }
    const StereoResidualSummary current_summary = training_evaluator.Evaluate(
        dataset, *scene_state,
        std::set<int>(dataset.training_pair_indices.begin(),
                      dataset.training_pair_indices.end()),
        "training");
    const double total_improvement =
        previous_summary.total_stereo_rmse - current_summary.total_stereo_rmse;
    std::cout << "[Stage6] optimization: iteration " << (iteration + 1)
              << " training RMSE=" << current_summary.total_stereo_rmse
              << " (cam0=" << current_summary.cam0_rmse
              << ", cam1=" << current_summary.cam1_rmse
              << "), improvement=" << total_improvement << std::endl;
    const bool cam1_worsened =
        current_summary.cam1_rmse > previous_summary.cam1_rmse + 1e-9;
    if (cam1_worsened && total_improvement < options.convergence_threshold) {
      *scene_state = previous_scene;
      std::cout << "[Stage6] optimization: rolling back iteration "
                << (iteration + 1)
                << " because cam1 RMSE worsened without total improvement."
                << std::endl;
      return true;
    }
    previous_summary = current_summary;
    if (!updated) {
      std::cout << "[Stage6] optimization: converged because no state update remained."
                << std::endl;
      return true;
    }
    if (total_improvement >= 0.0 &&
        total_improvement < options.convergence_threshold) {
      std::cout << "[Stage6] optimization: converged because total improvement "
                << total_improvement << " < threshold "
                << options.convergence_threshold << std::endl;
      return true;
    }
  }
  return true;
}

StereoViewSelectionScore ComputePairSelectionScore(
    const StereoMeasurementDataset& dataset,
    const StereoSceneState& scene_state,
    const StereoExtrinsicSolverOptions& options,
    int pair_index) {
  StereoViewSelectionScore score;
  score.shared_board_count = 0;
  score.shared_outer_point_count = 0;
  score.single_camera_only_board_count = 0;
  double rmse_sum = 0.0;
  int rmse_count = 0;
  for (int board_id : SortedBoardIds(dataset, pair_index)) {
    const bool has_cam0 = PairHasBoardVisibleInCamera(dataset, pair_index, 0, board_id);
    const bool has_cam1 = PairHasBoardVisibleInCamera(dataset, pair_index, 1, board_id);
    if (has_cam0 && has_cam1) {
      const StereoSharedBoardQuality quality =
          EvaluateSharedBoardQuality(dataset, scene_state, options, pair_index,
                                     board_id);
      if (SharedBoardQualityHardGateEnabled(options) && !quality.pass) {
        ++score.single_camera_only_board_count;
        continue;
      }
      ++score.shared_board_count;
      score.shared_outer_point_count +=
          quality.cam0_outer_point_count + quality.cam1_outer_point_count;
      if (quality.have_cam0) {
        rmse_sum += quality.cam0_outer_rmse;
        ++rmse_count;
      }
      if (quality.have_cam1) {
        rmse_sum += quality.cam1_outer_rmse;
        ++rmse_count;
      }
    } else if (has_cam0 || has_cam1) {
      ++score.single_camera_only_board_count;
    }
  }
  score.pose_fit_rmse =
      rmse_count > 0 ? rmse_sum / static_cast<double>(rmse_count)
                     : std::numeric_limits<double>::infinity();
  return score;
}

bool IsSelectionRowBetter(const StereoPairSelectionRow& lhs,
                          const StereoPairSelectionRow& rhs) {
  if (lhs.score_shared_board_count != rhs.score_shared_board_count) {
    return lhs.score_shared_board_count > rhs.score_shared_board_count;
  }
  if (lhs.score_shared_outer_point_count != rhs.score_shared_outer_point_count) {
    return lhs.score_shared_outer_point_count > rhs.score_shared_outer_point_count;
  }
  if (std::abs(lhs.score_pose_fit_rmse - rhs.score_pose_fit_rmse) > 1e-12) {
    return lhs.score_pose_fit_rmse < rhs.score_pose_fit_rmse;
  }
  if (lhs.score_single_camera_only_board_count !=
      rhs.score_single_camera_only_board_count) {
    return lhs.score_single_camera_only_board_count <
           rhs.score_single_camera_only_board_count;
  }
  return lhs.pair_index < rhs.pair_index;
}

double ComputeMedian(std::vector<double> values) {
  if (values.empty()) {
    return std::numeric_limits<double>::infinity();
  }
  std::sort(values.begin(), values.end());
  const std::size_t mid = values.size() / 2;
  if ((values.size() % 2) == 1) {
    return values[mid];
  }
  return 0.5 * (values[mid - 1] + values[mid]);
}

double ComputeMean(const std::vector<double>& values) {
  if (values.empty()) {
    return 0.0;
  }
  return std::accumulate(values.begin(), values.end(), 0.0) /
         static_cast<double>(values.size());
}

double ComputeStddev(const std::vector<double>& values, double mean) {
  if (values.empty()) {
    return 0.0;
  }
  double squared_sum = 0.0;
  for (double value : values) {
    squared_sum += (value - mean) * (value - mean);
  }
  return std::sqrt(squared_sum / static_cast<double>(values.size()));
}

int CountStereoObservations(const StereoMeasurementDataset& dataset,
                            int pair_index,
                            int camera_index,
                            int board_id,
                            JointPointType point_type) {
  int count = 0;
  for (const StereoObservation& observation : dataset.observations) {
    if (observation.pair_index == pair_index &&
        observation.camera_index == camera_index &&
        observation.board_id == board_id &&
        observation.point_type == point_type &&
        observation.used_in_solver) {
      ++count;
    }
  }
  return count;
}

double EvaluateSharedBoardCandidateRmse(
    const StereoMeasurementDataset& dataset,
    const StereoCameraFixedCalibration& cam0_calibration,
    const StereoCameraFixedCalibration& cam1_calibration,
    int pair_index,
    int board_id,
    const Eigen::Isometry3d& T_cam1_cam0,
    const Eigen::Isometry3d& T_cam0_board) {
  const DoubleSphereCameraModel cam0 =
      DoubleSphereCameraModel::FromConfig(MakeCameraConfig(cam0_calibration));
  const DoubleSphereCameraModel cam1 =
      DoubleSphereCameraModel::FromConfig(MakeCameraConfig(cam1_calibration));
  double squared_error_sum = 0.0;
  int count = 0;
  for (const StereoObservation& observation : dataset.observations) {
    if (observation.pair_index != pair_index ||
        observation.board_id != board_id ||
        !observation.used_in_solver) {
      continue;
    }
    const Eigen::Vector3d point_cam0 =
        T_cam0_board * observation.target_point_board;
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

template <typename GeometryT0, typename GeometryT1>
bool RunPairOnlyStereoBaTyped(
    const StereoMeasurementDataset& dataset,
    const StereoExtrinsicSolverOptions& options,
    const StereoCameraFixedCalibration& cam0_calibration,
    const StereoCameraFixedCalibration& cam1_calibration,
    const std::vector<TransformCandidateWithMeta>& accepted_candidates,
    const Eigen::Isometry3d& initial_baseline,
    Eigen::Isometry3d* refined_baseline,
    std::map<PairBoardKey, Eigen::Isometry3d>* optimized_local_poses,
    StereoPairOnlyBaInitSummary* summary,
    StereoCameraFixedCalibration* refined_cam0,
    StereoCameraFixedCalibration* refined_cam1) {
  if (refined_baseline == nullptr || optimized_local_poses == nullptr) {
    throw std::runtime_error(
        "RunPairOnlyStereoBaTyped requires valid baseline/local-pose outputs.");
  }
  if (accepted_candidates.empty()) {
    return false;
  }

  boost::shared_ptr<GeometryT0> cam0_geometry =
      MakeTypedStereoGeometry<GeometryT0>(cam0_calibration);
  boost::shared_ptr<GeometryT1> cam1_geometry =
      MakeTypedStereoGeometry<GeometryT1>(cam1_calibration);

  CameraDv<GeometryT0> cam0_dv(cam0_geometry);
  CameraDv<GeometryT1> cam1_dv(cam1_geometry);
  const bool optimize_projection =
      options.intrinsics_mode == StereoIntrinsicsMode::KalibrJointProjection;
  cam0_dv.setActive(optimize_projection, false, false);
  cam1_dv.setActive(optimize_projection, false, false);

  boost::shared_ptr<aslam::backend::OptimizationProblem> problem(
      new aslam::backend::OptimizationProblem);
  problem->addDesignVariable(cam0_dv.projectionDesignVariable());
  problem->addDesignVariable(cam0_dv.distortionDesignVariable());
  problem->addDesignVariable(cam0_dv.shutterDesignVariable());
  problem->addDesignVariable(cam1_dv.projectionDesignVariable());
  problem->addDesignVariable(cam1_dv.distortionDesignVariable());
  problem->addDesignVariable(cam1_dv.shutterDesignVariable());

  boost::shared_ptr<aslam::backend::MappedRotationQuaternion> baseline_rotation_dv;
  boost::shared_ptr<aslam::backend::MappedEuclideanPoint> baseline_translation_dv;
  sm::kinematics::Transformation baseline_transform(ToMatrix4d(initial_baseline));
  aslam::backend::TransformationExpression T_cam1_cam0_expr =
      aslam::backend::transformationToExpression(
          baseline_transform, baseline_rotation_dv, baseline_translation_dv);
  baseline_rotation_dv->setActive(options.final_ba_optimize_stereo_extrinsic);
  baseline_translation_dv->setActive(options.final_ba_optimize_stereo_extrinsic);
  problem->addDesignVariable(baseline_rotation_dv);
  problem->addDesignVariable(baseline_translation_dv);

  std::map<PairBoardKey, StereoPoseVariableState> local_pose_variables;
  std::set<PairBoardKey> accepted_keys;
  for (const TransformCandidateWithMeta& candidate : accepted_candidates) {
    const PairBoardKey key(candidate.pair_index, candidate.board_id);
    if (accepted_keys.count(key) > 0) {
      continue;
    }
    Eigen::Isometry3d T_cam0_board = Eigen::Isometry3d::Identity();
    double cam0_rmse = 0.0;
    if (!EstimateCameraBoardPoseWithRmse(dataset, cam0_calibration,
                                         candidate.pair_index, 0,
                                         candidate.board_id, &T_cam0_board,
                                         &cam0_rmse)) {
      continue;
    }
    StereoPoseVariableState& variable = local_pose_variables[key];
    variable.transform = sm::kinematics::Transformation(ToMatrix4d(T_cam0_board));
    variable.expression = aslam::backend::transformationToExpression(
        variable.transform, variable.rotation_dv, variable.translation_dv);
    variable.rotation_dv->setActive(options.final_ba_optimize_pair_poses);
    variable.translation_dv->setActive(options.final_ba_optimize_pair_poses);
    problem->addDesignVariable(variable.rotation_dv);
    problem->addDesignVariable(variable.translation_dv);
    accepted_keys.insert(key);
  }
  if (accepted_keys.empty()) {
    return false;
  }

  int reprojection_error_count = 0;
  for (const StereoObservation& observation : dataset.observations) {
    if (!observation.used_in_solver) {
      continue;
    }
    const PairBoardKey key(observation.pair_index, observation.board_id);
    const auto local_it = local_pose_variables.find(key);
    if (local_it == local_pose_variables.end()) {
      continue;
    }
    const aslam::backend::HomogeneousExpression point_board(
        observation.target_point_board);
    const aslam::backend::HomogeneousExpression point_cam0 =
        local_it->second.expression * point_board;
    const double weight = std::max(0.0, observation.weight);
    if (weight <= 0.0) {
      continue;
    }
    if (observation.camera_index == 0) {
      boost::shared_ptr<aslam::backend::ErrorTerm> error(
          new StereoCameraReprojectionError<GeometryT0>(
              observation.observed_image_xy, weight, point_cam0, cam0_dv, 100.0,
              options.pair_init_use_huber_loss));
      problem->addErrorTerm(error);
    } else {
      const aslam::backend::HomogeneousExpression point_cam1 =
          T_cam1_cam0_expr * point_cam0;
      boost::shared_ptr<aslam::backend::ErrorTerm> error(
          new StereoCameraReprojectionError<GeometryT1>(
              observation.observed_image_xy, weight, point_cam1, cam1_dv, 100.0,
              options.pair_init_use_huber_loss));
      problem->addErrorTerm(error);
    }
    ++reprojection_error_count;
  }
  if (reprojection_error_count <= 0) {
    return false;
  }

  aslam::backend::OptimizerOptions optimizer_options;
  optimizer_options.maxIterations = options.pair_init_max_iterations;
  optimizer_options.convergenceDeltaJ = options.pair_init_convergence_threshold;
  optimizer_options.convergenceDeltaX = options.pair_init_convergence_threshold;
  optimizer_options.levenbergMarquardtLambdaInit = 10.0;
  optimizer_options.doLevenbergMarquardt = true;
  optimizer_options.doSchurComplement = false;
  optimizer_options.verbose = false;
  aslam::backend::Optimizer optimizer(optimizer_options);
  optimizer.setProblem(problem);
  const aslam::backend::SolutionReturnValue solution = optimizer.optimize();
  const bool objective_finite = std::isfinite(solution.JStart) &&
                                std::isfinite(solution.JFinal);
  const bool objective_decreased =
      objective_finite && solution.JFinal <= solution.JStart;
  if (summary != nullptr) {
    summary->robust_loss_enabled = options.pair_init_use_huber_loss;
    summary->objective_finite = objective_finite;
    summary->objective_decreased = objective_decreased;
    summary->linear_solver_failure = solution.linearSolverFailure;
    summary->reached_max_iterations =
        solution.iterations >= optimizer_options.maxIterations;
    summary->optimization_iterations = solution.iterations;
    summary->objective_start = solution.JStart;
    summary->objective_final = solution.JFinal;
  }
  if (solution.linearSolverFailure || !objective_decreased ||
      !baseline_transform.T().allFinite()) {
    return false;
  }

  *refined_baseline = ToIsometry3d(baseline_transform.T());
  const StereoCameraFixedCalibration candidate_cam0 =
      UpdateStereoCalibrationFromGeometry(cam0_calibration, *cam0_geometry);
  const StereoCameraFixedCalibration candidate_cam1 =
      UpdateStereoCalibrationFromGeometry(cam1_calibration, *cam1_geometry);
  if (!IsStereoCalibrationModelValid(candidate_cam0) ||
      !IsStereoCalibrationModelValid(candidate_cam1)) {
    return false;
  }
  if (refined_cam0 != nullptr) {
    *refined_cam0 = candidate_cam0;
  }
  if (refined_cam1 != nullptr) {
    *refined_cam1 = candidate_cam1;
  }
  optimized_local_poses->clear();
  for (const auto& entry : local_pose_variables) {
    (*optimized_local_poses)[entry.first] = ToIsometry3d(entry.second.transform.T());
  }
  return true;
}

std::vector<TransformCandidateWithMeta> CollectSharedStereoExtrinsicCandidates(
    const StereoMeasurementDataset& dataset,
    const StereoCameraFixedCalibration& cam0,
    const StereoCameraFixedCalibration& cam1,
    const StereoExtrinsicSolverOptions& options,
    StereoPairOnlyBaInitSummary* summary) {
  std::vector<TransformCandidateWithMeta> candidates;
  StereoSceneState local_scene;
  local_scene.cam0 = cam0;
  local_scene.cam1 = cam1;
  for (int pair_index : dataset.training_pair_indices) {
    const int good_shared_board_count =
        SharedBoardQualityHardGateEnabled(options)
            ? CountQualitySharedBoards(
                  dataset, local_scene, options, pair_index)
            : 0;
    const bool has_enough_shared_boards =
        SharedBoardQualityHardGateEnabled(options)
            ? good_shared_board_count >=
                  std::max(options.min_shared_boards_for_extrinsic_candidate,
                           options.shared_board_quality_min_good_shared_boards)
            : PairHasSharedBoardCandidate(
                  dataset, pair_index,
                  options.min_shared_boards_for_extrinsic_candidate);
    if (!has_enough_shared_boards) {
      continue;
    }
    for (int board_id : SortedBoardIds(dataset, pair_index)) {
      const StereoSharedBoardQuality quality =
          EvaluateSharedBoardQuality(dataset, local_scene, options, pair_index,
                                     board_id);
      StereoPairInitCandidateRow row;
      row.pair_index = pair_index;
      row.board_id = board_id;
      row.raw_candidate = quality.have_cam0 && quality.have_cam1;
      row.cam0_outer_rmse = quality.cam0_outer_rmse;
      row.cam1_outer_rmse = quality.cam1_outer_rmse;
      row.shared_outer_point_count =
          quality.cam0_outer_point_count + quality.cam1_outer_point_count;
      if (!quality.have_cam0 || !quality.have_cam1) {
        row.reject_reason = "missing_cam0_or_cam1_outer_pose";
        if (summary != nullptr) {
          ++summary->failed_pair_board_count;
          summary->candidates.push_back(row);
        }
        continue;
      }
      if (SharedBoardQualityHardGateEnabled(options) && !quality.pass) {
        row.reject_reason = "outer_pose_rmse_above_guard";
        if (summary != nullptr) {
          ++summary->failed_pair_board_count;
          summary->candidates.push_back(row);
        }
        continue;
      }
      TransformCandidateWithMeta candidate;
      candidate.pair_index = pair_index;
      candidate.board_id = board_id;
      candidate.transform =
          quality.T_cam1_board * quality.T_cam0_board.inverse();
      candidate.score =
          0.5 * (quality.cam0_outer_rmse + quality.cam1_outer_rmse);
      row.T_cam1_cam0_candidate = ToMatrix4d(candidate.transform);
      row.candidate_baseline_length = candidate.transform.translation().norm();
      candidates.push_back(candidate);
      if (summary != nullptr) {
        summary->candidates.push_back(row);
      }
    }
  }
  if (summary != nullptr) {
    summary->raw_candidate_count = static_cast<int>(candidates.size());
  }
  return candidates;
}

StereoPairOnlyBaInitSummary RunPairOnlyStereoBaInitialization(
    const StereoMeasurementDataset& dataset,
    const StereoExtrinsicSolverOptions& options,
    StereoSceneState* scene_state) {
  StereoPairOnlyBaInitSummary summary;
  summary.enabled = options.enable_pair_only_stereo_ba_init;
  if (scene_state == nullptr) {
    summary.failure_reason = "missing_scene_state";
    return summary;
  }
  const StereoCameraFixedCalibration initial_cam0 = scene_state->cam0;
  const StereoCameraFixedCalibration initial_cam1 = scene_state->cam1;
  const Eigen::Isometry3d medoid_baseline =
      ToIsometry3d(scene_state->T_cam1_cam0);
  summary.medoid_baseline_length = medoid_baseline.translation().norm();

  std::vector<TransformCandidateWithMeta> raw_candidates =
      CollectSharedStereoExtrinsicCandidates(dataset, scene_state->cam0,
                                             scene_state->cam1, options,
                                             &summary);
  if (raw_candidates.empty()) {
    summary.failure_reason = "no_valid_pair_init_candidates";
    return summary;
  }

  std::vector<TransformCandidate> transform_candidates;
  transform_candidates.reserve(raw_candidates.size());
  for (const TransformCandidateWithMeta& candidate : raw_candidates) {
    TransformCandidate transform_candidate;
    transform_candidate.transform = candidate.transform;
    transform_candidate.weight = 1.0 / std::max(1e-6, candidate.score);
    transform_candidates.push_back(transform_candidate);
  }
  int rejected_for_consistency = 0;
  const std::vector<TransformCandidate> filtered =
      FilterConsistentCandidates(
          transform_candidates,
          DegreesToRadians(options.candidate_consistency_max_rotation_deg),
          options.candidate_consistency_max_translation_m,
          &rejected_for_consistency);
  summary.consistency_filtered_candidate_count = static_cast<int>(filtered.size());
  summary.consistency_rejected_candidate_count = rejected_for_consistency;
  if (filtered.empty()) {
    summary.failure_reason = "all_pair_init_candidates_rejected_by_consistency";
    return summary;
  }

  const Eigen::Isometry3d refined_consensus_baseline = AverageTransforms(filtered);

  for (StereoPairInitCandidateRow& row : summary.candidates) {
    if (!row.raw_candidate) {
      continue;
    }
    const Eigen::Isometry3d row_transform =
        ToIsometry3d(row.T_cam1_cam0_candidate);
    row.consistency_accepted =
        CandidateConsistent(
            row_transform, refined_consensus_baseline,
            DegreesToRadians(options.candidate_consistency_max_rotation_deg),
            options.candidate_consistency_max_translation_m);
    if (!row.consistency_accepted && row.reject_reason.empty()) {
      row.reject_reason = "rejected_by_consistency_guard";
    }
  }

  std::vector<TransformCandidateWithMeta> accepted_candidates;
  accepted_candidates.reserve(raw_candidates.size());
  for (const TransformCandidateWithMeta& candidate : raw_candidates) {
    if (CandidateConsistent(
            candidate.transform, refined_consensus_baseline,
            DegreesToRadians(options.candidate_consistency_max_rotation_deg),
            options.candidate_consistency_max_translation_m)) {
      accepted_candidates.push_back(candidate);
    }
  }
  if (accepted_candidates.empty()) {
    summary.failure_reason = "no_consistent_pair_init_candidates";
    return summary;
  }

  Eigen::Isometry3d refined_baseline = refined_consensus_baseline;
  StereoCameraFixedCalibration refined_cam0 = scene_state->cam0;
  StereoCameraFixedCalibration refined_cam1 = scene_state->cam1;
  std::map<PairBoardKey, Eigen::Isometry3d> optimized_local_poses;
  const std::string cam0_family = scene_state->cam0.camera_model_family;
  const std::string cam1_family = scene_state->cam1.camera_model_family;
  bool local_ba_success = false;
  if (cam0_family == "ds-none" && cam1_family == "ds-none") {
    local_ba_success = RunPairOnlyStereoBaTyped<DsGeometry, DsGeometry>(
        dataset, options, scene_state->cam0, scene_state->cam1, accepted_candidates,
        medoid_baseline, &refined_baseline, &optimized_local_poses, &summary,
        &refined_cam0, &refined_cam1);
  } else if (cam0_family == "ds-none" && cam1_family == "eucm-none") {
    local_ba_success = RunPairOnlyStereoBaTyped<DsGeometry, EucmGeometry>(
        dataset, options, scene_state->cam0, scene_state->cam1, accepted_candidates,
        medoid_baseline, &refined_baseline, &optimized_local_poses, &summary,
        &refined_cam0, &refined_cam1);
  } else if (cam0_family == "eucm-none" && cam1_family == "ds-none") {
    local_ba_success = RunPairOnlyStereoBaTyped<EucmGeometry, DsGeometry>(
        dataset, options, scene_state->cam0, scene_state->cam1, accepted_candidates,
        medoid_baseline, &refined_baseline, &optimized_local_poses, &summary,
        &refined_cam0, &refined_cam1);
  } else if (cam0_family == "eucm-none" && cam1_family == "eucm-none") {
    local_ba_success = RunPairOnlyStereoBaTyped<EucmGeometry, EucmGeometry>(
        dataset, options, scene_state->cam0, scene_state->cam1, accepted_candidates,
        medoid_baseline, &refined_baseline, &optimized_local_poses, &summary,
        &refined_cam0, &refined_cam1);
  } else if (cam0_family == "pinhole-equi" && cam1_family == "pinhole-equi") {
    local_ba_success =
        RunPairOnlyStereoBaTyped<PinholeEquiGeometry, PinholeEquiGeometry>(
            dataset, options, scene_state->cam0, scene_state->cam1,
            accepted_candidates, medoid_baseline, &refined_baseline,
            &optimized_local_poses, &summary, &refined_cam0, &refined_cam1);
  } else if (cam0_family == "ds-none" && cam1_family == "pinhole-equi") {
    local_ba_success =
        RunPairOnlyStereoBaTyped<DsGeometry, PinholeEquiGeometry>(
            dataset, options, scene_state->cam0, scene_state->cam1,
            accepted_candidates, medoid_baseline, &refined_baseline,
            &optimized_local_poses, &summary, &refined_cam0, &refined_cam1);
  } else if (cam0_family == "pinhole-equi" && cam1_family == "ds-none") {
    local_ba_success =
        RunPairOnlyStereoBaTyped<PinholeEquiGeometry, DsGeometry>(
            dataset, options, scene_state->cam0, scene_state->cam1,
            accepted_candidates, medoid_baseline, &refined_baseline,
            &optimized_local_poses, &summary, &refined_cam0, &refined_cam1);
  } else if (cam0_family == "eucm-none" && cam1_family == "pinhole-equi") {
    local_ba_success =
        RunPairOnlyStereoBaTyped<EucmGeometry, PinholeEquiGeometry>(
            dataset, options, scene_state->cam0, scene_state->cam1,
            accepted_candidates, medoid_baseline, &refined_baseline,
            &optimized_local_poses, &summary, &refined_cam0, &refined_cam1);
  } else if (cam0_family == "pinhole-equi" && cam1_family == "eucm-none") {
    local_ba_success =
        RunPairOnlyStereoBaTyped<PinholeEquiGeometry, EucmGeometry>(
            dataset, options, scene_state->cam0, scene_state->cam1,
            accepted_candidates, medoid_baseline, &refined_baseline,
            &optimized_local_poses, &summary, &refined_cam0, &refined_cam1);
  } else if (cam0_family == "omni-none" && cam1_family == "omni-none") {
    local_ba_success = RunPairOnlyStereoBaTyped<OmniGeometry, OmniGeometry>(
        dataset, options, scene_state->cam0, scene_state->cam1,
        accepted_candidates, medoid_baseline, &refined_baseline,
        &optimized_local_poses, &summary, &refined_cam0, &refined_cam1);
  } else if (cam0_family == "omni-radtan" &&
             cam1_family == "omni-radtan") {
    local_ba_success =
        RunPairOnlyStereoBaTyped<OmniRadtanGeometry, OmniRadtanGeometry>(
            dataset, options, scene_state->cam0, scene_state->cam1,
            accepted_candidates, medoid_baseline, &refined_baseline,
            &optimized_local_poses, &summary, &refined_cam0, &refined_cam1);
  }
  if (!local_ba_success) {
    summary.failure_reason = "pair_only_local_ba_failed";
    return summary;
  }

  if (summary.reached_max_iterations) {
    summary.warnings.push_back(
        "pair_only_optimizer_reached_max_iterations_seed_kept_after_health_checks");
  }

  scene_state->cam0 = refined_cam0;
  scene_state->cam1 = refined_cam1;

  summary.pair_ba_baseline_length = refined_baseline.translation().norm();
  summary.baseline_rotation_delta_deg =
      RotationDistanceRadians(medoid_baseline, refined_baseline) * 180.0 / M_PI;
  summary.baseline_translation_delta_m =
      (medoid_baseline.translation() - refined_baseline.translation()).norm();

  double before_squared_sum = 0.0;
  double after_squared_sum = 0.0;
  int residual_count = 0;
  for (const TransformCandidateWithMeta& candidate : accepted_candidates) {
    const PairBoardKey key(candidate.pair_index, candidate.board_id);
    const auto local_pose_it = optimized_local_poses.find(key);
    if (local_pose_it == optimized_local_poses.end()) {
      continue;
    }
    Eigen::Isometry3d T_cam0_board = Eigen::Isometry3d::Identity();
    double cam0_rmse = 0.0;
    if (!EstimateCameraBoardPoseWithRmse(dataset, initial_cam0,
                                         candidate.pair_index, 0,
                                         candidate.board_id, &T_cam0_board,
                                         &cam0_rmse)) {
      continue;
    }
    StereoPairInitResidualRow row;
    row.pair_index = candidate.pair_index;
    row.board_id = candidate.board_id;
    row.shared_point_count =
        CountStereoObservations(dataset, candidate.pair_index, 0,
                                candidate.board_id, JointPointType::Outer) +
        CountStereoObservations(dataset, candidate.pair_index, 1,
                                candidate.board_id, JointPointType::Outer) +
        CountStereoObservations(dataset, candidate.pair_index, 0,
                                candidate.board_id, JointPointType::Internal) +
        CountStereoObservations(dataset, candidate.pair_index, 1,
                                candidate.board_id, JointPointType::Internal);
    row.before_rmse = EvaluateSharedBoardCandidateRmse(
        dataset, initial_cam0, initial_cam1, candidate.pair_index,
        candidate.board_id, medoid_baseline, T_cam0_board);
    row.after_rmse = EvaluateSharedBoardCandidateRmse(
        dataset, scene_state->cam0, scene_state->cam1, candidate.pair_index,
        candidate.board_id, refined_baseline, local_pose_it->second);
    if (std::isfinite(row.before_rmse) && std::isfinite(row.after_rmse) &&
        row.shared_point_count > 0) {
      before_squared_sum += row.before_rmse * row.before_rmse *
                            static_cast<double>(row.shared_point_count);
      after_squared_sum += row.after_rmse * row.after_rmse *
                           static_cast<double>(row.shared_point_count);
      residual_count += row.shared_point_count;
    }
    summary.residual_rows.push_back(row);
  }
  if (residual_count > 0) {
    summary.before_shared_rmse =
        std::sqrt(before_squared_sum / static_cast<double>(residual_count));
    summary.after_shared_rmse =
        std::sqrt(after_squared_sum / static_cast<double>(residual_count));
  }
  const bool has_rmse_comparison =
      std::isfinite(summary.before_shared_rmse) &&
      std::isfinite(summary.after_shared_rmse);
  const bool refined_is_better =
      !has_rmse_comparison || summary.after_shared_rmse <= summary.before_shared_rmse;
  summary.used_refined_baseline = refined_is_better;
  if (refined_is_better) {
    scene_state->T_cam1_cam0 = ToMatrix4d(refined_baseline);
  } else {
    scene_state->cam0 = initial_cam0;
    scene_state->cam1 = initial_cam1;
  }
  summary.success = true;
  return summary;
}

StereoPairSelectionSummary SelectStereoPairs(
    const StereoMeasurementDataset& dataset,
    const StereoSceneState& scene_state,
    const StereoInitializationDiagnostics& initialization,
    const StereoExtrinsicSolverOptions& options) {
  StereoPairSelectionSummary summary;
  summary.success = true;
  summary.mode = options.view_selection_mode;
  summary.requested_pair_count = options.selected_pair_count;

  std::set<int> reachable_pairs(initialization.reachable_training_pair_indices.begin(),
                                initialization.reachable_training_pair_indices.end());
  std::set<int> initialized_pairs;
  for (const auto& entry : scene_state.T_cam0_world_by_pair) {
    initialized_pairs.insert(entry.first);
  }

  std::set<int> eligible_pairs;
  std::set<int> desired_board_ids;
  for (int pair_index : dataset.training_pair_indices) {
    const auto it = dataset.training_pair_board_ids.find(pair_index);
    if (it != dataset.training_pair_board_ids.end()) {
      desired_board_ids.insert(it->second.begin(), it->second.end());
    }
  }
  for (int pair_index : dataset.training_pair_indices) {
    StereoPairSelectionRow row;
    row.pair_index = pair_index;
    row.reachable = reachable_pairs.count(pair_index) > 0;
    row.initialized = initialized_pairs.count(pair_index) > 0;
    row.eligible =
        row.reachable &&
        row.initialized &&
        scene_state.excluded_training_pair_indices.count(pair_index) == 0;
    const auto board_ids_it = dataset.training_pair_board_ids.find(pair_index);
    if (board_ids_it != dataset.training_pair_board_ids.end()) {
      row.covered_board_ids.assign(board_ids_it->second.begin(),
                                   board_ids_it->second.end());
      row.covered_board_count =
          static_cast<int>(row.covered_board_ids.size());
    }
    row.missing_board_coverage_count =
        static_cast<int>(desired_board_ids.size()) - row.covered_board_count;
    if (!row.eligible) {
      if (!row.reachable) {
        row.rejection_reason = "graph_unreachable_from_gauge";
      } else if (scene_state.excluded_training_pair_indices.count(pair_index) > 0) {
        row.rejection_reason = "excluded_training_pair";
      } else {
        row.rejection_reason = "not_initialized";
      }
    } else {
      eligible_pairs.insert(pair_index);
      const StereoViewSelectionScore score =
          ComputePairSelectionScore(dataset, scene_state, options, pair_index);
      row.shared_board_count = score.shared_board_count;
      row.shared_outer_point_count = score.shared_outer_point_count;
      row.pose_fit_rmse = score.pose_fit_rmse;
      row.single_camera_only_board_count = score.single_camera_only_board_count;
      row.score_shared_board_count = score.shared_board_count;
      row.score_shared_outer_point_count = score.shared_outer_point_count;
      row.score_pose_fit_rmse = score.pose_fit_rmse;
      row.score_single_camera_only_board_count =
          score.single_camera_only_board_count;
    }
    summary.rows.push_back(row);
  }
  summary.reachable_pair_count = static_cast<int>(reachable_pairs.size());
  summary.initialized_pair_count = static_cast<int>(initialized_pairs.size());
  summary.eligible_pair_count = static_cast<int>(eligible_pairs.size());

  if (options.view_selection_mode == StereoViewSelectionMode::Off) {
    for (StereoPairSelectionRow& row : summary.rows) {
      if (row.eligible) {
        row.selected = true;
        summary.selected_pair_indices.insert(row.pair_index);
      }
    }
  } else {
    std::vector<StereoPairSelectionRow*> ranked_rows;
    ranked_rows.reserve(summary.rows.size());
    for (StereoPairSelectionRow& row : summary.rows) {
      if (row.eligible) {
        ranked_rows.push_back(&row);
      }
    }
    std::sort(ranked_rows.begin(), ranked_rows.end(),
              [](const StereoPairSelectionRow* lhs,
                 const StereoPairSelectionRow* rhs) {
                return IsSelectionRowBetter(*lhs, *rhs);
              });
    const int requested_count =
        options.selected_pair_count > 0 ? options.selected_pair_count
                                        : static_cast<int>(ranked_rows.size());
    for (StereoPairSelectionRow* row : ranked_rows) {
      if (static_cast<int>(summary.selected_pair_indices.size()) >= requested_count) {
        break;
      }
      row->selected = true;
      summary.selected_pair_indices.insert(row->pair_index);
    }

    const auto update_covered_boards = [&]() {
      summary.covered_board_ids.clear();
      for (int pair_index : summary.selected_pair_indices) {
        const auto it = dataset.training_pair_board_ids.find(pair_index);
        if (it != dataset.training_pair_board_ids.end()) {
          summary.covered_board_ids.insert(it->second.begin(), it->second.end());
        }
      }
    };
    update_covered_boards();
    if (!desired_board_ids.empty()) {
      for (StereoPairSelectionRow* row : ranked_rows) {
        if (summary.covered_board_ids.size() >= desired_board_ids.size()) {
          break;
        }
        if (summary.selected_pair_indices.count(row->pair_index) > 0) {
          continue;
        }
        const auto boards_it = dataset.training_pair_board_ids.find(row->pair_index);
        bool covers_missing_board = false;
        if (boards_it != dataset.training_pair_board_ids.end()) {
          for (int board_id : boards_it->second) {
            if (desired_board_ids.count(board_id) > 0 &&
                summary.covered_board_ids.count(board_id) == 0) {
              covers_missing_board = true;
              break;
            }
          }
        }
        if (!covers_missing_board) {
          continue;
        }
        row->selected = true;
        summary.selected_pair_indices.insert(row->pair_index);
        update_covered_boards();
      }
    }
  }

  for (StereoPairSelectionRow& row : summary.rows) {
    if (!row.eligible && row.rejection_reason.empty()) {
      row.rejection_reason = "not_eligible";
    }
    if (row.selected) {
      row.rejection_reason.clear();
      if (row.shared_board_count > 0) {
        ++summary.selected_shared_board_pair_count;
      } else {
        ++summary.selected_single_camera_only_pair_count;
      }
    }
  }
  summary.selected_pair_count = static_cast<int>(summary.selected_pair_indices.size());
  summary.selected_covered_board_count =
      static_cast<int>(summary.covered_board_ids.size());
  summary.selected_pair_board_keys.clear();
  for (int pair_index : summary.selected_pair_indices) {
    const auto boards_it = dataset.training_pair_board_ids.find(pair_index);
    if (boards_it == dataset.training_pair_board_ids.end()) {
      continue;
    }
    for (int board_id : boards_it->second) {
      summary.selected_pair_board_keys.insert(PairBoardKey(pair_index, board_id));
    }
  }
  std::vector<double> selected_pose_fit_values;
  for (const StereoPairSelectionRow& row : summary.rows) {
    if (!row.selected || !std::isfinite(row.pose_fit_rmse)) {
      continue;
    }
    selected_pose_fit_values.push_back(row.pose_fit_rmse);
  }
  if (!selected_pose_fit_values.empty()) {
    summary.selected_pose_fit_rmse_min =
        *std::min_element(selected_pose_fit_values.begin(),
                          selected_pose_fit_values.end());
    summary.selected_pose_fit_rmse_max =
        *std::max_element(selected_pose_fit_values.begin(),
                          selected_pose_fit_values.end());
    summary.selected_pose_fit_rmse_median =
        ComputeMedian(selected_pose_fit_values);
  }
  return summary;
}

StereoPairSelectionSummary SelectStereoLocalBoardPosePairs(
    const StereoMeasurementDataset& dataset,
    const StereoExtrinsicSolverOptions& options,
    StereoSceneState* scene_state) {
  if (scene_state == nullptr) {
    throw std::runtime_error(
        "SelectStereoLocalBoardPosePairs requires scene_state.");
  }
  StereoPairSelectionSummary summary;
  summary.success = true;
  summary.mode = StereoViewSelectionMode::Off;
  summary.requested_pair_count = options.selected_pair_count;

  std::set<int> desired_board_ids;
  for (int pair_index : dataset.training_pair_indices) {
    const auto boards_it = dataset.training_pair_board_ids.find(pair_index);
    if (boards_it != dataset.training_pair_board_ids.end()) {
      desired_board_ids.insert(boards_it->second.begin(), boards_it->second.end());
    }
  }

  for (int pair_index : dataset.training_pair_indices) {
    StereoPairSelectionRow row;
    row.pair_index = pair_index;
    row.reachable = true;
    const auto boards_it = dataset.training_pair_board_ids.find(pair_index);
    if (boards_it != dataset.training_pair_board_ids.end()) {
      row.covered_board_ids.assign(boards_it->second.begin(),
                                   boards_it->second.end());
      row.covered_board_count =
          static_cast<int>(row.covered_board_ids.size());
    }
    row.missing_board_coverage_count =
        static_cast<int>(desired_board_ids.size()) - row.covered_board_count;
    if (row.covered_board_ids.size() != 1u) {
      row.rejection_reason = "local_board_pose_requires_single_board_pair";
      summary.rows.push_back(row);
      continue;
    }
    const int board_id = row.covered_board_ids.front();
    const bool shared_board =
        dataset.pair_shared_board_ids.count(pair_index) > 0 &&
        dataset.pair_shared_board_ids.at(pair_index).count(board_id) > 0;
    if (!shared_board) {
      row.rejection_reason = "local_board_pose_requires_stereo_shared_board";
      summary.rows.push_back(row);
      continue;
    }
    Eigen::Isometry3d T_cam0_board = Eigen::Isometry3d::Identity();
    Eigen::Isometry3d T_cam1_board = Eigen::Isometry3d::Identity();
    double cam0_rmse = std::numeric_limits<double>::infinity();
    double cam1_rmse = std::numeric_limits<double>::infinity();
    const bool cam0_pose_ok =
        EstimateCameraBoardPoseWithRmse(dataset, scene_state->cam0, pair_index,
                                        0, board_id, &T_cam0_board,
                                        &cam0_rmse);
    const bool cam1_pose_ok =
        EstimateCameraBoardPoseWithRmse(dataset, scene_state->cam1, pair_index,
                                        1, board_id, &T_cam1_board,
                                        &cam1_rmse);
    if (!cam0_pose_ok || !cam1_pose_ok) {
      row.rejection_reason =
          cam0_pose_ok ? "cam1_local_board_pose_failed"
                       : "cam0_local_board_pose_failed";
      summary.rows.push_back(row);
      continue;
    }

    row.initialized = true;
    row.eligible = true;
    row.selected = true;
    row.shared_board_count = 1;
    row.score_shared_board_count = 1;
    row.shared_outer_point_count =
        CountOuterObservations(dataset, pair_index, 0, board_id) +
        CountOuterObservations(dataset, pair_index, 1, board_id);
    row.score_shared_outer_point_count = row.shared_outer_point_count;
    row.pose_fit_rmse = 0.5 * (cam0_rmse + cam1_rmse);
    row.score_pose_fit_rmse = row.pose_fit_rmse;

    summary.selected_pair_indices.insert(pair_index);
    summary.selected_pair_board_keys.insert(PairBoardKey(pair_index, board_id));
    summary.covered_board_ids.insert(board_id);

    const auto board_pose_it = scene_state->T_world_board_by_id.find(board_id);
    if (board_pose_it != scene_state->T_world_board_by_id.end()) {
      const Eigen::Isometry3d T_world_board =
          ToIsometry3d(board_pose_it->second);
      scene_state->T_cam0_world_by_pair[pair_index] =
          ToMatrix4d(T_cam0_board * T_world_board.inverse());
    } else {
      scene_state->T_cam0_world_by_pair[pair_index] = ToMatrix4d(T_cam0_board);
    }
    summary.rows.push_back(row);
  }

  summary.reachable_pair_count =
      static_cast<int>(dataset.training_pair_indices.size());
  summary.initialized_pair_count =
      static_cast<int>(summary.selected_pair_indices.size());
  summary.eligible_pair_count =
      static_cast<int>(summary.selected_pair_indices.size());
  summary.selected_pair_count =
      static_cast<int>(summary.selected_pair_indices.size());
  summary.selected_shared_board_pair_count = summary.selected_pair_count;
  summary.selected_single_camera_only_pair_count = 0;
  summary.selected_covered_board_count =
      static_cast<int>(summary.covered_board_ids.size());

  std::vector<double> selected_pose_fit_values;
  for (const StereoPairSelectionRow& row : summary.rows) {
    if (row.selected && std::isfinite(row.pose_fit_rmse)) {
      selected_pose_fit_values.push_back(row.pose_fit_rmse);
    }
  }
  if (!selected_pose_fit_values.empty()) {
    summary.selected_pose_fit_rmse_min =
        *std::min_element(selected_pose_fit_values.begin(),
                          selected_pose_fit_values.end());
    summary.selected_pose_fit_rmse_max =
        *std::max_element(selected_pose_fit_values.begin(),
                          selected_pose_fit_values.end());
    summary.selected_pose_fit_rmse_median =
        ComputeMedian(selected_pose_fit_values);
  }
  summary.warnings.push_back(
      "board_masking_local_board_pose_selection: selected all stereo-shared "
      "single-board pseudo-pairs with successful cam0/cam1 local pose fits.");
  return summary;
}

template <typename GeometryT0, typename GeometryT1>
bool RunGlobalSparseBaTyped(
    const StereoMeasurementDataset& dataset,
    const StereoExtrinsicSolverOptions& options,
    const StereoPairSelectionSummary& selection_summary,
    StereoGlobalSparseBaSummary* ba_summary,
    StereoSceneState* scene_state) {
  if (ba_summary == nullptr || scene_state == nullptr) {
    throw std::runtime_error("RunGlobalSparseBaTyped requires valid outputs.");
  }
  boost::shared_ptr<GeometryT0> cam0_geometry =
      MakeTypedStereoGeometry<GeometryT0>(scene_state->cam0);
  boost::shared_ptr<GeometryT1> cam1_geometry =
      MakeTypedStereoGeometry<GeometryT1>(scene_state->cam1);
  const bool use_spherical_residual =
      options.final_ba_residual_mode ==
          StereoFinalBaResidualMode::SphericalChordal ||
      options.final_ba_residual_mode ==
          StereoFinalBaResidualMode::SphericalTangent ||
      options.final_ba_residual_mode ==
          StereoFinalBaResidualMode::HybridPixelSpherical;
  const bool use_chordal_spherical_residual =
      options.final_ba_residual_mode ==
      StereoFinalBaResidualMode::SphericalChordal;
  const bool use_tangent_spherical_residual =
      options.final_ba_residual_mode ==
          StereoFinalBaResidualMode::SphericalTangent ||
      options.final_ba_residual_mode ==
          StereoFinalBaResidualMode::HybridPixelSpherical;
  const bool use_pixel_residual =
      options.final_ba_residual_mode == StereoFinalBaResidualMode::Pixel ||
      options.final_ba_residual_mode ==
          StereoFinalBaResidualMode::HybridPixelSpherical;
  if (use_spherical_residual &&
      (scene_state->cam0.camera_model_family != "ds-none" ||
       scene_state->cam1.camera_model_family != "ds-none")) {
    ba_summary->failure_reason =
        "spherical final BA residuals currently require ds-none cam0/cam1.";
    return false;
  }
  DoubleSphereCameraModel cam0_residual_camera;
  DoubleSphereCameraModel cam1_residual_camera;
  IntermediateCameraConfig cam0_residual_config;
  IntermediateCameraConfig cam1_residual_config;
  if (use_spherical_residual) {
    cam0_residual_config = MakeCameraConfig(scene_state->cam0);
    cam1_residual_config = MakeCameraConfig(scene_state->cam1);
    cam0_residual_camera =
        DoubleSphereCameraModel::FromConfig(cam0_residual_config);
    cam1_residual_camera =
        DoubleSphereCameraModel::FromConfig(cam1_residual_config);
  }
  BearingCovarianceOptions bearing_covariance_options;
  bearing_covariance_options.use_pixel_uncertainty =
      options.spherical_uncertainty_mode ==
          StereoSphericalUncertaintyMode::Pixel ||
      options.spherical_uncertainty_mode ==
          StereoSphericalUncertaintyMode::PixelModel;
  bearing_covariance_options.use_model_uncertainty =
      options.spherical_uncertainty_mode ==
          StereoSphericalUncertaintyMode::Model ||
      options.spherical_uncertainty_mode ==
          StereoSphericalUncertaintyMode::PixelModel;
  bearing_covariance_options.pixel_sigma_px =
      options.spherical_pixel_sigma_px;
  for (int index = 0; index < 6; ++index) {
    bearing_covariance_options.model_sigma(index) =
        options.spherical_model_sigma[index];
  }
  bearing_covariance_options.covariance_damping =
      options.spherical_covariance_damping;
  bearing_covariance_options.min_sigma_rad = options.spherical_min_sigma_rad;
  bearing_covariance_options.max_whitening_weight =
      options.spherical_max_whitening_weight;
  const bool use_uncertainty_whitening =
      use_tangent_spherical_residual &&
      options.spherical_uncertainty_mode !=
          StereoSphericalUncertaintyMode::None;
  BearingWhiteningStatsAccumulator whitening_stats;

  CameraDv<GeometryT0> cam0_dv(cam0_geometry);
  CameraDv<GeometryT1> cam1_dv(cam1_geometry);
  const bool optimize_intrinsics =
      options.final_ba_optimize_intrinsics &&
      !(use_spherical_residual && options.fixed_intrinsics_for_spherical);
  const bool dynamic_observed_ray =
      use_spherical_residual && optimize_intrinsics;
  cam0_dv.setActive(optimize_intrinsics, optimize_intrinsics, false);
  cam1_dv.setActive(optimize_intrinsics, optimize_intrinsics, false);

  boost::shared_ptr<aslam::backend::OptimizationProblem> problem(
      new aslam::backend::OptimizationProblem);
  problem->addDesignVariable(cam0_dv.projectionDesignVariable());
  problem->addDesignVariable(cam0_dv.distortionDesignVariable());
  problem->addDesignVariable(cam0_dv.shutterDesignVariable());
  problem->addDesignVariable(cam1_dv.projectionDesignVariable());
  problem->addDesignVariable(cam1_dv.distortionDesignVariable());
  problem->addDesignVariable(cam1_dv.shutterDesignVariable());

  const bool rig_centric =
      options.rig_param_mode == StereoRigParamMode::RigCentricSymmetric;
  const bool local_board_pose_ba =
      options.board_masking_use_local_board_pose_ba;
  if (local_board_pose_ba && rig_centric) {
    ba_summary->failure_reason =
        "board masking local-board-pose BA is only supported in cam0_reference mode.";
    return false;
  }
  const Eigen::Isometry3d initial_T_cam1_cam0 =
      ToIsometry3d(scene_state->T_cam1_cam0);

  boost::shared_ptr<aslam::backend::MappedRotationQuaternion> baseline_rotation_dv;
  boost::shared_ptr<aslam::backend::MappedEuclideanPoint> baseline_translation_dv;
  sm::kinematics::Transformation baseline_transform(scene_state->T_cam1_cam0);
  aslam::backend::TransformationExpression T_cam1_cam0_expr =
      aslam::backend::transformationToExpression(
          baseline_transform, baseline_rotation_dv, baseline_translation_dv);
  baseline_rotation_dv->setActive(!rig_centric);
  baseline_translation_dv->setActive(!rig_centric);
  if (!rig_centric) {
    problem->addDesignVariable(baseline_rotation_dv);
    problem->addDesignVariable(baseline_translation_dv);
  }

  boost::shared_ptr<aslam::backend::MappedRotationQuaternion> cam0_rig_rotation_dv;
  boost::shared_ptr<aslam::backend::MappedEuclideanPoint> cam0_rig_translation_dv;
  boost::shared_ptr<aslam::backend::MappedRotationQuaternion> cam1_rig_rotation_dv;
  boost::shared_ptr<aslam::backend::MappedEuclideanPoint> cam1_rig_translation_dv;
  sm::kinematics::Transformation T_cam0_rig_transform(Eigen::Matrix4d::Identity());
  sm::kinematics::Transformation T_cam1_rig_transform(scene_state->T_cam1_cam0);
  aslam::backend::TransformationExpression T_cam0_rig_expr(
      Eigen::Matrix4d::Identity());
  aslam::backend::TransformationExpression T_cam1_rig_expr(
      Eigen::Matrix4d::Identity());
  if (rig_centric) {
    // Convention: pair_variables store T_cam0_world in the legacy path. In the
    // rig-centric path the same per-pair variable is initialized as T_rig_world.
    // This fallback initialization is projection-equivalent to the legacy model:
    // T_rig_world=T_cam0_world, T_cam0_rig=I, T_cam1_rig=T_cam1_cam0.
    T_cam0_rig_expr = aslam::backend::transformationToExpression(
        T_cam0_rig_transform, cam0_rig_rotation_dv, cam0_rig_translation_dv);
    T_cam1_rig_expr = aslam::backend::transformationToExpression(
        T_cam1_rig_transform, cam1_rig_rotation_dv, cam1_rig_translation_dv);
    cam0_rig_rotation_dv->setActive(options.final_ba_optimize_stereo_extrinsic);
    cam0_rig_translation_dv->setActive(options.final_ba_optimize_stereo_extrinsic);
    cam1_rig_rotation_dv->setActive(options.final_ba_optimize_stereo_extrinsic);
    cam1_rig_translation_dv->setActive(options.final_ba_optimize_stereo_extrinsic);
    problem->addDesignVariable(cam0_rig_rotation_dv);
    problem->addDesignVariable(cam0_rig_translation_dv);
    problem->addDesignVariable(cam1_rig_rotation_dv);
    problem->addDesignVariable(cam1_rig_translation_dv);
    const double rig_t_weight =
        std::max(0.0, options.rig_camera_prior_translation_weight);
    const double rig_r_weight =
        std::max(0.0, options.rig_camera_prior_rotation_weight);
    if (rig_t_weight > 0.0 || rig_r_weight > 0.0) {
      problem->addErrorTerm(boost::shared_ptr<aslam::backend::ErrorTerm>(
          new aslam::backend::ErrorTermTransformation(
              T_cam0_rig_expr, T_cam0_rig_transform, rig_r_weight,
              rig_t_weight)));
      problem->addErrorTerm(boost::shared_ptr<aslam::backend::ErrorTerm>(
          new aslam::backend::ErrorTermTransformation(
              T_cam1_rig_expr, T_cam1_rig_transform, rig_r_weight,
              rig_t_weight)));
    }
    const double stereo_prior_weight =
        std::max(0.0, options.rig_stereo_relative_prior_weight);
    if (stereo_prior_weight > 0.0) {
      const aslam::backend::TransformationExpression T_cam1_cam0_from_rig =
          T_cam1_rig_expr * T_cam0_rig_expr.inverse();
      problem->addErrorTerm(boost::shared_ptr<aslam::backend::ErrorTerm>(
          new aslam::backend::ErrorTermTransformation(
              T_cam1_cam0_from_rig, baseline_transform, stereo_prior_weight,
              stereo_prior_weight)));
    }
  }

  std::map<int, StereoPoseVariableState> pair_variables;
  std::map<int, StereoPoseVariableState> board_variables;
  std::map<int, int> local_board_id_by_pair;
  std::set<int> active_board_ids;
  for (int pair_index : selection_summary.selected_pair_indices) {
    std::set<int> selected_boards_for_pair;
    const auto boards_it = dataset.training_pair_board_ids.find(pair_index);
    if (boards_it != dataset.training_pair_board_ids.end()) {
      for (int board_id : boards_it->second) {
        if (PairBoardSelected(selection_summary, pair_index, board_id)) {
          selected_boards_for_pair.insert(board_id);
          active_board_ids.insert(board_id);
        }
      }
    }
    Eigen::Matrix4d initial_pair_transform = Eigen::Matrix4d::Identity();
    if (local_board_pose_ba) {
      if (selected_boards_for_pair.size() != 1u) {
        std::ostringstream reason;
        reason << "local-board-pose BA requires exactly one selected board per "
               << "pair; pair=" << pair_index
               << " selected_board_count="
               << selected_boards_for_pair.size();
        ba_summary->failure_reason = reason.str();
        return false;
      }
      const int local_board_id = *selected_boards_for_pair.begin();
      Eigen::Isometry3d T_cam0_board = Eigen::Isometry3d::Identity();
      double local_pose_rmse = 0.0;
      if (!EstimateCameraBoardPoseWithRmse(dataset, scene_state->cam0,
                                           pair_index, 0, local_board_id,
                                           &T_cam0_board, &local_pose_rmse)) {
        const auto pair_pose_it =
            scene_state->T_cam0_world_by_pair.find(pair_index);
        const auto board_pose_it =
            scene_state->T_world_board_by_id.find(local_board_id);
        if (pair_pose_it == scene_state->T_cam0_world_by_pair.end() ||
            board_pose_it == scene_state->T_world_board_by_id.end()) {
          continue;
        }
        T_cam0_board = ToIsometry3d(pair_pose_it->second) *
                       ToIsometry3d(board_pose_it->second);
      }
      initial_pair_transform = ToMatrix4d(T_cam0_board);
      local_board_id_by_pair[pair_index] = local_board_id;
    } else {
      const auto pair_pose_it =
          scene_state->T_cam0_world_by_pair.find(pair_index);
      if (pair_pose_it == scene_state->T_cam0_world_by_pair.end()) {
        continue;
      }
      initial_pair_transform = pair_pose_it->second;
    }
    if (selected_boards_for_pair.empty()) {
      continue;
    }
    StereoPoseVariableState& variable = pair_variables[pair_index];
    variable.transform = sm::kinematics::Transformation(initial_pair_transform);
    variable.expression = aslam::backend::transformationToExpression(
        variable.transform, variable.rotation_dv, variable.translation_dv);
    const bool pair_pose_active =
        local_board_pose_ba || rig_centric ? options.final_ba_optimize_pair_poses
                                           : options.final_ba_optimize_board_poses;
    variable.rotation_dv->setActive(pair_pose_active);
    variable.translation_dv->setActive(pair_pose_active);
    problem->addDesignVariable(variable.rotation_dv);
    problem->addDesignVariable(variable.translation_dv);
  }
  for (int board_id : active_board_ids) {
    if (local_board_pose_ba) {
      continue;
    }
    if (board_id == scene_state->gauge_fixed_board_id) {
      continue;
    }
    const auto board_pose_it = scene_state->T_world_board_by_id.find(board_id);
    if (board_pose_it == scene_state->T_world_board_by_id.end()) {
      continue;
    }
    StereoPoseVariableState& variable = board_variables[board_id];
    variable.transform = sm::kinematics::Transformation(board_pose_it->second);
    variable.expression = aslam::backend::transformationToExpression(
        variable.transform, variable.rotation_dv, variable.translation_dv);
    variable.rotation_dv->setActive(true);
    variable.translation_dv->setActive(true);
    problem->addDesignVariable(variable.rotation_dv);
    problem->addDesignVariable(variable.translation_dv);
  }

  const aslam::backend::TransformationExpression identity_transform(
      Eigen::Matrix4d::Identity());
  double shared_total_base_weight = 0.0;
  double cam0_only_total_base_weight = 0.0;
  double cam1_only_total_base_weight = 0.0;
  for (const StereoObservation& observation : dataset.observations) {
    if (!observation.used_in_solver ||
        selection_summary.selected_pair_indices.count(observation.pair_index) == 0) {
      continue;
    }
    if (!PairBoardSelected(selection_summary, observation.pair_index,
                           observation.board_id)) {
      continue;
    }
    const double base_weight = std::max(0.0, observation.weight);
    if (base_weight <= 0.0) {
      continue;
    }
    const bool is_shared_observation =
        dataset.pair_shared_board_ids.count(observation.pair_index) > 0 &&
        dataset.pair_shared_board_ids.at(observation.pair_index).count(
            observation.board_id) > 0;
    const bool is_cam0_only_observation =
        dataset.pair_cam0_only_board_ids.count(observation.pair_index) > 0 &&
        dataset.pair_cam0_only_board_ids.at(observation.pair_index).count(
            observation.board_id) > 0;
    const bool is_cam1_only_observation =
        dataset.pair_cam1_only_board_ids.count(observation.pair_index) > 0 &&
        dataset.pair_cam1_only_board_ids.at(observation.pair_index).count(
            observation.board_id) > 0;
    if (is_shared_observation) {
      shared_total_base_weight += base_weight;
    } else if (is_cam0_only_observation) {
      cam0_only_total_base_weight += base_weight;
    } else if (is_cam1_only_observation) {
      cam1_only_total_base_weight += base_weight;
    }
  }
  const double per_side_budget_limit =
      shared_total_base_weight * options.ba_single_camera_only_per_side_budget_ratio;
  const double adaptive_per_side_cap =
      shared_total_base_weight *
      options.ba_adaptive_single_camera_only_per_side_cap_ratio;
  double cam0_only_effective_scale = options.ba_single_camera_only_base_scale;
  double cam1_only_effective_scale = options.ba_single_camera_only_base_scale;
  bool cam0_only_budget_clamped = false;
  bool cam1_only_budget_clamped = false;
  if (options.ba_single_camera_only_weight_mode ==
      StereoSingleCameraOnlyWeightMode::PerSideBudgetCap) {
    if (cam0_only_total_base_weight > 0.0) {
      const double budget_scale =
          per_side_budget_limit / cam0_only_total_base_weight;
      cam0_only_effective_scale =
          std::min(options.ba_single_camera_only_base_scale, budget_scale);
      cam0_only_budget_clamped =
          cam0_only_effective_scale < options.ba_single_camera_only_base_scale;
    } else {
      cam0_only_effective_scale = 0.0;
    }
    if (cam1_only_total_base_weight > 0.0) {
      const double budget_scale =
          per_side_budget_limit / cam1_only_total_base_weight;
      cam1_only_effective_scale =
          std::min(options.ba_single_camera_only_base_scale, budget_scale);
      cam1_only_budget_clamped =
          cam1_only_effective_scale < options.ba_single_camera_only_base_scale;
    } else {
      cam1_only_effective_scale = 0.0;
    }
  } else if (options.ba_single_camera_only_weight_mode ==
             StereoSingleCameraOnlyWeightMode::AdaptiveIndependentSideCap) {
    if (cam0_only_total_base_weight > 0.0) {
      const double cap_scale = adaptive_per_side_cap / cam0_only_total_base_weight;
      cam0_only_effective_scale =
          std::min(options.ba_single_camera_only_base_scale, cap_scale);
      cam0_only_budget_clamped =
          cam0_only_effective_scale < options.ba_single_camera_only_base_scale;
    } else {
      cam0_only_effective_scale = 0.0;
    }
    if (cam1_only_total_base_weight > 0.0) {
      const double cap_scale = adaptive_per_side_cap / cam1_only_total_base_weight;
      cam1_only_effective_scale =
          std::min(options.ba_single_camera_only_base_scale, cap_scale);
      cam1_only_budget_clamped =
          cam1_only_effective_scale < options.ba_single_camera_only_base_scale;
    } else {
      cam1_only_effective_scale = 0.0;
    }
  }
  int reprojection_error_count = 0;
  int shared_observation_count = 0;
  int cam0_only_observation_count = 0;
  int cam1_only_observation_count = 0;
  double shared_observation_weight_sum = 0.0;
  double cam0_only_observation_weight_sum = 0.0;
  double cam1_only_observation_weight_sum = 0.0;
  std::set<PairBoardKey> ba_quality_accepted_shared_boards;
  if (SharedBoardQualityHardGateEnabled(options) &&
      options.shared_board_quality_filter_final_ba) {
    for (int pair_index : selection_summary.selected_pair_indices) {
      for (int board_id : SortedBoardIds(dataset, pair_index)) {
        if (EvaluateSharedBoardQuality(dataset, *scene_state, options,
                                       pair_index, board_id)
                .pass) {
          ba_quality_accepted_shared_boards.insert(
              PairBoardKey(pair_index, board_id));
        }
      }
    }
  }
  const StereoPairBoardConsistencySummary consistency_gate_summary =
      options.enable_pair_board_consistency_gate
          ? BuildPairBoardConsistencySummary(dataset, *scene_state, options)
          : StereoPairBoardConsistencySummary();
  auto compute_sqrt_information =
      [&](const IntermediateCameraConfig& camera_config,
          const Eigen::Vector2d& observed_image_xy,
          const AngularObservationGeometry& observation_geometry,
          Eigen::Matrix2d* sqrt_information) {
        if (sqrt_information == nullptr) {
          return false;
        }
        sqrt_information->setIdentity();
        if (!use_uncertainty_whitening) {
          return false;
        }
        BearingCovarianceResult covariance_result;
        if (!ComputeBearingTangentCovariance(
                camera_config, observed_image_xy, observation_geometry,
                bearing_covariance_options, &covariance_result)) {
          ++ba_summary->spherical_covariance_invalid_count;
          return false;
        }
        ++ba_summary->spherical_covariance_valid_count;
        if (covariance_result.damping_applied) {
          ++ba_summary->spherical_covariance_damped_count;
        }
        if (covariance_result.whitening_clamped) {
          ++ba_summary->spherical_whitening_clamped_count;
        }
        whitening_stats.Add(covariance_result);
        *sqrt_information = covariance_result.sqrt_information;
        return true;
      };
  for (const StereoObservation& observation : dataset.observations) {
    if (!observation.used_in_solver ||
        selection_summary.selected_pair_indices.count(observation.pair_index) == 0) {
      continue;
    }
    if (!PairBoardSelected(selection_summary, observation.pair_index,
                           observation.board_id)) {
      continue;
    }
    const auto pair_it = pair_variables.find(observation.pair_index);
    if (pair_it == pair_variables.end()) {
      continue;
    }
    aslam::backend::TransformationExpression board_expr = identity_transform;
    if (!local_board_pose_ba &&
        observation.board_id != scene_state->gauge_fixed_board_id) {
      const auto board_it = board_variables.find(observation.board_id);
      if (board_it == board_variables.end()) {
        continue;
      }
      board_expr = board_it->second.expression;
    }
    const aslam::backend::HomogeneousExpression point_board(
        observation.target_point_board);
    const aslam::backend::HomogeneousExpression point_world = board_expr * point_board;
    const aslam::backend::HomogeneousExpression point_rig_or_cam0 =
        pair_it->second.expression *
        (local_board_pose_ba ? point_board : point_world);
    const aslam::backend::HomogeneousExpression point_cam0 =
        rig_centric ? (T_cam0_rig_expr * point_rig_or_cam0)
                    : point_rig_or_cam0;
    const aslam::backend::HomogeneousExpression point_cam1 =
        rig_centric ? (T_cam1_rig_expr * point_rig_or_cam0)
                    : (T_cam1_cam0_expr * point_cam0);
    const bool is_shared_observation =
        dataset.pair_shared_board_ids.count(observation.pair_index) > 0 &&
        dataset.pair_shared_board_ids.at(observation.pair_index).count(
            observation.board_id) > 0;
    const bool is_cam0_only_observation =
        dataset.pair_cam0_only_board_ids.count(observation.pair_index) > 0 &&
        dataset.pair_cam0_only_board_ids.at(observation.pair_index).count(
            observation.board_id) > 0;
    const bool is_cam1_only_observation =
        dataset.pair_cam1_only_board_ids.count(observation.pair_index) > 0 &&
        dataset.pair_cam1_only_board_ids.at(observation.pair_index).count(
            observation.board_id) > 0;
    if (SharedBoardQualityHardGateEnabled(options) &&
        options.shared_board_quality_filter_final_ba &&
        is_shared_observation &&
        ba_quality_accepted_shared_boards.count(
            PairBoardKey(observation.pair_index, observation.board_id)) == 0) {
      continue;
    }
    if (options.enable_pair_board_consistency_gate &&
        consistency_gate_summary.gate_rejected_pair_boards.count(
            std::make_pair(observation.pair_index, observation.board_id)) > 0) {
      continue;
    }
    double weight_scale = options.ba_shared_observation_weight_scale;
    if (options.solver_mode == StereoSolverMode::SharedOnlyGlobalSparseBa &&
        !is_shared_observation) {
      continue;
    }
    if (is_cam0_only_observation) {
      if (options.ba_single_camera_only_weight_mode ==
              StereoSingleCameraOnlyWeightMode::PerSideBudgetCap ||
          options.ba_single_camera_only_weight_mode ==
              StereoSingleCameraOnlyWeightMode::AdaptiveIndependentSideCap) {
        weight_scale = cam0_only_effective_scale;
      } else {
        weight_scale = options.ba_single_camera_only_observation_weight_scale;
      }
    } else if (is_cam1_only_observation) {
      if (options.ba_single_camera_only_weight_mode ==
              StereoSingleCameraOnlyWeightMode::PerSideBudgetCap ||
          options.ba_single_camera_only_weight_mode ==
              StereoSingleCameraOnlyWeightMode::AdaptiveIndependentSideCap) {
        weight_scale = cam1_only_effective_scale;
      } else {
        weight_scale = options.ba_single_camera_only_observation_weight_scale;
      }
    }
    const double weight = std::max(0.0, observation.weight) * weight_scale;
    if (weight <= 0.0) {
      continue;
    }
    if (is_shared_observation) {
      ++shared_observation_count;
      shared_observation_weight_sum += weight;
    } else if (is_cam0_only_observation) {
      ++cam0_only_observation_count;
      cam0_only_observation_weight_sum += weight;
    } else if (is_cam1_only_observation) {
      ++cam1_only_observation_count;
      cam1_only_observation_weight_sum += weight;
    }
    if (observation.camera_index == 0) {
      if (use_pixel_residual) {
        boost::shared_ptr<aslam::backend::ErrorTerm> error(
            new StereoCameraReprojectionError<GeometryT0>(
                observation.observed_image_xy, weight, point_cam0, cam0_dv,
                100.0));
        problem->addErrorTerm(error);
        ++reprojection_error_count;
      }
      if (use_spherical_residual) {
        AngularObservationGeometry observation_geometry;
        if (!ComputeAngularObservationGeometry(
                cam0_residual_camera, observation.observed_image_xy,
                &observation_geometry)) {
          ++ba_summary->invalid_spherical_unprojection_count;
          continue;
        }
        double spherical_weight = std::max(0.0, options.spherical_weight) *
                                  std::sqrt(std::max(0.0, weight));
        if (options.spherical_polar_weighting) {
          const double polar_scale =
              ComputePolarContinuousAngularWeight(
                  observation_geometry.polar_angle_deg,
                  options.spherical_min_polar_deg, 5.0);
          const double max_scale = std::max(1.0, options.spherical_max_weight);
          spherical_weight *=
              1.0 + (max_scale - 1.0) * polar_scale;
        }
        if (spherical_weight <= 0.0) {
          continue;
        }
        boost::shared_ptr<aslam::backend::ErrorTerm> error;
        if (use_chordal_spherical_residual) {
          error.reset(new StereoCameraSphericalChordalError(
              observation_geometry.observed_ray, spherical_weight, point_cam0,
              0.35));
        } else if (use_tangent_spherical_residual) {
          Eigen::Matrix2d sqrt_information = Eigen::Matrix2d::Identity();
          if (compute_sqrt_information(cam0_residual_config,
                                       observation.observed_image_xy,
                                       observation_geometry,
                                       &sqrt_information)) {
            error.reset(new StereoCameraAngularReprojectionError<GeometryT0>(
                observation_geometry, observation.observed_image_xy,
                dynamic_observed_ray,
                scene_state->cam0, options.spherical_use_normalize_jacobian,
                spherical_weight, sqrt_information,
                point_cam0, cam0_dv, 0.35));
          } else {
            error.reset(new StereoCameraAngularReprojectionError<GeometryT0>(
                observation_geometry, observation.observed_image_xy,
                dynamic_observed_ray,
                scene_state->cam0, options.spherical_use_normalize_jacobian,
                spherical_weight, point_cam0, cam0_dv, 0.35));
          }
        }
        if (error) {
          problem->addErrorTerm(error);
          ++reprojection_error_count;
        }
      }
    } else {
      if (use_pixel_residual) {
        boost::shared_ptr<aslam::backend::ErrorTerm> error(
            new StereoCameraReprojectionError<GeometryT1>(
                observation.observed_image_xy, weight, point_cam1, cam1_dv,
                100.0));
        problem->addErrorTerm(error);
        ++reprojection_error_count;
      }
      if (use_spherical_residual) {
        AngularObservationGeometry observation_geometry;
        if (!ComputeAngularObservationGeometry(
                cam1_residual_camera, observation.observed_image_xy,
                &observation_geometry)) {
          ++ba_summary->invalid_spherical_unprojection_count;
          continue;
        }
        double spherical_weight = std::max(0.0, options.spherical_weight) *
                                  std::sqrt(std::max(0.0, weight));
        if (options.spherical_polar_weighting) {
          const double polar_scale =
              ComputePolarContinuousAngularWeight(
                  observation_geometry.polar_angle_deg,
                  options.spherical_min_polar_deg, 5.0);
          const double max_scale = std::max(1.0, options.spherical_max_weight);
          spherical_weight *=
              1.0 + (max_scale - 1.0) * polar_scale;
        }
        if (spherical_weight <= 0.0) {
          continue;
        }
        boost::shared_ptr<aslam::backend::ErrorTerm> error;
        if (use_chordal_spherical_residual) {
          error.reset(new StereoCameraSphericalChordalError(
              observation_geometry.observed_ray, spherical_weight, point_cam1,
              0.35));
        } else if (use_tangent_spherical_residual) {
          Eigen::Matrix2d sqrt_information = Eigen::Matrix2d::Identity();
          if (compute_sqrt_information(cam1_residual_config,
                                       observation.observed_image_xy,
                                       observation_geometry,
                                       &sqrt_information)) {
            error.reset(new StereoCameraAngularReprojectionError<GeometryT1>(
                observation_geometry, observation.observed_image_xy,
                dynamic_observed_ray,
                scene_state->cam1, options.spherical_use_normalize_jacobian,
                spherical_weight, sqrt_information,
                point_cam1, cam1_dv, 0.35));
          } else {
            error.reset(new StereoCameraAngularReprojectionError<GeometryT1>(
                observation_geometry, observation.observed_image_xy,
                dynamic_observed_ray,
                scene_state->cam1, options.spherical_use_normalize_jacobian,
                spherical_weight, point_cam1, cam1_dv, 0.35));
          }
        }
        if (error) {
          problem->addErrorTerm(error);
          ++reprojection_error_count;
        }
      }
    }
  }

  CoObsFactorBuildStats coobs_factor_stats;
  if (options.coobs_factor_ba_apply_stereo_factor ||
      options.coobs_factor_ba_apply_layout_factor) {
    const std::map<CoObsLocalPoseKey, CoObsLocalPoseMeasurement>
        coobs_measurements =
            BuildCoObsLocalPoseMeasurements(dataset, *scene_state,
                                            selection_summary, options,
                                            &coobs_factor_stats);
    const double stereo_weight =
        std::max(0.0, options.coobs_factor_ba_current_stereo_weight);
    if (options.coobs_factor_ba_apply_stereo_factor && stereo_weight > 0.0) {
      for (int pair_index : selection_summary.selected_pair_indices) {
        const auto boards_it = dataset.training_pair_board_ids.find(pair_index);
        if (boards_it == dataset.training_pair_board_ids.end()) {
          continue;
        }
        for (int board_id : boards_it->second) {
          if (!PairBoardSelected(selection_summary, pair_index, board_id)) {
            continue;
          }
          const CoObsLocalPoseMeasurement* cam0 =
              FindCoObsMeasurement(coobs_measurements, pair_index, 0,
                                   board_id);
          const CoObsLocalPoseMeasurement* cam1 =
              FindCoObsMeasurement(coobs_measurements, pair_index, 1,
                                   board_id);
          if (cam0 == nullptr || cam1 == nullptr) {
            continue;
          }
          const Eigen::Isometry3d T_hat_cam1_cam0 =
              cam1->T_camera_board * cam0->T_camera_board.inverse();
          AccumulateCoObsFactorResidual(
              ToIsometry3d(scene_state->T_cam1_cam0), T_hat_cam1_cam0,
              &coobs_factor_stats.stereo_rot_sum_rad,
              &coobs_factor_stats.stereo_rot_max_rad,
              &coobs_factor_stats.stereo_trans_sum_m,
              &coobs_factor_stats.stereo_trans_max_m);
          boost::shared_ptr<aslam::backend::ErrorTerm> error(
              new aslam::backend::ErrorTermTransformation(
                  T_cam1_cam0_expr,
                  sm::kinematics::Transformation(
                      ToMatrix4d(T_hat_cam1_cam0)),
                  stereo_weight, stereo_weight));
          error->setMEstimatorPolicy(
              boost::shared_ptr<aslam::backend::MEstimator>(
                  new aslam::backend::HuberMEstimator(
                      std::max(1e-12,
                               options.coobs_factor_ba_huber_delta))));
          problem->addErrorTerm(error);
          ++coobs_factor_stats.stereo_factor_count;
        }
      }
    }

    const double layout_weight =
        std::max(0.0, options.coobs_factor_ba_current_layout_weight);
    if (options.coobs_factor_ba_apply_layout_factor && layout_weight > 0.0) {
      std::set<std::pair<int, int> > selected_layout_pairs;
      for (std::pair<int, int> pair :
           options.coobs_factor_ba_layout_selected_pairs) {
        if (pair.second < pair.first) {
          std::swap(pair.first, pair.second);
        }
        if (pair.first != pair.second) {
          selected_layout_pairs.insert(pair);
        }
      }
      auto board_expression_for = [&](int board_id)
          -> aslam::backend::TransformationExpression {
        if (board_id == scene_state->gauge_fixed_board_id) {
          return identity_transform;
        }
        const auto board_it = board_variables.find(board_id);
        if (board_it == board_variables.end()) {
          return aslam::backend::TransformationExpression(
              Eigen::Matrix4d::Identity());
        }
        return board_it->second.expression;
      };
      for (int pair_index : selection_summary.selected_pair_indices) {
        for (int camera_index = 0; camera_index <= 1; ++camera_index) {
          for (const std::pair<int, int>& board_pair :
               selected_layout_pairs) {
            const int board_a = board_pair.first;
            const int board_b = board_pair.second;
            if (!PairBoardSelected(selection_summary, pair_index, board_a) ||
                !PairBoardSelected(selection_summary, pair_index, board_b)) {
              continue;
            }
            const CoObsLocalPoseMeasurement* meas_a =
                FindCoObsMeasurement(coobs_measurements, pair_index,
                                     camera_index, board_a);
            const CoObsLocalPoseMeasurement* meas_b =
                FindCoObsMeasurement(coobs_measurements, pair_index,
                                     camera_index, board_b);
            if (meas_a == nullptr || meas_b == nullptr) {
              continue;
            }
            ++coobs_factor_stats.layout_pair_candidate_count;
            const aslam::backend::TransformationExpression T_world_board_a =
                board_expression_for(board_a);
            const aslam::backend::TransformationExpression T_world_board_b =
                board_expression_for(board_b);
            if (board_a != scene_state->gauge_fixed_board_id &&
                board_variables.find(board_a) == board_variables.end()) {
              continue;
            }
            if (board_b != scene_state->gauge_fixed_board_id &&
                board_variables.find(board_b) == board_variables.end()) {
              continue;
            }
            const aslam::backend::TransformationExpression
                T_board_a_board_b_expr =
                    T_world_board_a.inverse() * T_world_board_b;
            const Eigen::Isometry3d T_hat_board_a_board_b =
                meas_a->T_camera_board.inverse() * meas_b->T_camera_board;
            Eigen::Isometry3d T_current_board_a_board_b =
                Eigen::Isometry3d::Identity();
            if (board_a == scene_state->gauge_fixed_board_id) {
              T_current_board_a_board_b =
                  board_b == scene_state->gauge_fixed_board_id
                      ? Eigen::Isometry3d::Identity()
                      : ToIsometry3d(
                            scene_state->T_world_board_by_id.at(board_b));
            } else if (board_b == scene_state->gauge_fixed_board_id) {
              T_current_board_a_board_b =
                  ToIsometry3d(
                      scene_state->T_world_board_by_id.at(board_a)).inverse();
            } else {
              T_current_board_a_board_b =
                  ToIsometry3d(
                      scene_state->T_world_board_by_id.at(board_a)).inverse() *
                  ToIsometry3d(
                      scene_state->T_world_board_by_id.at(board_b));
            }
            AccumulateCoObsFactorResidual(
                T_current_board_a_board_b, T_hat_board_a_board_b,
                &coobs_factor_stats.layout_rot_sum_rad,
                &coobs_factor_stats.layout_rot_max_rad,
                &coobs_factor_stats.layout_trans_sum_m,
                &coobs_factor_stats.layout_trans_max_m);
            boost::shared_ptr<aslam::backend::ErrorTerm> error(
                new aslam::backend::ErrorTermTransformation(
                    T_board_a_board_b_expr,
                    sm::kinematics::Transformation(
                        ToMatrix4d(T_hat_board_a_board_b)),
                    layout_weight, layout_weight));
            error->setMEstimatorPolicy(
                boost::shared_ptr<aslam::backend::MEstimator>(
                    new aslam::backend::HuberMEstimator(
                        std::max(1e-12,
                                 options.coobs_factor_ba_huber_delta))));
            problem->addErrorTerm(error);
            ++coobs_factor_stats.layout_factor_count;
          }
        }
      }
    }
  }

  ba_summary->reprojection_error_count = reprojection_error_count;
  ba_summary->coobs_stereo_factor_count =
      coobs_factor_stats.stereo_factor_count;
  ba_summary->coobs_layout_factor_count =
      coobs_factor_stats.layout_factor_count;
  ba_summary->coobs_stereo_factor_weight =
      options.coobs_factor_ba_current_stereo_weight;
  ba_summary->coobs_layout_factor_weight =
      options.coobs_factor_ba_current_layout_weight;
  if (coobs_factor_stats.stereo_factor_count > 0) {
    ba_summary->coobs_stereo_initial_rot_mean_deg =
        coobs_factor_stats.stereo_rot_sum_rad /
        coobs_factor_stats.stereo_factor_count * 180.0 / M_PI;
    ba_summary->coobs_stereo_initial_rot_max_deg =
        coobs_factor_stats.stereo_rot_max_rad * 180.0 / M_PI;
    ba_summary->coobs_stereo_initial_trans_mean_m =
        coobs_factor_stats.stereo_trans_sum_m /
        coobs_factor_stats.stereo_factor_count;
    ba_summary->coobs_stereo_initial_trans_max_m =
        coobs_factor_stats.stereo_trans_max_m;
  }
  if (coobs_factor_stats.layout_factor_count > 0) {
    ba_summary->coobs_layout_initial_rot_mean_deg =
        coobs_factor_stats.layout_rot_sum_rad /
        coobs_factor_stats.layout_factor_count * 180.0 / M_PI;
    ba_summary->coobs_layout_initial_rot_max_deg =
        coobs_factor_stats.layout_rot_max_rad * 180.0 / M_PI;
    ba_summary->coobs_layout_initial_trans_mean_m =
        coobs_factor_stats.layout_trans_sum_m /
        coobs_factor_stats.layout_factor_count;
    ba_summary->coobs_layout_initial_trans_max_m =
        coobs_factor_stats.layout_trans_max_m;
  }
  ba_summary->rig_param_mode = options.rig_param_mode;
  ba_summary->board_masking_use_local_board_pose_ba =
      options.board_masking_use_local_board_pose_ba;
  ba_summary->rig_camera_prior_translation_weight =
      options.rig_camera_prior_translation_weight;
  ba_summary->rig_camera_prior_rotation_weight =
      options.rig_camera_prior_rotation_weight;
  ba_summary->rig_stereo_relative_prior_weight =
      options.rig_stereo_relative_prior_weight;
  if (rig_centric) {
    ComputeRigProjectionEquivalenceDiagnostics(
        dataset, selection_summary, *scene_state, *cam0_geometry, *cam1_geometry,
        ToIsometry3d(T_cam0_rig_transform.T()),
        ToIsometry3d(T_cam1_rig_transform.T()), pair_variables,
        board_variables, &ba_summary->rig_projection_equivalence_max_pixel_diff,
        &ba_summary->rig_projection_equivalence_max_angular_diff_rad);
  }
  ba_summary->spherical_weight = options.spherical_weight;
  ba_summary->spherical_polar_weighting = options.spherical_polar_weighting;
  ba_summary->spherical_min_polar_deg = options.spherical_min_polar_deg;
  ba_summary->spherical_max_weight = options.spherical_max_weight;
  ba_summary->spherical_uncertainty_mode =
      options.spherical_uncertainty_mode;
  ba_summary->spherical_pixel_sigma_px = options.spherical_pixel_sigma_px;
  ba_summary->spherical_model_sigma = options.spherical_model_sigma;
  ba_summary->spherical_covariance_damping =
      options.spherical_covariance_damping;
  ba_summary->spherical_min_sigma_rad = options.spherical_min_sigma_rad;
  ba_summary->spherical_max_whitening_weight =
      options.spherical_max_whitening_weight;
  ba_summary->spherical_use_normalize_jacobian =
      options.spherical_use_normalize_jacobian;
  whitening_stats.WriteTo(ba_summary);
  ba_summary->optimize_intrinsics = optimize_intrinsics;
  ba_summary->optimize_stereo_extrinsic =
      options.final_ba_optimize_stereo_extrinsic;
  ba_summary->optimize_pair_poses = options.final_ba_optimize_pair_poses;
  ba_summary->optimize_board_poses = options.final_ba_optimize_board_poses;
  ba_summary->active_board_count = static_cast<int>(active_board_ids.size());
  ba_summary->shared_observation_count = shared_observation_count;
  ba_summary->cam0_only_observation_count = cam0_only_observation_count;
  ba_summary->cam1_only_observation_count = cam1_only_observation_count;
  ba_summary->shared_observation_weight_scale =
      options.ba_shared_observation_weight_scale;
  ba_summary->single_camera_only_observation_weight_scale =
      options.ba_single_camera_only_observation_weight_scale;
  ba_summary->single_camera_only_weight_mode =
      options.ba_single_camera_only_weight_mode;
  ba_summary->single_camera_only_base_scale =
      options.ba_single_camera_only_base_scale;
  ba_summary->single_camera_only_per_side_budget_ratio =
      options.ba_single_camera_only_per_side_budget_ratio;
  ba_summary->shared_total_base_weight = shared_total_base_weight;
  ba_summary->cam0_only_total_base_weight = cam0_only_total_base_weight;
  ba_summary->cam1_only_total_base_weight = cam1_only_total_base_weight;
  ba_summary->per_side_budget_limit = per_side_budget_limit;
  ba_summary->adaptive_single_camera_only_per_side_cap_ratio =
      options.ba_adaptive_single_camera_only_per_side_cap_ratio;
  ba_summary->cam0_only_cap = adaptive_per_side_cap;
  ba_summary->cam1_only_cap = adaptive_per_side_cap;
  ba_summary->cam0_only_effective_scale = cam0_only_effective_scale;
  ba_summary->cam1_only_effective_scale = cam1_only_effective_scale;
  ba_summary->cam0_only_budget_clamped = cam0_only_budget_clamped;
  ba_summary->cam1_only_budget_clamped = cam1_only_budget_clamped;
  ba_summary->shared_observation_weight_sum = shared_observation_weight_sum;
  ba_summary->cam0_only_observation_weight_sum = cam0_only_observation_weight_sum;
  ba_summary->cam1_only_observation_weight_sum = cam1_only_observation_weight_sum;
  if (reprojection_error_count <= 0) {
    ba_summary->failure_reason = "Global sparse BA problem contains zero error terms.";
    return false;
  }

  aslam::backend::OptimizerOptions optimizer_options;
  optimizer_options.maxIterations = options.ba_max_iterations;
  optimizer_options.convergenceDeltaJ = options.ba_convergence_threshold;
  optimizer_options.convergenceDeltaX = options.ba_convergence_threshold;
  optimizer_options.levenbergMarquardtLambdaInit = 10.0;
  optimizer_options.doLevenbergMarquardt = true;
  optimizer_options.doSchurComplement = false;
  optimizer_options.verbose = false;
  aslam::backend::Optimizer optimizer(optimizer_options);
  optimizer.setProblem(problem);
  ba_summary->objective_start = EvaluateTotalProblemObjective(problem.get());
  const aslam::backend::SolutionReturnValue solution = optimizer.optimize();
  ba_summary->objective_final = solution.JFinal;
  ba_summary->iterations = solution.iterations;
  ba_summary->failed_iterations = solution.failedIterations;
  ba_summary->linear_solver_failure = solution.linearSolverFailure;

  if (rig_centric) {
    const Eigen::Isometry3d final_T_cam0_rig =
        ToIsometry3d(T_cam0_rig_transform.T());
    const Eigen::Isometry3d final_T_cam1_rig =
        ToIsometry3d(T_cam1_rig_transform.T());
    const Eigen::Isometry3d final_T_cam1_cam0 =
        final_T_cam1_rig * final_T_cam0_rig.inverse();
    scene_state->T_cam1_cam0 = ToMatrix4d(final_T_cam1_cam0);
    ba_summary->rig_stereo_relative_rotation_drift_deg =
        RotationDistanceRadians(initial_T_cam1_cam0, final_T_cam1_cam0) *
        180.0 / M_PI;
    ba_summary->rig_stereo_relative_translation_drift_m =
        (final_T_cam1_cam0.translation() -
         initial_T_cam1_cam0.translation()).norm();
  } else {
    scene_state->T_cam1_cam0 = baseline_transform.T();
  }
  if (optimize_intrinsics) {
    scene_state->cam0 =
        UpdateStereoCalibrationFromGeometry<GeometryT0>(scene_state->cam0,
                                                       *cam0_geometry);
    scene_state->cam1 =
        UpdateStereoCalibrationFromGeometry<GeometryT1>(scene_state->cam1,
                                                       *cam1_geometry);
  }
  for (const auto& entry : pair_variables) {
    if (local_board_pose_ba) {
      const auto local_board_it = local_board_id_by_pair.find(entry.first);
      if (local_board_it == local_board_id_by_pair.end()) {
        continue;
      }
      const int board_id = local_board_it->second;
      const auto board_pose_it = scene_state->T_world_board_by_id.find(board_id);
      if (board_pose_it == scene_state->T_world_board_by_id.end()) {
        scene_state->T_cam0_world_by_pair[entry.first] =
            entry.second.transform.T();
        ba_summary->warnings.push_back(
            "local-board-pose BA could not fold pair pose through missing "
            "T_world_board; stored T_cam0_board as pair pose for pair " +
            std::to_string(entry.first) + ", board " +
            std::to_string(board_id));
        continue;
      }
      const Eigen::Isometry3d T_cam0_board =
          ToIsometry3d(entry.second.transform.T());
      const Eigen::Isometry3d T_world_board =
          ToIsometry3d(board_pose_it->second);
      scene_state->T_cam0_world_by_pair[entry.first] =
          ToMatrix4d(T_cam0_board * T_world_board.inverse());
    } else if (rig_centric) {
      const Eigen::Isometry3d final_T_cam0_rig =
          ToIsometry3d(T_cam0_rig_transform.T());
      const Eigen::Isometry3d final_T_rig_world =
          ToIsometry3d(entry.second.transform.T());
      scene_state->T_cam0_world_by_pair[entry.first] =
          ToMatrix4d(final_T_cam0_rig * final_T_rig_world);
    } else {
      scene_state->T_cam0_world_by_pair[entry.first] =
          entry.second.transform.T();
    }
  }
  for (const auto& entry : board_variables) {
    scene_state->T_world_board_by_id[entry.first] = entry.second.transform.T();
  }
  return true;
}

bool RunGlobalSparseBa(const StereoMeasurementDataset& dataset,
                       const StereoExtrinsicSolverOptions& options,
                       const StereoPairSelectionSummary& selection_summary,
                       StereoGlobalSparseBaSummary* ba_summary,
                       StereoSceneState* scene_state) {
  if (scene_state == nullptr || ba_summary == nullptr) {
    throw std::runtime_error("RunGlobalSparseBa requires valid outputs.");
  }
  const std::string cam0_family = scene_state->cam0.camera_model_family;
  const std::string cam1_family = scene_state->cam1.camera_model_family;
  if (cam0_family == "ds-none" && cam1_family == "ds-none") {
    return RunGlobalSparseBaTyped<DsGeometry, DsGeometry>(
        dataset, options, selection_summary, ba_summary, scene_state);
  }
  if (cam0_family == "ds-none" && cam1_family == "eucm-none") {
    return RunGlobalSparseBaTyped<DsGeometry, EucmGeometry>(
        dataset, options, selection_summary, ba_summary, scene_state);
  }
  if (cam0_family == "eucm-none" && cam1_family == "ds-none") {
    return RunGlobalSparseBaTyped<EucmGeometry, DsGeometry>(
        dataset, options, selection_summary, ba_summary, scene_state);
  }
  if (cam0_family == "eucm-none" && cam1_family == "eucm-none") {
    return RunGlobalSparseBaTyped<EucmGeometry, EucmGeometry>(
        dataset, options, selection_summary, ba_summary, scene_state);
  }
  if (cam0_family == "pinhole-equi" && cam1_family == "pinhole-equi") {
    return RunGlobalSparseBaTyped<PinholeEquiGeometry, PinholeEquiGeometry>(
        dataset, options, selection_summary, ba_summary, scene_state);
  }
  if (cam0_family == "ds-none" && cam1_family == "pinhole-equi") {
    return RunGlobalSparseBaTyped<DsGeometry, PinholeEquiGeometry>(
        dataset, options, selection_summary, ba_summary, scene_state);
  }
  if (cam0_family == "pinhole-equi" && cam1_family == "ds-none") {
    return RunGlobalSparseBaTyped<PinholeEquiGeometry, DsGeometry>(
        dataset, options, selection_summary, ba_summary, scene_state);
  }
  if (cam0_family == "eucm-none" && cam1_family == "pinhole-equi") {
    return RunGlobalSparseBaTyped<EucmGeometry, PinholeEquiGeometry>(
        dataset, options, selection_summary, ba_summary, scene_state);
  }
  if (cam0_family == "pinhole-equi" && cam1_family == "eucm-none") {
    return RunGlobalSparseBaTyped<PinholeEquiGeometry, EucmGeometry>(
        dataset, options, selection_summary, ba_summary, scene_state);
  }
  ba_summary->failure_reason =
      "Unsupported Stage6 global sparse BA camera families: " + cam0_family +
      " / " + cam1_family;
  return false;
}

int CountSharedInternalObservationsForPair(const StereoMeasurementDataset& dataset,
                                           int pair_index) {
  int count = 0;
  const auto shared_it = dataset.pair_shared_board_ids.find(pair_index);
  if (shared_it == dataset.pair_shared_board_ids.end()) {
    return 0;
  }
  for (const StereoObservation& observation : dataset.observations) {
    if (observation.pair_index == pair_index &&
        observation.point_type == JointPointType::Internal &&
        observation.used_in_solver &&
        shared_it->second.count(observation.board_id) > 0) {
      ++count;
    }
  }
  return count;
}

double ComputeStereoPairTrialCandidateScore(const StereoPairSelectionRow& row,
                                            int shared_internal_point_count,
                                            double coverage_gain) {
  double score = 0.0;
  score += 10.0 * static_cast<double>(row.shared_board_count);
  score += 1.5 * static_cast<double>(row.single_camera_only_board_count);
  score += 0.05 * static_cast<double>(row.shared_outer_point_count);
  score += 0.01 * static_cast<double>(shared_internal_point_count);
  score += 3.0 * coverage_gain;
  if (std::isfinite(row.pose_fit_rmse)) {
    score -= 0.25 * row.pose_fit_rmse;
  } else if (row.shared_board_count > 0) {
    score -= 100.0;
  }
  return score;
}

StereoPairSelectionSummary BuildSelectionSummaryFromSelectedPairs(
    const StereoPairSelectionSummary& base_summary,
    const std::set<int>& selected_pairs,
    StereoViewSelectionMode mode) {
  StereoPairSelectionSummary summary = base_summary;
  summary.mode = mode;
  summary.success = true;
  summary.selected_pair_indices = selected_pairs;
  summary.selected_pair_count = static_cast<int>(selected_pairs.size());
  summary.selected_pair_board_keys.clear();
  summary.selected_shared_board_pair_count = 0;
  summary.selected_single_camera_only_pair_count = 0;
  summary.selected_covered_board_count = 0;
  summary.covered_board_ids.clear();
  summary.selected_pose_fit_rmse_min = std::numeric_limits<double>::infinity();
  summary.selected_pose_fit_rmse_median = std::numeric_limits<double>::infinity();
  summary.selected_pose_fit_rmse_max = std::numeric_limits<double>::infinity();
  std::vector<double> selected_pose_fit_values;
  for (StereoPairSelectionRow& row : summary.rows) {
    row.selected = selected_pairs.count(row.pair_index) > 0;
    if (row.selected) {
      row.rejection_reason.clear();
      if (row.shared_board_count > 0) {
        ++summary.selected_shared_board_pair_count;
      } else {
        ++summary.selected_single_camera_only_pair_count;
      }
      summary.covered_board_ids.insert(row.covered_board_ids.begin(),
                                       row.covered_board_ids.end());
      for (int board_id : row.covered_board_ids) {
        summary.selected_pair_board_keys.insert(
            PairBoardKey(row.pair_index, board_id));
      }
      if (std::isfinite(row.pose_fit_rmse)) {
        selected_pose_fit_values.push_back(row.pose_fit_rmse);
      }
    } else if (row.eligible) {
      row.rejection_reason = "not_selected_by_trial_selection";
    }
  }
  summary.selected_covered_board_count =
      static_cast<int>(summary.covered_board_ids.size());
  if (!selected_pose_fit_values.empty()) {
    summary.selected_pose_fit_rmse_min =
        *std::min_element(selected_pose_fit_values.begin(),
                          selected_pose_fit_values.end());
    summary.selected_pose_fit_rmse_max =
        *std::max_element(selected_pose_fit_values.begin(),
                          selected_pose_fit_values.end());
    summary.selected_pose_fit_rmse_median =
        ComputeMedian(selected_pose_fit_values);
  }
  return summary;
}

StereoPairTrialSelectionDecision MakeTrialDecisionFromRow(
    const StereoMeasurementDataset& dataset,
    const StereoPairSelectionRow& row,
    const std::set<int>& covered_boards) {
  StereoPairTrialSelectionDecision decision;
  decision.pair_index = row.pair_index;
  const StereoFramePair* pair = FindPair(dataset, row.pair_index);
  if (pair != nullptr) {
    decision.left_frame_label = pair->left_frame_label;
    decision.right_frame_label = pair->right_frame_label;
  }
  decision.shared_board_count = row.shared_board_count;
  const auto cam0_only_it = dataset.pair_cam0_only_board_count.find(row.pair_index);
  const auto cam1_only_it = dataset.pair_cam1_only_board_count.find(row.pair_index);
  decision.cam0_only_board_count =
      cam0_only_it == dataset.pair_cam0_only_board_count.end()
          ? 0
          : cam0_only_it->second;
  decision.cam1_only_board_count =
      cam1_only_it == dataset.pair_cam1_only_board_count.end()
          ? 0
          : cam1_only_it->second;
  decision.shared_outer_point_count = row.shared_outer_point_count;
  decision.shared_internal_point_count =
      CountSharedInternalObservationsForPair(dataset, row.pair_index);
  int new_board_count = 0;
  for (int board_id : row.covered_board_ids) {
    if (covered_boards.count(board_id) == 0) {
      ++new_board_count;
    }
  }
  decision.coverage_gain = static_cast<double>(new_board_count);
  decision.candidate_score =
      ComputeStereoPairTrialCandidateScore(row,
                                           decision.shared_internal_point_count,
                                           decision.coverage_gain);
  return decision;
}

StereoPairTrialSelectionSummary RunKalibrStylePairTrialSelection(
    const StereoMeasurementDataset& dataset,
    const StereoExtrinsicSolverOptions& options,
    const StereoPairSelectionSummary& base_selection,
    StereoPairSelectionSummary* final_selection,
    StereoSceneState* scene_state) {
  StereoPairTrialSelectionSummary trial_summary;
  trial_summary.enabled = options.enable_kalibr_style_pair_selection;
  trial_summary.requested_seed_count = options.pair_selection_seed_count;
  if (final_selection == nullptr || scene_state == nullptr) {
    trial_summary.failure_reason = "missing_output_pointer";
    return trial_summary;
  }
  if (options.solver_mode != StereoSolverMode::GlobalSparseBa &&
      options.solver_mode != StereoSolverMode::SharedOnlyGlobalSparseBa) {
    trial_summary.failure_reason =
        "kalibr_style_pair_selection_requires_global_sparse_ba_solver_mode";
    return trial_summary;
  }

  std::vector<const StereoPairSelectionRow*> ranked_rows;
  std::vector<const StereoPairSelectionRow*> ranked_shared_rows;
  std::vector<const StereoPairSelectionRow*> ranked_single_only_rows;
  for (const StereoPairSelectionRow& row : base_selection.rows) {
    if (!row.eligible) {
      continue;
    }
    if (row.shared_board_count >= options.pair_selection_min_shared_boards) {
      ranked_shared_rows.push_back(&row);
    } else if (row.shared_board_count == 0 &&
               row.single_camera_only_board_count > 0) {
      ranked_single_only_rows.push_back(&row);
    }
  }
  std::sort(ranked_shared_rows.begin(), ranked_shared_rows.end(),
            [](const StereoPairSelectionRow* lhs,
               const StereoPairSelectionRow* rhs) {
              return IsSelectionRowBetter(*lhs, *rhs);
            });
  std::sort(ranked_single_only_rows.begin(), ranked_single_only_rows.end(),
            [](const StereoPairSelectionRow* lhs,
               const StereoPairSelectionRow* rhs) {
              if (lhs->single_camera_only_board_count !=
                  rhs->single_camera_only_board_count) {
                return lhs->single_camera_only_board_count >
                       rhs->single_camera_only_board_count;
              }
              if (lhs->covered_board_count != rhs->covered_board_count) {
                return lhs->covered_board_count > rhs->covered_board_count;
              }
              return lhs->pair_index < rhs->pair_index;
            });
  ranked_rows.reserve(ranked_shared_rows.size() + ranked_single_only_rows.size());
  ranked_rows.insert(ranked_rows.end(), ranked_shared_rows.begin(),
                     ranked_shared_rows.end());
  ranked_rows.insert(ranked_rows.end(), ranked_single_only_rows.begin(),
                     ranked_single_only_rows.end());
  trial_summary.candidate_count = static_cast<int>(ranked_rows.size());
  if (ranked_rows.empty()) {
    trial_summary.failure_reason = "no_eligible_trial_pair_candidates";
    return trial_summary;
  }

  std::set<int> selected_pairs;
  std::set<int> covered_boards;
  const std::vector<const StereoPairSelectionRow*>& seed_source =
      !ranked_shared_rows.empty() ? ranked_shared_rows : ranked_rows;
  int seed_count =
      std::min(std::max(1, options.pair_selection_seed_count),
               static_cast<int>(seed_source.size()));
  if (seed_count >= static_cast<int>(ranked_rows.size()) &&
      ranked_rows.size() > 1) {
    // Leave at least one pair for the incremental trial stage so the
    // Kalibr-style add-and-test logic actually runs.
    seed_count = static_cast<int>(ranked_rows.size()) - 1;
  }
  seed_count = std::min(seed_count, static_cast<int>(seed_source.size()));
  for (int index = 0; index < seed_count; ++index) {
    const StereoPairSelectionRow& row =
        *seed_source[static_cast<std::size_t>(index)];
    selected_pairs.insert(row.pair_index);
    covered_boards.insert(row.covered_board_ids.begin(), row.covered_board_ids.end());
    StereoPairTrialSelectionDecision decision =
        MakeTrialDecisionFromRow(dataset, row, covered_boards);
    decision.seed = true;
    decision.accepted = true;
    trial_summary.decisions.push_back(decision);
  }
  trial_summary.seed_count = static_cast<int>(selected_pairs.size());

  StereoExtrinsicSolverOptions trial_options = options;
  trial_options.final_ba_residual_mode = options.selection_ba_residual_mode;
  trial_options.final_ba_optimize_intrinsics = false;
  trial_options.final_ba_optimize_stereo_extrinsic = true;
  trial_options.final_ba_optimize_pair_poses = true;
  trial_options.final_ba_optimize_board_poses = true;
  trial_options.ba_max_iterations = std::max(1, std::min(options.ba_max_iterations, 3));
  trial_options.shared_board_quality_filter_final_ba = false;
  ApplySelectionCoObsFactorBaOptions(&trial_options);
  const bool commit_trial_state =
      options.selection_optimization_mode ==
      StereoSelectionOptimizationMode::TrialBaCommit;
  StereoSceneState current_scene = *scene_state;
  StereoPairSelectionSummary current_selection =
      BuildSelectionSummaryFromSelectedPairs(
          base_selection, selected_pairs, StereoViewSelectionMode::KalibrStyleTrial);
  StereoGlobalSparseBaSummary seed_ba_summary;
  StereoSceneState seed_trial_scene = current_scene;
  if (!RunGlobalSparseBa(dataset, trial_options, current_selection,
                         &seed_ba_summary, &seed_trial_scene)) {
    trial_summary.failure_reason =
        seed_ba_summary.failure_reason.empty()
            ? "seed_pair_trial_ba_failed"
            : seed_ba_summary.failure_reason;
    return trial_summary;
  }
  if (commit_trial_state) {
    current_scene = seed_trial_scene;
  }

  StereoResidualEvaluator evaluator(
      StereoResidualEvaluationOptions{
          false,
          options.pair_pose_refit_mode,
          options.symmetric_refit_max_iterations,
          options.symmetric_refit_step,
          false});
  StereoResidualSummary current_residual =
      evaluator.Evaluate(dataset, current_scene, selected_pairs,
                         "stage6_pair_trial_seed");
  trial_summary.initial_seed_rmse = current_residual.total_stereo_rmse;

  int valid_pair_candidate_count = 0;
  for (std::size_t ranked_index = static_cast<std::size_t>(seed_count);
       ranked_index < ranked_rows.size(); ++ranked_index) {
    const StereoPairSelectionRow& row = *ranked_rows[ranked_index];
    if (selected_pairs.count(row.pair_index) == 0) {
      ++valid_pair_candidate_count;
    }
  }
  const StereoCandidateTraversalBudget pair_budget =
      ComputeStereoCandidateTraversalBudget(
          options.pair_selection_budget_mode,
          valid_pair_candidate_count,
          options.pair_selection_max_candidate_additions,
          options.pair_selection_adaptive_budget_ratio,
          options.pair_selection_adaptive_budget_min,
          options.pair_selection_adaptive_budget_max,
          options.pair_selection_runtime_safety_ceiling);
  trial_summary.budget_mode = pair_budget.mode;
  trial_summary.valid_candidate_count = valid_pair_candidate_count;
  trial_summary.runtime_safety_ceiling =
      pair_budget.runtime_safety_ceiling;
  trial_summary.max_candidate_additions_effective =
      pair_budget.max_candidate_additions_effective;

  int added_count = 0;
  for (std::size_t ranked_index = static_cast<std::size_t>(seed_count);
       ranked_index < ranked_rows.size(); ++ranked_index) {
    if (pair_budget.mode == StereoCandidateBudgetMode::Fixed &&
        added_count >= pair_budget.traversal_limit) {
      break;
    }
    const StereoPairSelectionRow& row = *ranked_rows[ranked_index];
    if (selected_pairs.count(row.pair_index) > 0) {
      continue;
    }
    StereoPairTrialSelectionDecision decision =
        MakeTrialDecisionFromRow(dataset, row, covered_boards);
    if (pair_budget.mode != StereoCandidateBudgetMode::Fixed &&
        trial_summary.valid_candidate_traversed_count >=
            pair_budget.traversal_limit) {
      decision.reject_reason =
          pair_budget.mode == StereoCandidateBudgetMode::KalibrStyle
              ? "runtime_safety_ceiling"
              : "candidate_budget_limit";
      if (pair_budget.mode == StereoCandidateBudgetMode::KalibrStyle) {
        if (!trial_summary.safety_ceiling_hit) {
          trial_summary.safety_ceiling_hit = true;
          trial_summary.warnings.push_back(
              "runtime_safety_ceiling_hit; result is runtime-capped");
        }
      }
      trial_summary.decisions.push_back(decision);
      ++trial_summary.rejected_count;
      break;
    }
    ++trial_summary.valid_candidate_traversed_count;
    decision.attempted = true;
    ++trial_summary.attempted_count;

    std::set<int> trial_selected_pairs = selected_pairs;
    trial_selected_pairs.insert(row.pair_index);
    StereoPairSelectionSummary trial_selection =
        BuildSelectionSummaryFromSelectedPairs(
            base_selection, trial_selected_pairs,
            StereoViewSelectionMode::KalibrStyleTrial);
    StereoSceneState trial_scene = current_scene;
    const Eigen::Isometry3d baseline_before =
        ToIsometry3d(current_scene.T_cam1_cam0);
    StereoGlobalSparseBaSummary trial_ba_summary;
    if (!RunGlobalSparseBa(dataset, trial_options, trial_selection,
                           &trial_ba_summary, &trial_scene)) {
      decision.reject_reason =
          trial_ba_summary.failure_reason.empty() ? "trial_ba_failed"
                                                  : trial_ba_summary.failure_reason;
      trial_summary.decisions.push_back(decision);
      ++trial_summary.rejected_count;
      continue;
    }
    const StereoResidualSummary trial_residual =
        evaluator.Evaluate(dataset, trial_scene, trial_selected_pairs,
                           "stage6_pair_trial_candidate");
    decision.initial_total_rmse = current_residual.total_stereo_rmse;
    decision.trial_total_rmse = trial_residual.total_stereo_rmse;
    decision.total_rmse_delta =
        trial_residual.total_stereo_rmse - current_residual.total_stereo_rmse;
    decision.cam0_rmse_delta =
        trial_residual.cam0_rmse - current_residual.cam0_rmse;
    decision.cam1_rmse_delta =
        trial_residual.cam1_rmse - current_residual.cam1_rmse;
    const Eigen::Isometry3d baseline_after =
        ToIsometry3d(trial_scene.T_cam1_cam0);
    decision.baseline_rotation_delta_deg =
        RotationDistanceRadians(baseline_before, baseline_after) * 180.0 / M_PI;
    decision.baseline_translation_delta_m =
        (baseline_before.translation() - baseline_after.translation()).norm();

    if (decision.total_rmse_delta > options.pair_selection_max_rmse_delta) {
      decision.reject_reason = "total_rmse_delta_gate";
    } else if (decision.cam0_rmse_delta >
               options.pair_selection_max_camera_rmse_delta) {
      decision.reject_reason = "cam0_rmse_delta_gate";
    } else if (decision.cam1_rmse_delta >
               options.pair_selection_max_camera_rmse_delta) {
      decision.reject_reason = "cam1_rmse_delta_gate";
    } else if (decision.baseline_rotation_delta_deg >
               options.pair_selection_max_baseline_rotation_delta_deg) {
      decision.reject_reason = "baseline_rotation_delta_gate";
    } else if (decision.baseline_translation_delta_m >
               options.pair_selection_max_baseline_translation_delta_m) {
      decision.reject_reason = "baseline_translation_delta_gate";
    } else {
      decision.accepted = true;
    }

    if (decision.accepted) {
      selected_pairs.insert(row.pair_index);
      covered_boards.insert(row.covered_board_ids.begin(), row.covered_board_ids.end());
      if (commit_trial_state) {
        current_scene = trial_scene;
      }
      current_residual = trial_residual;
      ++trial_summary.accepted_count;
      ++added_count;
    } else {
      ++trial_summary.rejected_count;
    }
    trial_summary.decisions.push_back(decision);
  }

  trial_summary.final_selected_pair_count = static_cast<int>(selected_pairs.size());
  trial_summary.final_selected_rmse = current_residual.total_stereo_rmse;
  trial_summary.selected_pair_indices = selected_pairs;
  trial_summary.success = !selected_pairs.empty();
  *scene_state = current_scene;
  *final_selection =
      BuildSelectionSummaryFromSelectedPairs(
          base_selection, selected_pairs, StereoViewSelectionMode::KalibrStyleTrial);
  return trial_summary;
}

StereoMeasurementDataset MakePairBoardMaskedDataset(
    const StereoMeasurementDataset& dataset,
    const std::set<PairBoardKey>& selected_pair_boards) {
  if (selected_pair_boards.empty()) {
    return dataset;
  }
  StereoMeasurementDataset masked = dataset;
  for (StereoObservation& observation : masked.observations) {
    if (!observation.used_in_solver) {
      continue;
    }
    const bool is_training =
        std::find(masked.training_pair_indices.begin(),
                  masked.training_pair_indices.end(),
                  observation.pair_index) != masked.training_pair_indices.end();
    if (!is_training) {
      continue;
    }
    observation.used_in_solver =
        selected_pair_boards.count(
            PairBoardKey(observation.pair_index, observation.board_id)) > 0;
  }
  return masked;
}

int CountSharedObservationsForPairBoard(const StereoMeasurementDataset& dataset,
                                        int pair_index,
                                        int board_id) {
  int count = 0;
  for (const StereoObservation& observation : dataset.observations) {
    if (observation.pair_index == pair_index &&
        observation.board_id == board_id &&
        observation.used_in_solver) {
      ++count;
    }
  }
  return count;
}

double ComputePairBoardTrialCandidateScore(
    const StereoPairBoardTrialSelectionDecision& decision) {
  double score = 0.0;
  score += decision.shared_board ? 20.0 : 0.0;
  score += 0.03 * static_cast<double>(decision.shared_point_count);
  score += 0.25 * static_cast<double>(
      decision.cam0_outer_point_count + decision.cam1_outer_point_count);
  if (std::isfinite(decision.cam0_outer_rmse)) {
    score -= 0.6 * decision.cam0_outer_rmse;
  }
  if (std::isfinite(decision.cam1_outer_rmse)) {
    score -= 0.6 * decision.cam1_outer_rmse;
  }
  return score;
}

std::set<int> PairIndicesFromPairBoards(
    const std::set<PairBoardKey>& selected_pair_boards) {
  std::set<int> pair_indices;
  for (const PairBoardKey& key : selected_pair_boards) {
    pair_indices.insert(key.first);
  }
  return pair_indices;
}

std::set<int> BoardIdsFromPairBoards(
    const std::set<PairBoardKey>& selected_pair_boards) {
  std::set<int> board_ids;
  for (const PairBoardKey& key : selected_pair_boards) {
    board_ids.insert(key.second);
  }
  return board_ids;
}

int CountSelectedPairBoardForPair(
    const std::set<PairBoardKey>& selected_pair_boards,
    int pair_index) {
  int count = 0;
  for (const PairBoardKey& key : selected_pair_boards) {
    if (key.first == pair_index) {
      ++count;
    }
  }
  return count;
}

int CountSelectedPairBoardForBoard(
    const std::set<PairBoardKey>& selected_pair_boards,
    int board_id) {
  int count = 0;
  for (const PairBoardKey& key : selected_pair_boards) {
    if (key.second == board_id) {
      ++count;
    }
  }
  return count;
}

double ComputePairBoardCoverageGain(
    const std::set<PairBoardKey>& selected_pair_boards,
    int pair_index,
    int board_id) {
  const std::set<int> selected_pairs = PairIndicesFromPairBoards(selected_pair_boards);
  const std::set<int> selected_boards = BoardIdsFromPairBoards(selected_pair_boards);
  double gain = 0.0;
  if (selected_pairs.count(pair_index) == 0) {
    gain += 1.0;
  }
  if (selected_boards.count(board_id) == 0) {
    gain += 1.0;
  }
  if (CountSelectedPairBoardForPair(selected_pair_boards, pair_index) == 0) {
    gain += 0.5;
  }
  if (CountSelectedPairBoardForBoard(selected_pair_boards, board_id) == 0) {
    gain += 0.5;
  }
  return gain;
}

std::map<int, int> CountSelectedBoardsPerPair(
    const std::set<PairBoardKey>& selected_pair_boards) {
  std::map<int, int> counts;
  for (const PairBoardKey& key : selected_pair_boards) {
    ++counts[key.first];
  }
  return counts;
}

int CountPairsBelowSelectedBoardMinimum(
    const std::set<PairBoardKey>& selected_pair_boards,
    int min_boards_per_pair) {
  if (min_boards_per_pair <= 1) {
    return 0;
  }
  int count = 0;
  const std::map<int, int> counts =
      CountSelectedBoardsPerPair(selected_pair_boards);
  for (const auto& entry : counts) {
    if (entry.second < min_boards_per_pair) {
      ++count;
    }
  }
  return count;
}

int CountPairsBelowPairCohesionTarget(
    const std::set<PairBoardKey>& selected_pair_boards,
    const StereoMeasurementDataset& dataset,
    int min_boards_per_pair) {
  int count = 0;
  const std::map<int, int> counts =
      CountSelectedBoardsPerPair(selected_pair_boards);
  for (const auto& entry : counts) {
    const auto shared_it = dataset.pair_shared_board_ids.find(entry.first);
    if (shared_it == dataset.pair_shared_board_ids.end()) {
      continue;
    }
    const int target_board_count =
        std::max(min_boards_per_pair,
                 static_cast<int>(shared_it->second.size()));
    if (entry.second < target_board_count) {
      ++count;
    }
  }
  return count;
}

void DropPairsBelowSelectedBoardMinimum(
    std::set<PairBoardKey>* selected_pair_boards,
    int min_boards_per_pair,
    int* dropped_pair_count) {
  if (dropped_pair_count != nullptr) {
    *dropped_pair_count = 0;
  }
  if (selected_pair_boards == nullptr || min_boards_per_pair <= 1) {
    return;
  }
  const std::map<int, int> counts =
      CountSelectedBoardsPerPair(*selected_pair_boards);
  std::set<int> pairs_to_drop;
  for (const auto& entry : counts) {
    if (entry.second < min_boards_per_pair) {
      pairs_to_drop.insert(entry.first);
    }
  }
  if (dropped_pair_count != nullptr) {
    *dropped_pair_count = static_cast<int>(pairs_to_drop.size());
  }
  for (auto it = selected_pair_boards->begin();
       it != selected_pair_boards->end();) {
    if (pairs_to_drop.count(it->first) > 0) {
      it = selected_pair_boards->erase(it);
    } else {
      ++it;
    }
  }
}

std::string FormatPairBoardKey(const PairBoardKey& key) {
  std::ostringstream stream;
  stream << key.first << ":" << key.second;
  return stream.str();
}

int ApplyAblationExcludedPairBoards(
    const std::vector<std::pair<int, int> >& excluded_pair_boards,
    std::set<PairBoardKey>* selected_pair_boards,
    std::vector<std::string>* warnings) {
  if (selected_pair_boards == nullptr || excluded_pair_boards.empty()) {
    return 0;
  }
  int removed_count = 0;
  for (const std::pair<int, int>& excluded : excluded_pair_boards) {
    const PairBoardKey key(excluded.first, excluded.second);
    const int removed = static_cast<int>(selected_pair_boards->erase(key));
    removed_count += removed;
    if (warnings != nullptr) {
      std::ostringstream warning;
      warning << "ablation_exclude_pair_board="
              << FormatPairBoardKey(key)
              << " removed=" << removed;
      warnings->push_back(warning.str());
    }
  }
  return removed_count;
}

StereoPairSelectionSummary BuildSelectionSummaryFromSelectedPairBoards(
    const StereoPairSelectionSummary& base_summary,
    const StereoMeasurementDataset& dataset,
    const std::set<PairBoardKey>& selected_pair_boards);

bool RefitAfterPairBoardAblation(
    const StereoMeasurementDataset& dataset,
    const StereoExtrinsicSolverOptions& options,
    const StereoExtrinsicSolverOptions& ba_options,
    const StereoPairSelectionSummary& base_selection,
    const std::set<PairBoardKey>& selected_pair_boards,
    const std::string& evaluation_label,
    StereoSceneState* current_scene,
    StereoResidualSummary* current_residual,
    std::vector<std::string>* warnings) {
  if (current_scene == nullptr || current_residual == nullptr) {
    return false;
  }
  if (selected_pair_boards.empty()) {
    if (warnings != nullptr) {
      warnings->push_back("ablation_refit_skipped_empty_selection");
    }
    return false;
  }
  const StereoPairSelectionSummary ablation_selection =
      BuildSelectionSummaryFromSelectedPairBoards(
          base_selection, dataset, selected_pair_boards);
  const StereoMeasurementDataset ablation_dataset =
      MakePairBoardMaskedDataset(dataset, selected_pair_boards);
  StereoGlobalSparseBaSummary ba_summary;
  StereoSceneState ablation_scene = *current_scene;
  if (!RunGlobalSparseBa(ablation_dataset, ba_options, ablation_selection,
                         &ba_summary, &ablation_scene)) {
    if (warnings != nullptr) {
      warnings->push_back(
          ba_summary.failure_reason.empty()
              ? "ablation_refit_failed"
              : "ablation_refit_failed:" + ba_summary.failure_reason);
    }
    return false;
  }
  *current_scene = ablation_scene;
  const StereoResidualEvaluator evaluator(
      StereoResidualEvaluationOptions{
          false,
          options.pair_pose_refit_mode,
          options.symmetric_refit_max_iterations,
          options.symmetric_refit_step,
          false});
  *current_residual = evaluator.Evaluate(
      ablation_dataset,
      *current_scene,
      ablation_selection.selected_pair_indices,
      evaluation_label);
  if (warnings != nullptr) {
    warnings->push_back("ablation_refit_success");
  }
  return true;
}

StereoPairSelectionSummary BuildSelectionSummaryFromSelectedPairBoards(
    const StereoPairSelectionSummary& base_summary,
    const StereoMeasurementDataset& dataset,
    const std::set<PairBoardKey>& selected_pair_boards) {
  StereoPairSelectionSummary summary =
      BuildSelectionSummaryFromSelectedPairs(
          base_summary, PairIndicesFromPairBoards(selected_pair_boards),
          StereoViewSelectionMode::KalibrStyleTrial);
  summary.selected_pair_board_keys = selected_pair_boards;
  summary.covered_board_ids.clear();
  for (const PairBoardKey& key : selected_pair_boards) {
    summary.covered_board_ids.insert(key.second);
  }
  summary.selected_covered_board_count =
      static_cast<int>(summary.covered_board_ids.size());
  for (StereoPairSelectionRow& row : summary.rows) {
    if (!row.selected) {
      continue;
    }
    bool has_selected_board = false;
    for (int board_id : row.covered_board_ids) {
      if (selected_pair_boards.count(PairBoardKey(row.pair_index, board_id)) > 0) {
        has_selected_board = true;
        break;
      }
    }
    row.selected = has_selected_board;
    if (!has_selected_board) {
      summary.selected_pair_indices.erase(row.pair_index);
      row.rejection_reason = "no_selected_pair_board";
    }
  }
  summary.selected_pair_count =
      static_cast<int>(summary.selected_pair_indices.size());
  (void)dataset;
  return summary;
}

Stage6IncrementalBatchEstimatorOptions MakeStage6IncrementalEstimatorOptions(
    const StereoExtrinsicSolverOptions& options,
    bool pair_level);

Stage6IncrementalBatchCandidate MakeStage6IncrementalBatchCandidate(
    const char* batch_type,
    int pair_index,
    const std::set<PairBoardKey>& selected_pair_boards_before,
    const std::set<PairBoardKey>& selected_pair_boards_after,
    bool force);

void CopyIncrementalBatchResultToDecision(
    const Stage6IncrementalBatchResult& result,
    const std::string& batch_type,
    StereoPairBoardTrialSelectionDecision* decision);

StereoPairBoardTrialSelectionSummary RunKalibrStylePairBoardTrialSelection(
    const StereoMeasurementDataset& dataset,
    const StereoExtrinsicSolverOptions& options,
    const StereoPairSelectionSummary& base_selection,
    StereoPairSelectionSummary* final_selection,
    StereoSceneState* scene_state) {
  StereoPairBoardTrialSelectionSummary summary;
  summary.enabled = options.enable_pair_board_trial_selection;
  summary.pairboard_selection_mode = options.pairboard_selection_mode;
  if (final_selection == nullptr || scene_state == nullptr) {
    summary.failure_reason = "missing_output_pointer";
    return summary;
  }
  if (!options.enable_pair_board_trial_selection) {
    summary.failure_reason = "disabled";
    return summary;
  }
  if (options.solver_mode != StereoSolverMode::GlobalSparseBa &&
      options.solver_mode != StereoSolverMode::SharedOnlyGlobalSparseBa) {
    summary.failure_reason =
        "pair_board_trial_selection_requires_global_sparse_ba_solver_mode";
    return summary;
  }

  std::vector<StereoPairBoardTrialSelectionDecision> seed_decisions;
  std::vector<StereoPairBoardTrialSelectionDecision> candidate_decisions;
  for (const StereoPairSelectionRow& pair_row : base_selection.rows) {
    if (!pair_row.eligible || pair_row.shared_board_count <= 0) {
      continue;
    }
    const auto shared_it = dataset.pair_shared_board_ids.find(pair_row.pair_index);
    if (shared_it == dataset.pair_shared_board_ids.end()) {
      continue;
    }
    for (int board_id : shared_it->second) {
      StereoPairBoardTrialSelectionDecision decision;
      decision.pair_index = pair_row.pair_index;
      decision.board_id = board_id;
      decision.pairboard_selection_mode = options.pairboard_selection_mode;
      decision.shared_board = true;
      decision.cam0_outer_point_count =
          CountOuterObservationsForCamera(dataset, pair_row.pair_index, 0, board_id);
      decision.cam1_outer_point_count =
          CountOuterObservationsForCamera(dataset, pair_row.pair_index, 1, board_id);
      decision.shared_point_count =
          CountSharedObservationsForPairBoard(dataset, pair_row.pair_index, board_id);
      const StereoSharedBoardQuality quality = EvaluateSharedBoardQuality(
          dataset, *scene_state, options, pair_row.pair_index, board_id);
      decision.cam0_outer_rmse = quality.cam0_outer_rmse;
      decision.cam1_outer_rmse = quality.cam1_outer_rmse;
      decision.candidate_score = ComputePairBoardTrialCandidateScore(decision);
      if (quality.pass) {
        decision.seed = true;
        decision.accepted = true;
        seed_decisions.push_back(decision);
      } else {
        candidate_decisions.push_back(decision);
      }
    }
  }

  std::sort(seed_decisions.begin(), seed_decisions.end(),
            [](const StereoPairBoardTrialSelectionDecision& lhs,
               const StereoPairBoardTrialSelectionDecision& rhs) {
              if (lhs.candidate_score != rhs.candidate_score) {
                return lhs.candidate_score > rhs.candidate_score;
              }
              if (lhs.pair_index != rhs.pair_index) {
                return lhs.pair_index < rhs.pair_index;
              }
              return lhs.board_id < rhs.board_id;
            });
  std::sort(candidate_decisions.begin(), candidate_decisions.end(),
            [](const StereoPairBoardTrialSelectionDecision& lhs,
               const StereoPairBoardTrialSelectionDecision& rhs) {
              if (lhs.candidate_score != rhs.candidate_score) {
                return lhs.candidate_score > rhs.candidate_score;
              }
              if (lhs.pair_index != rhs.pair_index) {
                return lhs.pair_index < rhs.pair_index;
              }
              return lhs.board_id < rhs.board_id;
            });

  summary.candidate_count =
      static_cast<int>(seed_decisions.size() + candidate_decisions.size());
  if (seed_decisions.empty()) {
    summary.failure_reason = "no_quality_seed_pair_boards";
    return summary;
  }

  std::set<PairBoardKey> selected_pair_boards;
  const int requested_seed_count =
      std::max(1, std::min(options.pair_board_selection_seed_count,
                           static_cast<int>(seed_decisions.size())));
  for (int index = 0; index < requested_seed_count; ++index) {
    const StereoPairBoardTrialSelectionDecision& decision =
        seed_decisions[static_cast<std::size_t>(index)];
    selected_pair_boards.insert(PairBoardKey(decision.pair_index, decision.board_id));
    summary.decisions.push_back(decision);
  }
  summary.seed_count = static_cast<int>(selected_pair_boards.size());
  for (std::size_t index = static_cast<std::size_t>(requested_seed_count);
       index < seed_decisions.size(); ++index) {
    StereoPairBoardTrialSelectionDecision decision = seed_decisions[index];
    decision.seed = false;
    decision.accepted = false;
    candidate_decisions.push_back(decision);
  }
  std::sort(candidate_decisions.begin(), candidate_decisions.end(),
            [](const StereoPairBoardTrialSelectionDecision& lhs,
               const StereoPairBoardTrialSelectionDecision& rhs) {
              if (lhs.candidate_score != rhs.candidate_score) {
                return lhs.candidate_score > rhs.candidate_score;
              }
              if (lhs.pair_index != rhs.pair_index) {
                return lhs.pair_index < rhs.pair_index;
              }
              return lhs.board_id < rhs.board_id;
            });
  const StereoCandidateTraversalBudget pair_board_budget =
      ComputeStereoCandidateTraversalBudget(
          options.pair_board_selection_budget_mode,
          static_cast<int>(candidate_decisions.size()),
          options.pair_board_selection_max_candidate_additions,
          options.pair_board_selection_adaptive_budget_ratio,
          options.pair_board_selection_adaptive_budget_min,
          options.pair_board_selection_adaptive_budget_max,
          options.pair_board_selection_runtime_safety_ceiling);
  summary.budget_mode = pair_board_budget.mode;
  summary.valid_candidate_count =
      static_cast<int>(candidate_decisions.size());
  summary.runtime_safety_ceiling =
      pair_board_budget.runtime_safety_ceiling;
  summary.max_candidate_additions_effective =
      pair_board_budget.max_candidate_additions_effective;
  const bool kalibr_information_gain_policy =
      options.batch_acceptance_policy ==
      KalibrStyleBatchAcceptancePolicy::KalibrInformationGain;
  summary.incremental_estimator_enabled = kalibr_information_gain_policy;
  summary.marginal_information_gain_proxy_enabled =
      kalibr_information_gain_policy;
  summary.incremental_mi_tol = options.incremental_mi_tol;
  summary.incremental_rank_threshold = options.incremental_rank_threshold;
  summary.incremental_info_block = ToString(options.incremental_info_block);

  StereoExtrinsicSolverOptions trial_options = options;
  trial_options.final_ba_residual_mode = options.selection_ba_residual_mode;
  trial_options.final_ba_optimize_intrinsics = false;
  trial_options.final_ba_optimize_stereo_extrinsic = true;
  trial_options.final_ba_optimize_pair_poses = true;
  trial_options.final_ba_optimize_board_poses = true;
  trial_options.ba_max_iterations = std::max(1, std::min(options.ba_max_iterations, 10));
  ApplySelectionCoObsFactorBaOptions(&trial_options);
  const bool commit_trial_state =
      options.selection_optimization_mode ==
      StereoSelectionOptimizationMode::TrialBaCommit;
  const bool information_gain_only =
      options.selection_optimization_mode ==
      StereoSelectionOptimizationMode::InformationGainOnly;
  StereoSceneState current_scene = *scene_state;
  StereoPairSelectionSummary current_selection =
      BuildSelectionSummaryFromSelectedPairBoards(
          base_selection, dataset, selected_pair_boards);
  StereoMeasurementDataset current_dataset =
      MakePairBoardMaskedDataset(dataset, selected_pair_boards);
  StereoGlobalSparseBaSummary seed_ba_summary;
  StereoSceneState seed_trial_scene = current_scene;
  if (!information_gain_only &&
      !RunGlobalSparseBa(current_dataset, trial_options, current_selection,
                         &seed_ba_summary, &seed_trial_scene)) {
    summary.failure_reason =
        seed_ba_summary.failure_reason.empty()
            ? "seed_pair_board_trial_ba_failed"
            : seed_ba_summary.failure_reason;
    return summary;
  }
  if (commit_trial_state) {
    current_scene = seed_trial_scene;
  }

  StereoResidualEvaluator evaluator(
      StereoResidualEvaluationOptions{
          false,
          options.pair_pose_refit_mode,
          options.symmetric_refit_max_iterations,
          options.symmetric_refit_step,
          false});
  StereoResidualSummary current_residual = evaluator.Evaluate(
      current_dataset, current_scene, current_selection.selected_pair_indices,
      "stage6_pair_board_trial_seed");
  summary.initial_seed_rmse = current_residual.total_stereo_rmse;
  const Stage6IncrementalBatchEstimator pair_board_estimator(
      MakeStage6IncrementalEstimatorOptions(options, false));

  auto try_add_candidate =
      [&](StereoPairBoardTrialSelectionDecision* decision,
          bool enforce_score_gate) {
        if (decision == nullptr) {
          return false;
        }
        const bool kalibr_style_batch_mode =
            options.pairboard_selection_mode ==
            StereoPairBoardSelectionMode::KalibrStyleBatch;
        decision->pairboard_selection_mode = options.pairboard_selection_mode;
        const PairBoardKey key(decision->pair_index, decision->board_id);
        if (selected_pair_boards.count(key) > 0) {
          return false;
        }
        decision->selected_pair_board_count_before =
            static_cast<int>(selected_pair_boards.size());
        decision->selected_pair_count_before =
            static_cast<int>(
                PairIndicesFromPairBoards(selected_pair_boards).size());
        decision->selected_board_count_before =
            static_cast<int>(
                BoardIdsFromPairBoards(selected_pair_boards).size());
        decision->coverage_gain =
            ComputePairBoardCoverageGain(selected_pair_boards,
                                         decision->pair_index,
                                         decision->board_id);
        const int selected_for_pair =
            CountSelectedPairBoardForPair(selected_pair_boards,
                                          decision->pair_index);
        const int selected_for_board =
            CountSelectedPairBoardForBoard(selected_pair_boards,
                                           decision->board_id);
        const bool pair_cap_exceeded =
            options.pair_board_selection_max_accepted_per_pair > 0 &&
            selected_for_pair >=
                options.pair_board_selection_max_accepted_per_pair;
        const bool board_cap_exceeded =
            options.pair_board_selection_max_accepted_per_board > 0 &&
            selected_for_board >=
                options.pair_board_selection_max_accepted_per_board;
        if (!kalibr_style_batch_mode && enforce_score_gate &&
            decision->candidate_score <
                options.pair_board_selection_min_candidate_score) {
          decision->reject_reason = "candidate_score_gate";
          return false;
        }
        if (!kalibr_style_batch_mode &&
            decision->coverage_gain <
            options.pair_board_selection_min_coverage_gain) {
          decision->reject_reason = "coverage_gain_gate";
          return false;
        }
        const bool relax_cap_gates =
            decision->pair_cohesion_candidate &&
            options.pair_cohesion_relax_cap_gates;
        const auto is_pair_completion_candidate = [&]() {
          const auto shared_it =
              dataset.pair_shared_board_ids.find(decision->pair_index);
          const int shared_board_capacity =
              shared_it == dataset.pair_shared_board_ids.end()
                  ? 0
                  : static_cast<int>(shared_it->second.size());
          return selected_for_pair > 0 &&
                 selected_for_pair < shared_board_capacity;
        };
        if (!kalibr_style_batch_mode && !relax_cap_gates &&
            pair_cap_exceeded) {
          decision->reject_reason = "max_accepted_per_pair_gate";
          return false;
        }
        if (!kalibr_style_batch_mode && !relax_cap_gates &&
            board_cap_exceeded) {
          decision->reject_reason = "max_accepted_per_board_gate";
          return false;
        }
        if (relax_cap_gates) {
          decision->pair_cohesion_cap_gate_relaxed = true;
        }
        decision->attempted = true;

        std::set<PairBoardKey> trial_pair_boards = selected_pair_boards;
        trial_pair_boards.insert(key);
        StereoPairSelectionSummary trial_selection =
            BuildSelectionSummaryFromSelectedPairBoards(
                base_selection, dataset, trial_pair_boards);
        StereoMeasurementDataset trial_dataset =
            MakePairBoardMaskedDataset(dataset, trial_pair_boards);
        StereoSceneState trial_scene = current_scene;
        const Eigen::Isometry3d baseline_before =
            ToIsometry3d(current_scene.T_cam1_cam0);
        if (information_gain_only) {
          ++summary.batch_acceptance_attempted_count;
          const Stage6IncrementalBatchCandidate batch_candidate =
              MakeStage6IncrementalBatchCandidate(
                  decision->pair_cohesion_candidate
                      ? "pair_board_cohesion_information_gain_only"
                      : "pair_board_information_gain_only",
                  decision->pair_index, selected_pair_boards,
                  trial_pair_boards, false);
          const Stage6IncrementalBatchResult batch_result =
              pair_board_estimator.EvaluateInformationGainOnly(
                  dataset, current_scene, batch_candidate);
          CopyIncrementalBatchResultToDecision(
              batch_result, batch_candidate.batch_type, decision);
          decision->selection_optimization_mode =
              ToString(options.selection_optimization_mode);
          if (batch_result.batchAccepted) {
            decision->accepted = true;
            ++summary.batch_acceptance_accepted_count;
          } else {
            ++summary.batch_acceptance_rejected_score_count;
          }
          if (decision->accepted) {
            selected_pair_boards.insert(key);
            current_dataset = trial_dataset;
            current_selection = trial_selection;
            return true;
          }
          return false;
        }
        StereoGlobalSparseBaSummary trial_ba_summary;
        if (!RunGlobalSparseBa(trial_dataset, trial_options, trial_selection,
                               &trial_ba_summary, &trial_scene)) {
          if (kalibr_style_batch_mode) {
            ++summary.batch_acceptance_attempted_count;
            ++summary.batch_acceptance_rejected_hard_validity_count;
          }
          decision->reject_reason =
              kalibr_style_batch_mode
                  ? "hard_validity_gate"
                  : (trial_ba_summary.failure_reason.empty()
                         ? "trial_ba_failed"
                         : trial_ba_summary.failure_reason);
          return false;
        }
        CopyCoObsTrialDiagnostics(trial_ba_summary, decision);
        const StereoResidualSummary trial_residual = evaluator.Evaluate(
            trial_dataset, trial_scene, trial_selection.selected_pair_indices,
            decision->pair_cohesion_candidate
                ? "stage6_pair_board_trial_pair_cohesion"
                : "stage6_pair_board_trial_candidate");
        decision->initial_total_rmse = current_residual.total_stereo_rmse;
        decision->trial_total_rmse = trial_residual.total_stereo_rmse;
        decision->total_rmse_delta =
            trial_residual.total_stereo_rmse -
            current_residual.total_stereo_rmse;
        decision->cam0_rmse_delta =
            trial_residual.cam0_rmse - current_residual.cam0_rmse;
        decision->cam1_rmse_delta =
            trial_residual.cam1_rmse - current_residual.cam1_rmse;
        const Eigen::Isometry3d baseline_after =
            ToIsometry3d(trial_scene.T_cam1_cam0);
        decision->baseline_rotation_delta_deg =
            RotationDistanceRadians(baseline_before, baseline_after) *
            180.0 / M_PI;
        decision->baseline_translation_delta_m =
            (baseline_before.translation() -
             baseline_after.translation()).norm();

        decision->legacy_rmse_pass =
            decision->total_rmse_delta <=
                options.pair_board_selection_max_rmse_delta &&
            decision->cam0_rmse_delta <=
                options.pair_board_selection_max_camera_rmse_delta &&
            decision->cam1_rmse_delta <=
                options.pair_board_selection_max_camera_rmse_delta;

        if (!kalibr_style_batch_mode) {
          if (decision->total_rmse_delta >
              options.pair_board_selection_max_rmse_delta) {
            decision->reject_reason = "total_rmse_delta_gate";
          } else if (decision->cam0_rmse_delta >
                     options.pair_board_selection_max_camera_rmse_delta) {
            decision->reject_reason = "cam0_rmse_delta_gate";
          } else if (decision->cam1_rmse_delta >
                     options.pair_board_selection_max_camera_rmse_delta) {
            decision->reject_reason = "cam1_rmse_delta_gate";
          } else if (
              decision->baseline_rotation_delta_deg >
              options
                  .pair_board_selection_max_baseline_rotation_delta_deg) {
            decision->reject_reason = "baseline_rotation_delta_gate";
          } else if (
              decision->baseline_translation_delta_m >
              options.pair_board_selection_max_baseline_translation_delta_m) {
            decision->reject_reason = "baseline_translation_delta_gate";
          } else {
            decision->accepted = true;
          }
        } else {
          ++summary.batch_acceptance_attempted_count;
          const bool residual_finite =
              std::isfinite(trial_residual.total_stereo_rmse) &&
              std::isfinite(trial_residual.cam0_rmse) &&
              std::isfinite(trial_residual.cam1_rmse) &&
              std::isfinite(decision->total_rmse_delta) &&
              std::isfinite(decision->cam0_rmse_delta) &&
              std::isfinite(decision->cam1_rmse_delta);
          const bool baseline_finite =
              std::isfinite(decision->baseline_rotation_delta_deg) &&
              std::isfinite(decision->baseline_translation_delta_m);
          const bool objective_valid =
              std::isfinite(trial_ba_summary.objective_start) &&
              std::isfinite(trial_ba_summary.objective_final) &&
              !(trial_ba_summary.objective_final >
                10.0 * trial_ba_summary.objective_start);
          const bool baseline_stable =
              decision->baseline_rotation_delta_deg <=
                  options
                      .pair_board_selection_max_baseline_rotation_delta_deg &&
              decision->baseline_translation_delta_m <=
                  options.pair_board_selection_max_baseline_translation_delta_m;
          decision->hard_validity_pass =
              trial_residual.success && trial_residual.point_count > 0 &&
              residual_finite && baseline_finite &&
              !trial_ba_summary.linear_solver_failure && objective_valid &&
              baseline_stable;
          if (!decision->hard_validity_pass) {
            decision->reject_reason = "hard_validity_gate";
            ++summary.batch_acceptance_rejected_hard_validity_count;
          } else {
            const double max_total_delta =
                std::max(options.pair_board_selection_max_rmse_delta,
                         1e-12);
            const double max_camera_delta =
                std::max(options.pair_board_selection_max_camera_rmse_delta,
                         1e-12);
            decision->catastrophic_residual =
                decision->total_rmse_delta > 5.0 * max_total_delta ||
                decision->cam0_rmse_delta > 4.0 * max_camera_delta ||
                decision->cam1_rmse_delta > 4.0 * max_camera_delta;
            if (decision->catastrophic_residual) {
              decision->reject_reason = "batch_catastrophic_residual_gate";
              ++summary.batch_acceptance_rejected_catastrophic_residual_count;
            } else if (kalibr_information_gain_policy) {
              const Stage6IncrementalBatchCandidate batch_candidate =
                  MakeStage6IncrementalBatchCandidate(
                      decision->pair_cohesion_candidate
                          ? "pair_board_cohesion"
                          : "pair_board",
                      decision->pair_index, selected_pair_boards,
                      trial_pair_boards, false);
	              Stage6IncrementalBatchResult batch_result =
	                  pair_board_estimator.AddBatch(
	                      dataset, current_scene, trial_scene, current_residual,
	                      trial_residual, trial_ba_summary, batch_candidate);
              const bool coobs_pair_completion_candidate =
                  is_pair_completion_candidate();
	              TryAcceptByCoObsAwareScore(
	                  trial_ba_summary, options, batch_result.solution_valid,
	                  coobs_pair_completion_candidate, &batch_result, decision);
              CopyIncrementalBatchResultToDecision(
                  batch_result, batch_candidate.batch_type, decision);
              if (batch_result.batchAccepted) {
                decision->accepted = true;
                ++summary.batch_acceptance_accepted_count;
                if (!decision->legacy_rmse_pass) {
                  ++summary
                        .batch_acceptance_rescued_from_legacy_rmse_gate_count;
                }
              } else if (batch_result.reject_reason == "hard_validity_gate") {
                ++summary.batch_acceptance_rejected_hard_validity_count;
              } else {
                ++summary.batch_acceptance_rejected_score_count;
              }
            } else if (decision->legacy_rmse_pass) {
              decision->accepted = true;
              ++summary.batch_acceptance_accepted_count;
            } else {
              const double score_denominator =
                  std::max(options.pair_board_selection_min_candidate_score,
                           1.0);
              const double raw_score_term =
                  std::isfinite(decision->candidate_score)
                      ? decision->candidate_score / score_denominator
                      : 0.0;
              decision->score_term =
                  std::min(2.0, std::max(0.0, raw_score_term));
              decision->coverage_term = std::max(0.0, decision->coverage_gain);
              const auto shared_it =
                  dataset.pair_shared_board_ids.find(decision->pair_index);
              const int shared_board_capacity =
                  shared_it == dataset.pair_shared_board_ids.end()
                      ? 0
                      : static_cast<int>(shared_it->second.size());
              decision->pair_completion_bonus =
                  selected_for_pair > 0 &&
                          selected_for_pair < shared_board_capacity
                      ? 1.0
                      : 0.0;
              decision->new_board_bonus = selected_for_board == 0 ? 0.5 : 0.0;
              decision->cap_penalty =
                  (pair_cap_exceeded ? 0.5 : 0.0) +
                  (board_cap_exceeded ? 0.5 : 0.0);
              decision->information_gain_proxy =
                  decision->coverage_term + 0.5 * decision->score_term +
                  decision->pair_completion_bonus + decision->new_board_bonus;
              const double total_overage =
                  std::max(0.0,
                           decision->total_rmse_delta / max_total_delta -
                               1.0);
              const double cam0_overage =
                  std::max(0.0,
                           decision->cam0_rmse_delta / max_camera_delta -
                               1.0);
              const double cam1_overage =
                  std::max(0.0,
                           decision->cam1_rmse_delta / max_camera_delta -
                               1.0);
              decision->residual_overage_penalty =
                  total_overage + 0.5 * cam0_overage + 0.5 * cam1_overage;
              decision->batch_acceptance_score =
                  decision->information_gain_proxy -
                  decision->residual_overage_penalty - decision->cap_penalty;
              if (decision->information_gain_proxy >= 1.0 &&
                  decision->batch_acceptance_score >= 0.5) {
                decision->accepted = true;
                decision->accepted_by_batch_acceptance = true;
                ++summary.batch_acceptance_accepted_count;
                ++summary
                      .batch_acceptance_rescued_from_legacy_rmse_gate_count;
	              } else {
	                decision->reject_reason = "batch_acceptance_score_gate";
	                if (TryAcceptByCoObsAwareScore(
	                        trial_ba_summary, options,
	                        decision->hard_validity_pass,
	                        is_pair_completion_candidate(),
	                        decision)) {
	                  ++summary.batch_acceptance_accepted_count;
	                  ++summary
	                        .batch_acceptance_rescued_from_legacy_rmse_gate_count;
	                } else {
	                  ++summary.batch_acceptance_rejected_score_count;
	                }
	              }
            }
          }
        }

        if (decision->accepted) {
          selected_pair_boards.insert(key);
          if (commit_trial_state) {
            current_scene = trial_scene;
          }
          current_dataset = trial_dataset;
          current_selection = trial_selection;
          current_residual = trial_residual;
          return true;
        }
        return false;
      };

  int added_count = 0;
  for (StereoPairBoardTrialSelectionDecision decision : candidate_decisions) {
    if (pair_board_budget.mode == StereoCandidateBudgetMode::Fixed &&
        added_count >= pair_board_budget.traversal_limit) {
      break;
    }
    if (pair_board_budget.mode != StereoCandidateBudgetMode::Fixed &&
        summary.valid_candidate_traversed_count >=
            pair_board_budget.traversal_limit) {
      decision.reject_reason =
          pair_board_budget.mode == StereoCandidateBudgetMode::KalibrStyle
              ? "runtime_safety_ceiling"
              : "candidate_budget_limit";
      if (pair_board_budget.mode == StereoCandidateBudgetMode::KalibrStyle) {
        if (!summary.safety_ceiling_hit) {
          summary.safety_ceiling_hit = true;
          summary.warnings.push_back(
              "runtime_safety_ceiling_hit; result is runtime-capped");
        }
      }
      summary.decisions.push_back(decision);
      ++summary.rejected_count;
      break;
    }
    ++summary.valid_candidate_traversed_count;
    const bool accepted = try_add_candidate(&decision, true);
    if (decision.attempted) {
      ++summary.attempted_count;
    }
    if (accepted) {
      ++summary.accepted_count;
      ++added_count;
    } else {
      ++summary.rejected_count;
    }
    summary.decisions.push_back(decision);
  }

  summary.single_board_pair_count_before_rescue =
      CountPairsBelowSelectedBoardMinimum(
          selected_pair_boards,
          options.pair_cohesion_min_boards_per_pair);
  summary.pair_cohesion_under_target_pair_count_before_rescue =
      CountPairsBelowPairCohesionTarget(
          selected_pair_boards,
          dataset,
          options.pair_cohesion_min_boards_per_pair);

  if (options.single_board_pair_policy ==
      StereoSingleBoardPairPolicy::LowWeight) {
    summary.warnings.push_back(
        "single_board_pair_policy_low_weight_unsupported_without_ba_weight_change");
  }

  if (options.enable_pair_cohesion &&
      options.pair_cohesion_min_boards_per_pair > 1) {
    std::map<PairBoardKey, StereoPairBoardTrialSelectionDecision>
        decisions_by_key;
    for (const StereoPairBoardTrialSelectionDecision& decision :
         seed_decisions) {
      decisions_by_key[PairBoardKey(decision.pair_index, decision.board_id)] =
          decision;
    }
    for (const StereoPairBoardTrialSelectionDecision& decision :
         candidate_decisions) {
      decisions_by_key[PairBoardKey(decision.pair_index, decision.board_id)] =
          decision;
    }

    std::vector<int> rescue_pair_indices;
    const std::map<int, int> selected_board_counts =
        CountSelectedBoardsPerPair(selected_pair_boards);
    for (const auto& entry : selected_board_counts) {
      const auto shared_it = dataset.pair_shared_board_ids.find(entry.first);
      if (shared_it == dataset.pair_shared_board_ids.end()) {
        continue;
      }
      const int target_board_count =
          std::max(options.pair_cohesion_min_boards_per_pair,
                   static_cast<int>(shared_it->second.size()));
      if (entry.second < target_board_count) {
        rescue_pair_indices.push_back(entry.first);
      }
    }
    std::sort(rescue_pair_indices.begin(), rescue_pair_indices.end());

    bool stop_pair_cohesion_traversal = false;
    for (int pair_index : rescue_pair_indices) {
      if (stop_pair_cohesion_traversal) {
        break;
      }
      const auto shared_it = dataset.pair_shared_board_ids.find(pair_index);
      if (shared_it == dataset.pair_shared_board_ids.end()) {
        continue;
      }
      std::vector<StereoPairBoardTrialSelectionDecision> companions;
      for (int board_id : shared_it->second) {
        const PairBoardKey key(pair_index, board_id);
        if (selected_pair_boards.count(key) > 0) {
          continue;
        }
        auto decision_it = decisions_by_key.find(key);
        if (decision_it == decisions_by_key.end()) {
          continue;
        }
        StereoPairBoardTrialSelectionDecision decision = decision_it->second;
        decision.seed = false;
        decision.pair_cohesion_candidate = true;
        decision.attempted = false;
        decision.accepted = false;
        decision.reject_reason.clear();
        companions.push_back(decision);
      }
      summary.valid_candidate_count += static_cast<int>(companions.size());
      std::sort(companions.begin(), companions.end(),
                [](const StereoPairBoardTrialSelectionDecision& lhs,
                   const StereoPairBoardTrialSelectionDecision& rhs) {
                  const bool lhs_quality =
                      std::isfinite(lhs.cam0_outer_rmse) &&
                      std::isfinite(lhs.cam1_outer_rmse);
                  const bool rhs_quality =
                      std::isfinite(rhs.cam0_outer_rmse) &&
                      std::isfinite(rhs.cam1_outer_rmse);
                  if (lhs_quality != rhs_quality) {
                    return lhs_quality;
                  }
                  if (lhs.candidate_score != rhs.candidate_score) {
                    return lhs.candidate_score > rhs.candidate_score;
                  }
                  return lhs.board_id < rhs.board_id;
                });
      const int target_board_count =
          std::max(options.pair_cohesion_min_boards_per_pair,
                   static_cast<int>(shared_it->second.size()));
      summary.pair_cohesion_auto_target_board_count =
          std::max(summary.pair_cohesion_auto_target_board_count,
                   target_board_count);
      const int available_companion_count =
          static_cast<int>(companions.size());
      int max_to_attempt = available_companion_count;
      if (options.pair_cohesion_max_companions_per_pair > 0) {
        max_to_attempt =
            std::min(max_to_attempt,
                     options.pair_cohesion_max_companions_per_pair);
      }
      int attempted_for_pair = 0;
      for (StereoPairBoardTrialSelectionDecision decision : companions) {
        if (stop_pair_cohesion_traversal) {
          break;
        }
        if (attempted_for_pair >= max_to_attempt) {
          break;
        }
        ++summary.pair_cohesion_candidate_count;
      if (pair_board_budget.mode == StereoCandidateBudgetMode::KalibrStyle &&
          summary.valid_candidate_traversed_count >=
              pair_board_budget.traversal_limit) {
        decision.reject_reason = "runtime_safety_ceiling";
        if (!summary.safety_ceiling_hit) {
          summary.safety_ceiling_hit = true;
          summary.warnings.push_back(
              "runtime_safety_ceiling_hit; result is runtime-capped");
        }
        summary.decisions.push_back(decision);
          ++summary.pair_cohesion_rejected_count;
          stop_pair_cohesion_traversal = true;
          break;
        }
        if (pair_board_budget.mode == StereoCandidateBudgetMode::Adaptive &&
            summary.valid_candidate_traversed_count >=
                pair_board_budget.traversal_limit) {
          decision.reject_reason = "candidate_budget_limit";
          summary.decisions.push_back(decision);
          ++summary.pair_cohesion_rejected_count;
          stop_pair_cohesion_traversal = true;
          break;
        }
        ++summary.valid_candidate_traversed_count;
        const bool enforce_score_gate =
            !options.pair_cohesion_relax_score_gate;
        const bool accepted =
            try_add_candidate(&decision, enforce_score_gate);
        if (decision.attempted) {
          ++summary.pair_cohesion_attempted_count;
          ++attempted_for_pair;
        }
        if (accepted) {
          ++summary.pair_cohesion_accepted_count;
        } else {
          ++summary.pair_cohesion_rejected_count;
        }
        summary.decisions.push_back(decision);
        if (CountSelectedPairBoardForPair(selected_pair_boards, pair_index) >=
            target_board_count) {
          break;
        }
      }
    }
  }

  summary.single_board_pair_count_after_rescue =
      CountPairsBelowSelectedBoardMinimum(
          selected_pair_boards,
          options.pair_cohesion_min_boards_per_pair);
  summary.pair_cohesion_under_target_pair_count_after_rescue =
      CountPairsBelowPairCohesionTarget(
          selected_pair_boards,
          dataset,
          options.pair_cohesion_min_boards_per_pair);

  if (options.single_board_pair_policy == StereoSingleBoardPairPolicy::Drop) {
    DropPairsBelowSelectedBoardMinimum(
        &selected_pair_boards,
        options.pair_cohesion_min_boards_per_pair,
        &summary.dropped_single_board_pair_count);
    current_selection = BuildSelectionSummaryFromSelectedPairBoards(
        base_selection, dataset, selected_pair_boards);
    current_dataset = MakePairBoardMaskedDataset(dataset, selected_pair_boards);
    if (!selected_pair_boards.empty()) {
      StereoGlobalSparseBaSummary drop_ba_summary;
      if (RunGlobalSparseBa(current_dataset, trial_options,
                            current_selection, &drop_ba_summary,
                            &current_scene)) {
        current_residual = evaluator.Evaluate(
            current_dataset, current_scene,
            current_selection.selected_pair_indices,
            "stage6_pair_board_trial_drop_policy");
      } else {
        summary.warnings.push_back(
            drop_ba_summary.failure_reason.empty()
                ? "single_board_pair_policy_drop_refit_failed"
                : "single_board_pair_policy_drop_refit_failed:" +
                      drop_ba_summary.failure_reason);
      }
    }
  }

  if (ApplyAblationExcludedPairBoards(
          options.ablation_excluded_pair_boards,
          &selected_pair_boards,
          &summary.warnings) > 0) {
    RefitAfterPairBoardAblation(
        dataset,
        options,
        trial_options,
        base_selection,
        selected_pair_boards,
        "stage6_pair_board_trial_ablation",
        &current_scene,
        &current_residual,
        &summary.warnings);
  }

  summary.single_board_pair_count_after_policy =
      CountPairsBelowSelectedBoardMinimum(
          selected_pair_boards,
          options.pair_cohesion_min_boards_per_pair);

  summary.final_selected_pair_board_count =
      static_cast<int>(selected_pair_boards.size());
  summary.final_selected_rmse = current_residual.total_stereo_rmse;
  summary.selected_pair_board_keys = selected_pair_boards;
  summary.success = !selected_pair_boards.empty();
  *scene_state = current_scene;
  *final_selection =
      BuildSelectionSummaryFromSelectedPairBoards(
          base_selection, dataset, selected_pair_boards);
  return summary;
}

StereoPairBoardTrialSelectionDecision MakePairBoardDecision(
    const StereoMeasurementDataset& dataset,
    const StereoSceneState& scene_state,
    const StereoExtrinsicSolverOptions& options,
    int pair_index,
    int board_id) {
  StereoPairBoardTrialSelectionDecision decision;
  decision.pair_index = pair_index;
  decision.board_id = board_id;
  decision.pairboard_selection_mode = options.pairboard_selection_mode;
  decision.shared_board = true;
  decision.cam0_outer_point_count =
      CountOuterObservationsForCamera(dataset, pair_index, 0, board_id);
  decision.cam1_outer_point_count =
      CountOuterObservationsForCamera(dataset, pair_index, 1, board_id);
  decision.shared_point_count =
      CountSharedObservationsForPairBoard(dataset, pair_index, board_id);
  const StereoSharedBoardQuality quality =
      EvaluateSharedBoardQuality(dataset, scene_state, options, pair_index, board_id);
  decision.cam0_outer_rmse = quality.cam0_outer_rmse;
  decision.cam1_outer_rmse = quality.cam1_outer_rmse;
  decision.candidate_score = ComputePairBoardTrialCandidateScore(decision);
  return decision;
}

bool EvaluateKalibrStylePairBoardBatchAcceptance(
    const StereoMeasurementDataset& dataset,
    const StereoExtrinsicSolverOptions& options,
    const std::set<PairBoardKey>& selected_pair_boards,
    const StereoResidualSummary& current_residual,
    const StereoResidualSummary& trial_residual,
    const StereoGlobalSparseBaSummary& trial_ba_summary,
    const Eigen::Isometry3d& baseline_before,
    const StereoSceneState& trial_scene,
    StereoPairBoardTrialSelectionDecision* decision,
    StereoPairBoardTrialSelectionSummary* summary) {
  if (decision == nullptr || summary == nullptr) {
    return false;
  }
  ++summary->batch_acceptance_attempted_count;
  const bool bootstrap_backend = selected_pair_boards.empty();
  const Eigen::Isometry3d baseline_after = ToIsometry3d(trial_scene.T_cam1_cam0);
  decision->initial_total_rmse = bootstrap_backend
                                      ? trial_residual.total_stereo_rmse
                                      : current_residual.total_stereo_rmse;
  decision->trial_total_rmse = trial_residual.total_stereo_rmse;
  decision->total_rmse_delta =
      bootstrap_backend
          ? 0.0
          : trial_residual.total_stereo_rmse - current_residual.total_stereo_rmse;
  decision->cam0_rmse_delta =
      bootstrap_backend ? 0.0
                        : trial_residual.cam0_rmse - current_residual.cam0_rmse;
  decision->cam1_rmse_delta =
      bootstrap_backend ? 0.0
                        : trial_residual.cam1_rmse - current_residual.cam1_rmse;
  decision->baseline_rotation_delta_deg =
      RotationDistanceRadians(baseline_before, baseline_after) * 180.0 / M_PI;
  decision->baseline_translation_delta_m =
      (baseline_before.translation() - baseline_after.translation()).norm();
  decision->legacy_rmse_pass =
      decision->total_rmse_delta <= options.pair_board_selection_max_rmse_delta &&
      decision->cam0_rmse_delta <=
          options.pair_board_selection_max_camera_rmse_delta &&
      decision->cam1_rmse_delta <=
          options.pair_board_selection_max_camera_rmse_delta;

  const bool residual_finite =
      trial_residual.success && trial_residual.point_count > 0 &&
      std::isfinite(trial_residual.total_stereo_rmse) &&
      std::isfinite(trial_residual.cam0_rmse) &&
      std::isfinite(trial_residual.cam1_rmse) &&
      std::isfinite(decision->total_rmse_delta) &&
      std::isfinite(decision->cam0_rmse_delta) &&
      std::isfinite(decision->cam1_rmse_delta);
  const bool baseline_finite =
      std::isfinite(decision->baseline_rotation_delta_deg) &&
      std::isfinite(decision->baseline_translation_delta_m);
  const bool objective_valid =
      std::isfinite(trial_ba_summary.objective_start) &&
      std::isfinite(trial_ba_summary.objective_final) &&
      !(trial_ba_summary.objective_final >
        10.0 * trial_ba_summary.objective_start);
  const bool baseline_stable =
      decision->baseline_rotation_delta_deg <=
          options.pair_board_selection_max_baseline_rotation_delta_deg &&
      decision->baseline_translation_delta_m <=
          options.pair_board_selection_max_baseline_translation_delta_m;
  decision->hard_validity_pass =
      residual_finite && baseline_finite &&
      !trial_ba_summary.linear_solver_failure && objective_valid &&
      baseline_stable;
  if (!decision->hard_validity_pass) {
    decision->reject_reason = "hard_validity_gate";
    ++summary->batch_acceptance_rejected_hard_validity_count;
    return false;
  }
  if (decision->legacy_rmse_pass) {
    decision->accepted = true;
    ++summary->batch_acceptance_accepted_count;
    return true;
  }

  const double max_total_delta =
      std::max(options.pair_board_selection_max_rmse_delta, 1e-12);
  const double max_camera_delta =
      std::max(options.pair_board_selection_max_camera_rmse_delta, 1e-12);
  const int selected_for_pair =
      CountSelectedPairBoardForPair(selected_pair_boards, decision->pair_index);
  const int selected_for_board =
      CountSelectedPairBoardForBoard(selected_pair_boards, decision->board_id);
  const bool pair_cap_exceeded =
      options.pair_board_selection_max_accepted_per_pair > 0 &&
      selected_for_pair >= options.pair_board_selection_max_accepted_per_pair;
  const bool board_cap_exceeded =
      options.pair_board_selection_max_accepted_per_board > 0 &&
      selected_for_board >= options.pair_board_selection_max_accepted_per_board;
  const double score_denominator =
      std::max(options.pair_board_selection_min_candidate_score, 1.0);
  const double raw_score_term =
      std::isfinite(decision->candidate_score)
          ? decision->candidate_score / score_denominator
          : 0.0;
  decision->score_term = std::min(2.0, std::max(0.0, raw_score_term));
  decision->coverage_term = std::max(0.0, decision->coverage_gain);
  const auto shared_it = dataset.pair_shared_board_ids.find(decision->pair_index);
  const int shared_board_capacity =
      shared_it == dataset.pair_shared_board_ids.end()
          ? 0
          : static_cast<int>(shared_it->second.size());
  decision->pair_completion_bonus =
      selected_for_pair > 0 && selected_for_pair < shared_board_capacity ? 1.0
                                                                         : 0.0;
  decision->new_board_bonus = selected_for_board == 0 ? 0.5 : 0.0;
  decision->cap_penalty =
      (pair_cap_exceeded ? 0.5 : 0.0) + (board_cap_exceeded ? 0.5 : 0.0);
  decision->information_gain_proxy =
      decision->coverage_term + 0.5 * decision->score_term +
      decision->pair_completion_bonus + decision->new_board_bonus;
  const bool committing_absolute_residual_health =
      options.enable_committing_pair_batch_selection &&
      trial_residual.total_stereo_rmse <= 2.0 &&
      trial_residual.cam0_rmse <= 2.5 &&
      trial_residual.cam1_rmse <= 2.5 &&
      decision->information_gain_proxy >= 1.0;
  if (committing_absolute_residual_health) {
    decision->accepted = true;
    decision->accepted_by_batch_acceptance = true;
    ++summary->batch_acceptance_accepted_count;
    ++summary->batch_acceptance_rescued_from_legacy_rmse_gate_count;
    return true;
  }
  decision->catastrophic_residual =
      decision->total_rmse_delta > 5.0 * max_total_delta ||
      decision->cam0_rmse_delta > 4.0 * max_camera_delta ||
      decision->cam1_rmse_delta > 4.0 * max_camera_delta;
  if (decision->catastrophic_residual) {
    decision->reject_reason = "batch_catastrophic_residual_gate";
    ++summary->batch_acceptance_rejected_catastrophic_residual_count;
    return false;
  }
  decision->residual_overage_penalty =
      std::max(0.0, decision->total_rmse_delta / max_total_delta - 1.0) +
      0.5 * std::max(0.0, decision->cam0_rmse_delta / max_camera_delta - 1.0) +
      0.5 * std::max(0.0, decision->cam1_rmse_delta / max_camera_delta - 1.0);
  decision->batch_acceptance_score =
      decision->information_gain_proxy - decision->residual_overage_penalty -
      decision->cap_penalty;
  if (decision->information_gain_proxy >= 1.0 &&
      decision->batch_acceptance_score >= 0.5) {
    decision->accepted = true;
    decision->accepted_by_batch_acceptance = true;
    ++summary->batch_acceptance_accepted_count;
    ++summary->batch_acceptance_rescued_from_legacy_rmse_gate_count;
    return true;
  }
  decision->reject_reason = "batch_acceptance_score_gate";
  ++summary->batch_acceptance_rejected_score_count;
  return false;
}

void AccumulatePairBatchDiagnostics(
    StereoPairTrialSelectionDecision* pair_decision,
    const std::vector<StereoPairBoardTrialSelectionDecision>& board_decisions) {
  if (pair_decision == nullptr || board_decisions.empty()) {
    return;
  }
  pair_decision->candidate_score = 0.0;
  pair_decision->coverage_gain = 0.0;
  pair_decision->shared_internal_point_count = 0;
  pair_decision->shared_outer_point_count = 0;
  for (const StereoPairBoardTrialSelectionDecision& board_decision :
       board_decisions) {
    pair_decision->candidate_score += board_decision.candidate_score;
    pair_decision->coverage_gain += board_decision.coverage_gain;
    pair_decision->shared_internal_point_count +=
        std::max(0, board_decision.shared_point_count -
                        board_decision.cam0_outer_point_count -
                        board_decision.cam1_outer_point_count);
    pair_decision->shared_outer_point_count +=
        board_decision.cam0_outer_point_count +
        board_decision.cam1_outer_point_count;
  }
}

void CopyPersistentReturnValueToPairDecision(
    const aslam::calibration::IncrementalEstimator::ReturnValue& ret,
    int rank_before,
    int normalization_count,
    const std::string& batch_type,
    bool solution_valid,
    const std::string& committed_or_rollback,
    StereoPairTrialSelectionDecision* decision) {
  if (decision == nullptr) {
    return;
  }
  decision->incremental_estimator_enabled = true;
  decision->persistent_incremental_estimator_used = true;
  decision->candidate_batch_type = batch_type;
  decision->batchAccepted = ret.batchAccepted && solution_valid;
  decision->solution_valid = solution_valid;
  decision->optimization_success = ret.numIterations > 0 ||
                                   (std::isfinite(ret.JStart) &&
                                    std::isfinite(ret.JFinal));
  decision->num_iterations = static_cast<int>(ret.numIterations);
  decision->objective_before = ret.JStart;
  decision->objective_after = ret.JFinal;
  decision->marginal_information_gain_proxy = ret.informationGain;
  decision->normalized_information_gain =
      normalization_count > 0
          ? ret.informationGain / static_cast<double>(normalization_count)
          : ret.informationGain;
  decision->information_gain_normalization_count =
      std::max(1, normalization_count);
  decision->rank_before = rank_before;
  decision->rank_after = static_cast<int>(ret.rankTheta);
  decision->rank_psi_after = static_cast<int>(ret.rankPsi);
  decision->rank_psi_deficiency_after =
      static_cast<int>(ret.rankPsiDeficiency);
  decision->rank_theta_deficiency_after =
      static_cast<int>(ret.rankThetaDeficiency);
  decision->rank_proxy_increases =
      rank_before >= 0 && static_cast<int>(ret.rankTheta) > rank_before;
  decision->svd_tolerance = ret.svdTolerance;
  decision->qr_tolerance = ret.qrTolerance;
  decision->elapsed_time_seconds = ret.elapsedTime;
  decision->committed_or_rollback = committed_or_rollback;
}

void CopyPersistentReturnValueToBoardDecision(
    const aslam::calibration::IncrementalEstimator::ReturnValue& ret,
    int rank_before,
    int normalization_count,
    const std::string& batch_type,
    bool solution_valid,
    const std::string& committed_or_rollback,
    StereoPairBoardTrialSelectionDecision* decision) {
  if (decision == nullptr) {
    return;
  }
  decision->incremental_estimator_enabled = true;
  decision->persistent_incremental_estimator_used = true;
  decision->candidate_batch_type = batch_type;
  decision->batchAccepted = ret.batchAccepted && solution_valid;
  decision->solution_valid = solution_valid;
  decision->optimization_success = ret.numIterations > 0 ||
                                   (std::isfinite(ret.JStart) &&
                                    std::isfinite(ret.JFinal));
  decision->num_iterations = static_cast<int>(ret.numIterations);
  decision->objective_before = ret.JStart;
  decision->objective_after = ret.JFinal;
  decision->marginal_information_gain_proxy = ret.informationGain;
  decision->normalized_information_gain =
      normalization_count > 0
          ? ret.informationGain / static_cast<double>(normalization_count)
          : ret.informationGain;
  decision->information_gain_normalization_count =
      std::max(1, normalization_count);
  decision->rank_before = rank_before;
  decision->rank_after = static_cast<int>(ret.rankTheta);
  decision->rank_psi_after = static_cast<int>(ret.rankPsi);
  decision->rank_psi_deficiency_after =
      static_cast<int>(ret.rankPsiDeficiency);
  decision->rank_theta_deficiency_after =
      static_cast<int>(ret.rankThetaDeficiency);
  decision->rank_proxy_increases =
      rank_before >= 0 && static_cast<int>(ret.rankTheta) > rank_before;
  decision->svd_tolerance = ret.svdTolerance;
  decision->qr_tolerance = ret.qrTolerance;
  decision->elapsed_time_seconds = ret.elapsedTime;
  decision->committed_or_rollback = committed_or_rollback;
}

template <typename GeometryT0, typename GeometryT1>
StereoPairBoardTrialSelectionSummary
RunPersistentIncrementalPairCohesiveSelectionTyped(
    const StereoMeasurementDataset& dataset,
    const StereoExtrinsicSolverOptions& options,
    const StereoPairSelectionSummary& base_selection,
    StereoPairTrialSelectionSummary* pair_summary,
    StereoPairSelectionSummary* final_selection,
    StereoSceneState* scene_state) {
  const Clock::time_point start_time = Clock::now();
  StereoPairBoardTrialSelectionSummary board_summary;
  board_summary.enabled = true;
  board_summary.success = false;
  board_summary.pairboard_selection_mode = options.pairboard_selection_mode;
  board_summary.incremental_estimator_enabled = true;
  board_summary.persistent_incremental_estimator_used = true;
  board_summary.marginal_information_gain_proxy_enabled = false;
  board_summary.rmse_delta_diagnostics_only = true;
  board_summary.incremental_mi_tol = options.incremental_mi_tol;
  board_summary.incremental_rank_threshold = options.incremental_rank_threshold;
  board_summary.incremental_info_block = ToString(options.incremental_info_block);
  const StereoIntrinsicsPolicyDecision intrinsics_policy =
      EvaluateStereoIntrinsicsPolicy(dataset, options);
  const bool has_separate_distortion_dv =
      scene_state != nullptr &&
      (!scene_state->cam0.distortion_coeffs.empty() ||
       !scene_state->cam1.distortion_coeffs.empty());
  board_summary.persistent_incremental_information_group =
      intrinsics_policy.projection_active
          ? "stereo_extrinsic_plus_camera_projection"
          : "stereo_extrinsic";
  board_summary.persistent_pose_structure =
      options.persistent_pose_structure ==
              StereoPersistentPoseStructure::IndependentPairBoard
          ? "independent_pair_board"
          : "shared_frame_layout";
  board_summary.requested_intrinsics_mode =
      StereoIntrinsicsModeToString(intrinsics_policy.requested_mode);
  board_summary.effective_intrinsics_mode =
      StereoIntrinsicsModeToString(intrinsics_policy.effective_mode);
  board_summary.projection_intrinsics_active =
      intrinsics_policy.projection_active;
  board_summary.projection_prior_enabled =
      intrinsics_policy.projection_prior_enabled;
  board_summary.distortion_intrinsics_active =
      intrinsics_policy.distortion_active && has_separate_distortion_dv;
  board_summary.distortion_prior_enabled =
      intrinsics_policy.distortion_prior_enabled && has_separate_distortion_dv;
  board_summary.projection_release_reason = intrinsics_policy.reason;
  board_summary.projection_policy_training_pair_count =
      intrinsics_policy.training_pair_count;
  board_summary.projection_policy_shared_pair_board_count =
      intrinsics_policy.shared_pair_board_count;
  board_summary.projection_policy_distinct_board_count =
      intrinsics_policy.distinct_board_count;
  board_summary.projection_policy_observation_point_count =
      intrinsics_policy.observation_point_count;
  board_summary.projection_prior_shape_sigma =
      options.persistent_incremental_projection_prior_shape_sigma;
  board_summary.projection_prior_focal_relative_sigma =
      options.persistent_incremental_projection_prior_focal_relative_sigma;
  board_summary.projection_prior_principal_sigma_px =
      options.persistent_incremental_projection_prior_principal_sigma_px;
  board_summary.distortion_prior_sigma =
      options.persistent_incremental_distortion_prior_sigma;
  if (pair_summary != nullptr) {
    *pair_summary = StereoPairTrialSelectionSummary();
    pair_summary->enabled = true;
    pair_summary->requested_seed_count =
        options.persistent_incremental_seed_pair_count;
    pair_summary->warnings.push_back(
        "persistent_incremental_pair_cohesive_selection");
  }
  if (final_selection == nullptr || scene_state == nullptr) {
    board_summary.failure_reason = "missing_output_pointer";
    return board_summary;
  }
  if (options.rig_param_mode != StereoRigParamMode::Cam0Reference) {
    board_summary.failure_reason =
        "persistent_incremental_first_version_requires_cam0_reference";
    return board_summary;
  }
  if (options.final_ba_residual_mode != options.selection_ba_residual_mode) {
    board_summary.failure_reason =
        "persistent_incremental_requires_matching_final_and_selection_residual_modes";
    return board_summary;
  }
  Stage6PersistentResidualMetric residual_metric =
      Stage6PersistentResidualMetric::Pixel;
  try {
    residual_metric =
        PersistentResidualMetricForMode(options.selection_ba_residual_mode);
  } catch (const std::exception& exception) {
    board_summary.failure_reason = exception.what();
    return board_summary;
  }
  board_summary.persistent_incremental_residual_metric_name =
      ToString(residual_metric);
  if (UsesTangentPlaneAngularResidual(residual_metric) &&
      (scene_state->cam0.camera_model_family != "ds-none" ||
       scene_state->cam1.camera_model_family != "ds-none")) {
    board_summary.failure_reason =
        "persistent_incremental_tangent_plane_requires_ds_none_cam0_cam1";
    return board_summary;
  }

  std::vector<const StereoPairSelectionRow*> ranked_rows;
  for (const StereoPairSelectionRow& row : base_selection.rows) {
    if (!row.eligible || row.shared_board_count <= 0) {
      continue;
    }
    if (dataset.pair_shared_board_ids.count(row.pair_index) == 0) {
      continue;
    }
    ranked_rows.push_back(&row);
  }
  std::sort(ranked_rows.begin(), ranked_rows.end(),
            [](const StereoPairSelectionRow* lhs,
               const StereoPairSelectionRow* rhs) {
              return IsSelectionRowBetter(*lhs, *rhs);
            });
  board_summary.candidate_count = static_cast<int>(ranked_rows.size());
  board_summary.valid_candidate_count = board_summary.candidate_count;
  board_summary.budget_mode = options.pair_selection_budget_mode;
  board_summary.runtime_safety_ceiling =
      options.pair_selection_runtime_safety_ceiling;
  board_summary.max_candidate_additions_effective =
      options.pair_selection_budget_mode == StereoCandidateBudgetMode::KalibrStyle
          ? "adaptive_incremental_all_until_safety_ceiling"
          : std::to_string(options.pair_selection_max_candidate_additions);
  if (pair_summary != nullptr) {
    pair_summary->candidate_count = board_summary.candidate_count;
    pair_summary->valid_candidate_count = board_summary.valid_candidate_count;
    pair_summary->budget_mode = board_summary.budget_mode;
    pair_summary->runtime_safety_ceiling = board_summary.runtime_safety_ceiling;
    pair_summary->max_candidate_additions_effective =
        board_summary.max_candidate_additions_effective;
  }
  if (ranked_rows.empty()) {
    board_summary.failure_reason =
        "no_valid_persistent_incremental_pair_candidates";
    return board_summary;
  }

  aslam::calibration::IncrementalEstimator::Options inc_options;
  inc_options.infoGainDelta = options.incremental_mi_tol;
  inc_options.checkValidity = false;
  inc_options.verbose = false;
  aslam::calibration::LinearSolverOptions solver_options;
  solver_options.columnScaling = true;
  aslam::backend::Optimizer2Options optimizer_options;
  optimizer_options.maxIterations =
      std::max(1, options.persistent_incremental_max_iterations);
  optimizer_options.convergenceDeltaJ =
      options.persistent_incremental_convergence_delta_j;
  optimizer_options.convergenceDeltaX =
      options.persistent_incremental_convergence_delta_x;
  optimizer_options.verbose = false;
  optimizer_options.nThreads = 4;

  Stage6PersistentStereoProblemBuilder<GeometryT0, GeometryT1> builder(
      dataset, options, *scene_state);
  aslam::calibration::IncrementalEstimator estimator(
      kStage6StereoExtrinsicInformationGroupId, inc_options, solver_options,
      optimizer_options);

  const StereoCandidateTraversalBudget pair_budget =
      ComputeStereoCandidateTraversalBudget(
          options.pair_selection_budget_mode,
          static_cast<int>(ranked_rows.size()),
          options.pair_selection_max_candidate_additions,
          options.pair_selection_adaptive_budget_ratio,
          options.pair_selection_adaptive_budget_min,
          options.pair_selection_adaptive_budget_max,
          options.pair_selection_runtime_safety_ceiling);
  board_summary.budget_mode = pair_budget.mode;
  board_summary.runtime_safety_ceiling = pair_budget.runtime_safety_ceiling;
  board_summary.max_candidate_additions_effective =
      pair_budget.max_candidate_additions_effective;
  if (pair_summary != nullptr) {
    pair_summary->budget_mode = pair_budget.mode;
    pair_summary->runtime_safety_ceiling = pair_budget.runtime_safety_ceiling;
    pair_summary->max_candidate_additions_effective =
        pair_budget.max_candidate_additions_effective;
  }

  std::set<PairBoardKey> selected_pair_boards;
  std::set<PairBoardKey> seed_pair_boards;
  std::set<int> covered_boards;
  int current_rank = -1;
  const int seed_target =
      std::min<int>(std::max(1, options.persistent_incremental_seed_pair_count),
                    static_cast<int>(ranked_rows.size()));
  auto build_pair_cohesive_candidate =
      [&](const StereoPairSelectionRow& row,
          const std::set<PairBoardKey>& already_selected,
          std::vector<StereoPairBoardTrialSelectionDecision>* board_decisions,
          std::set<PairBoardKey>* batch_keys) {
        if (board_decisions == nullptr || batch_keys == nullptr) {
          return;
        }
        const auto shared_it =
            dataset.pair_shared_board_ids.find(row.pair_index);
        if (shared_it == dataset.pair_shared_board_ids.end()) {
          return;
        }
        for (int board_id : shared_it->second) {
          StereoPairBoardTrialSelectionDecision board_decision =
              MakePairBoardDecision(dataset, *scene_state, options,
                                    row.pair_index, board_id);
          board_decision.pair_cohesion_candidate = true;
          board_decision.selected_pair_board_count_before =
              static_cast<int>(already_selected.size());
          board_decision.selected_pair_count_before = static_cast<int>(
              PairIndicesFromPairBoards(already_selected).size());
          board_decision.selected_board_count_before = static_cast<int>(
              BoardIdsFromPairBoards(already_selected).size());
          board_decision.coverage_gain =
              ComputePairBoardCoverageGain(already_selected, row.pair_index,
                                           board_id);
          board_decisions->push_back(board_decision);
          batch_keys->insert(PairBoardKey(row.pair_index, board_id));
        }
      };

  int traversed = seed_target;

  std::vector<const StereoPairSelectionRow*> seed_rows;
  std::set<int> seed_pair_indices;
  std::vector<StereoPairTrialSelectionDecision> seed_pair_decisions;
  std::vector<StereoPairBoardTrialSelectionDecision> seed_board_decisions;
  for (int i = 0; i < seed_target; ++i) {
    const StereoPairSelectionRow* row = ranked_rows[i];
    if (row == nullptr) {
      continue;
    }
    seed_rows.push_back(row);
    seed_pair_indices.insert(row->pair_index);
    std::vector<StereoPairBoardTrialSelectionDecision> row_board_decisions;
    std::set<PairBoardKey> row_keys;
    build_pair_cohesive_candidate(*row, seed_pair_boards,
                                  &row_board_decisions, &row_keys);
    if (row_keys.empty()) {
      continue;
    }
    StereoPairTrialSelectionDecision pair_decision =
        MakeTrialDecisionFromRow(dataset, *row, covered_boards);
    pair_decision.attempted = true;
    pair_decision.incremental_estimator_enabled = true;
    pair_decision.persistent_incremental_estimator_used = true;
    pair_decision.candidate_batch_type =
        "persistent_pair_cohesive_seed_batch";
    pair_decision.force = true;
    pair_decision.seed = true;
    AccumulatePairBatchDiagnostics(&pair_decision, row_board_decisions);
    seed_pair_decisions.push_back(pair_decision);
    seed_board_decisions.insert(seed_board_decisions.end(),
                                row_board_decisions.begin(),
                                row_board_decisions.end());
    seed_pair_boards.insert(row_keys.begin(), row_keys.end());
  }
  board_summary.valid_candidate_traversed_count = traversed;
  if (pair_summary != nullptr) {
    pair_summary->valid_candidate_traversed_count = traversed;
  }
  if (seed_pair_boards.empty()) {
    board_summary.failure_reason = "empty_persistent_incremental_seed_batch";
    return board_summary;
  }

  typename Stage6PersistentStereoProblemBuilder<GeometryT0, GeometryT1>::
      StateSnapshot seed_state = builder.CaptureState();
  boost::shared_ptr<CalibrationBatch> seed_batch =
      builder.BuildBatch(seed_pair_boards, true, &seed_state, true);
  aslam::calibration::IncrementalEstimator::ReturnValue seed_ret{};
  bool seed_add_exception = false;
  std::string seed_invalid_reason;
  try {
    seed_ret = estimator.addBatch(seed_batch, true);
  } catch (const std::exception& exception) {
    seed_add_exception = true;
    seed_invalid_reason =
        std::string("incremental_seed_batch_exception: ") + exception.what();
  }
  bool seed_state_valid = false;
  if (!seed_add_exception) {
    seed_state_valid = builder.CurrentStateFinite(&seed_invalid_reason);
  }
  const bool seed_objective_finite =
      !seed_add_exception && std::isfinite(seed_ret.JStart) &&
      std::isfinite(seed_ret.JFinal);
  const bool seed_objective_decreased =
      seed_objective_finite && seed_ret.JFinal <= seed_ret.JStart;
  const bool seed_accepted = !seed_add_exception && seed_ret.batchAccepted &&
                             seed_state_valid && seed_objective_decreased;
  ++board_summary.batch_acceptance_attempted_count;
  board_summary.attempted_count += static_cast<int>(seed_rows.size());
  if (pair_summary != nullptr) {
    pair_summary->attempted_count += static_cast<int>(seed_rows.size());
  }
  const int seed_normalization_count =
      std::max(1, static_cast<int>(seed_pair_boards.size()));
  if (!seed_accepted) {
    ++board_summary.batch_acceptance_rejected_score_count;
    board_summary.rejected_count += static_cast<int>(seed_rows.size());
    if (pair_summary != nullptr) {
      pair_summary->rejected_count += static_cast<int>(seed_rows.size());
    }
    if (!seed_add_exception && seed_ret.batchAccepted) {
      estimator.rejectBatch(seed_batch);
    }
    builder.RestoreState(seed_state);
    if (seed_invalid_reason.empty()) {
      seed_invalid_reason =
          seed_ret.batchAccepted ? "persistent_seed_validity_gate"
                                 : "persistent_seed_estimator_rejected_batch";
    }
    for (StereoPairTrialSelectionDecision pair_decision :
         seed_pair_decisions) {
      pair_decision.accepted = false;
      pair_decision.batchAccepted = false;
      pair_decision.reject_reason = seed_invalid_reason;
      CopyPersistentReturnValueToPairDecision(
          seed_ret, current_rank, seed_normalization_count,
          "persistent_pair_cohesive_seed_batch", false, "rollback",
          &pair_decision);
      if (pair_summary != nullptr) {
        pair_summary->decisions.push_back(pair_decision);
      }
    }
    for (StereoPairBoardTrialSelectionDecision board_decision :
         seed_board_decisions) {
      board_decision.attempted = true;
      board_decision.accepted = false;
      board_decision.batchAccepted = false;
      board_decision.force = true;
      board_decision.seed = true;
      board_decision.reject_reason = seed_invalid_reason;
      CopyPersistentReturnValueToBoardDecision(
          seed_ret, current_rank, seed_normalization_count,
          "persistent_pair_cohesive_seed_batch", false, "rollback",
          &board_decision);
      board_summary.decisions.push_back(board_decision);
    }
    board_summary.failure_reason = seed_invalid_reason;
    return board_summary;
  }

  ++board_summary.batch_acceptance_accepted_count;
  board_summary.accepted_count += static_cast<int>(seed_rows.size());
  board_summary.seed_count = static_cast<int>(seed_rows.size());
  board_summary.persistent_incremental_seed_batch_count = 1;
  board_summary.persistent_incremental_seed_pair_count =
      static_cast<int>(seed_pair_indices.size());
  board_summary.persistent_incremental_seed_pair_board_count =
      static_cast<int>(seed_pair_boards.size());
  board_summary.persistent_incremental_seed_point_count =
      CountStereoObservationPointsForKeys(dataset, seed_pair_boards);
  board_summary.persistent_incremental_seed_rank_theta =
      static_cast<int>(seed_ret.rankTheta);
  board_summary.persistent_incremental_seed_information_gain =
      seed_ret.informationGain;
  if (pair_summary != nullptr) {
    pair_summary->accepted_count += static_cast<int>(seed_rows.size());
    pair_summary->seed_count = static_cast<int>(seed_rows.size());
  }
  selected_pair_boards.insert(seed_pair_boards.begin(), seed_pair_boards.end());
  covered_boards = BoardIdsFromPairBoards(selected_pair_boards);
  current_rank = static_cast<int>(seed_ret.rankTheta);
  Stage6PersistentResidualStats seed_stats =
      builder.EvaluateAccepted(selected_pair_boards, residual_metric);
  board_summary.initial_seed_rmse = seed_stats.Rmse();
  board_summary.final_selected_rmse = seed_stats.Rmse();
  if (pair_summary != nullptr) {
    pair_summary->initial_seed_rmse = seed_stats.Rmse();
    pair_summary->final_selected_rmse = seed_stats.Rmse();
  }
  for (StereoPairTrialSelectionDecision pair_decision : seed_pair_decisions) {
    pair_decision.accepted = true;
    pair_decision.batchAccepted = true;
    pair_decision.accept_reason = "forced_seed_batch";
    pair_decision.trial_total_rmse = seed_stats.Rmse();
    pair_decision.initial_total_rmse = seed_stats.Rmse();
    pair_decision.total_rmse_delta = 0.0;
    pair_decision.cam0_rmse_delta = 0.0;
    pair_decision.cam1_rmse_delta = 0.0;
    CopyPersistentReturnValueToPairDecision(
        seed_ret, -1, seed_normalization_count,
        "persistent_pair_cohesive_seed_batch", true, "committed",
        &pair_decision);
    if (pair_summary != nullptr) {
      pair_summary->decisions.push_back(pair_decision);
    }
  }
  for (StereoPairBoardTrialSelectionDecision board_decision :
       seed_board_decisions) {
    board_decision.attempted = true;
    board_decision.accepted = true;
    board_decision.batchAccepted = true;
    board_decision.force = true;
    board_decision.seed = true;
    board_decision.accept_reason = "forced_seed_batch";
    board_decision.trial_total_rmse = seed_stats.Rmse();
    board_decision.initial_total_rmse = seed_stats.Rmse();
    board_decision.total_rmse_delta = 0.0;
    board_decision.cam0_rmse_delta = 0.0;
    board_decision.cam1_rmse_delta = 0.0;
    CopyPersistentReturnValueToBoardDecision(
        seed_ret, -1, seed_normalization_count,
        "persistent_pair_cohesive_seed_batch", true, "committed",
        &board_decision);
    board_summary.decisions.push_back(board_decision);
  }

  for (const StereoPairSelectionRow* row : ranked_rows) {
    if (row == nullptr) {
      continue;
    }
    if (seed_pair_indices.count(row->pair_index) > 0) {
      continue;
    }
    if (pair_budget.traversal_limit != std::numeric_limits<int>::max() &&
        traversed >= pair_budget.traversal_limit) {
      board_summary.safety_ceiling_hit =
          pair_budget.runtime_safety_ceiling > 0 &&
          traversed >= pair_budget.runtime_safety_ceiling;
      break;
    }
    ++traversed;
    board_summary.valid_candidate_traversed_count = traversed;
    if (pair_summary != nullptr) {
      pair_summary->valid_candidate_traversed_count = traversed;
    }

    std::vector<StereoPairBoardTrialSelectionDecision> board_decisions;
    std::set<PairBoardKey> batch_keys;
    build_pair_cohesive_candidate(*row, selected_pair_boards,
                                  &board_decisions, &batch_keys);
    if (batch_keys.empty()) {
      continue;
    }
    StereoPairTrialSelectionDecision pair_decision =
        MakeTrialDecisionFromRow(dataset, *row, covered_boards);
    pair_decision.attempted = true;
    pair_decision.incremental_estimator_enabled = true;
    pair_decision.persistent_incremental_estimator_used = true;
    pair_decision.candidate_batch_type = "persistent_pair_cohesive";
    AccumulatePairBatchDiagnostics(&pair_decision, board_decisions);

    const bool force = false;
    const int normalization_count =
        std::max(1, static_cast<int>(batch_keys.size()));
    typename Stage6PersistentStereoProblemBuilder<GeometryT0, GeometryT1>::
        StateSnapshot state = builder.CaptureState();
    const Stage6PersistentResidualStats before_stats =
        builder.EvaluateAccepted(selected_pair_boards, residual_metric);
    boost::shared_ptr<CalibrationBatch> batch =
        builder.BuildBatch(batch_keys, true, &state, false);
    aslam::calibration::IncrementalEstimator::ReturnValue ret{};
    bool add_exception = false;
    std::string state_invalid_reason;
    try {
      ret = estimator.addBatch(batch, force);
    } catch (const std::exception& exception) {
      add_exception = true;
      state_invalid_reason =
          std::string("incremental_add_batch_exception: ") + exception.what();
    }
    bool state_valid = false;
    if (!add_exception) {
      state_valid = builder.CurrentStateFinite(&state_invalid_reason);
    }
    const bool objective_finite =
        !add_exception && std::isfinite(ret.JStart) && std::isfinite(ret.JFinal);
    const bool objective_decreased =
        objective_finite && (ret.JFinal <= ret.JStart || force);
    const bool incremental_kept = !add_exception && ret.batchAccepted;
    bool accepted = incremental_kept && state_valid && objective_decreased;
    std::set<PairBoardKey> trial_pair_boards = selected_pair_boards;
    trial_pair_boards.insert(batch_keys.begin(), batch_keys.end());
    const Stage6PersistentResidualStats trial_stats =
        builder.EvaluateAccepted(trial_pair_boards, residual_metric);
    FillPersistentResidualDiagnostics(before_stats, trial_stats,
                                      &pair_decision);
    bool residual_health_guard_pass = true;
    std::string residual_health_reject_reason;
    // The health guard must be expressed in the active selection residual
    // domain. Pixel gates are intentionally retained unchanged; a tangent
    // run uses radians and an adaptive floor tied to the committed state.
    const double angular_health_floor_rad = 0.02;
    const double catastrophic_total_delta =
        residual_metric == Stage6PersistentResidualMetric::TangentPlaneAngular
            ? std::max(angular_health_floor_rad,
                       2.0 * std::max(0.0, before_stats.Rmse()))
            : std::max(1.0, 10.0 * options.pair_selection_max_rmse_delta);
    const double catastrophic_camera_delta =
        residual_metric == Stage6PersistentResidualMetric::TangentPlaneAngular
            ? std::max(angular_health_floor_rad,
                       2.0 * std::max(0.0, std::max(before_stats.Cam0Rmse(),
                                                     before_stats.Cam1Rmse())))
            : std::max(1.0, 10.0 * options.pair_selection_max_camera_rmse_delta);
    if (accepted) {
      if (!std::isfinite(pair_decision.trial_total_rmse) ||
          !std::isfinite(pair_decision.initial_total_rmse)) {
        residual_health_guard_pass = false;
        residual_health_reject_reason =
            "persistent_catastrophic_residual_guard nonfinite_committed_rmse";
      } else if (pair_decision.total_rmse_delta >
                 catastrophic_total_delta) {
        residual_health_guard_pass = false;
        std::ostringstream reason;
        reason << "persistent_catastrophic_residual_guard total_delta="
               << pair_decision.total_rmse_delta << " max="
               << catastrophic_total_delta;
        residual_health_reject_reason = reason.str();
      } else if (pair_decision.cam0_rmse_delta >
                 catastrophic_camera_delta) {
        residual_health_guard_pass = false;
        std::ostringstream reason;
        reason << "persistent_catastrophic_residual_guard cam0_delta="
               << pair_decision.cam0_rmse_delta << " max="
               << catastrophic_camera_delta;
        residual_health_reject_reason = reason.str();
      } else if (pair_decision.cam1_rmse_delta >
                 catastrophic_camera_delta) {
        residual_health_guard_pass = false;
        std::ostringstream reason;
        reason << "persistent_catastrophic_residual_guard cam1_delta="
               << pair_decision.cam1_rmse_delta << " max="
               << catastrophic_camera_delta;
        residual_health_reject_reason = reason.str();
      }
    }
    if (!residual_health_guard_pass) {
      accepted = false;
    }
    const std::string committed_or_rollback =
        accepted ? "committed" : "rollback";

    CopyPersistentReturnValueToPairDecision(
        ret, current_rank, normalization_count, "persistent_pair_cohesive",
        state_valid && objective_decreased, committed_or_rollback,
        &pair_decision);
    pair_decision.force = force;
    pair_decision.accepted = accepted;
    pair_decision.batchAccepted = accepted;
    pair_decision.info_gain_threshold = options.incremental_mi_tol;
    if (accepted) {
      pair_decision.accept_reason =
          force ? "forced_seed"
                : (ret.informationGain > options.incremental_mi_tol
                       ? "incremental_information_gain"
                       : "incremental_rank_gain");
    } else {
      if (add_exception) {
        pair_decision.reject_reason = state_invalid_reason;
      } else if (!incremental_kept) {
        pair_decision.reject_reason = "incremental_estimator_rejected_batch";
      } else if (!state_valid) {
        pair_decision.reject_reason = state_invalid_reason;
      } else if (!objective_decreased) {
        std::ostringstream reason;
        reason << "incremental_objective_increase_gate JStart=" << ret.JStart
               << " JFinal=" << ret.JFinal;
        pair_decision.reject_reason = reason.str();
      } else if (!residual_health_guard_pass) {
        pair_decision.reject_reason = residual_health_reject_reason;
      } else {
        pair_decision.reject_reason = "persistent_incremental_validity_gate";
      }
    }

    ++board_summary.batch_acceptance_attempted_count;
    ++board_summary.attempted_count;
    if (pair_summary != nullptr) {
      ++pair_summary->attempted_count;
    }
    if (accepted) {
      ++board_summary.batch_acceptance_accepted_count;
      ++board_summary.accepted_count;
      if (pair_summary != nullptr) {
        ++pair_summary->accepted_count;
      }
      selected_pair_boards.insert(batch_keys.begin(), batch_keys.end());
      covered_boards = BoardIdsFromPairBoards(selected_pair_boards);
      current_rank = static_cast<int>(ret.rankTheta);
      board_summary.final_selected_rmse = trial_stats.Rmse();
      if (pair_summary != nullptr) {
        pair_summary->final_selected_rmse = trial_stats.Rmse();
      }
      for (StereoPairBoardTrialSelectionDecision board_decision :
           board_decisions) {
        board_decision.attempted = true;
        board_decision.accepted = true;
        board_decision.batchAccepted = true;
        board_decision.force = force;
        board_decision.seed = force;
        board_decision.accept_reason = pair_decision.accept_reason;
        CopyPersistentReturnValueToBoardDecision(
            ret, pair_decision.rank_before, normalization_count,
            "persistent_pair_cohesive", true, "committed",
            &board_decision);
        FillPersistentResidualDiagnostics(before_stats, trial_stats,
                                          &board_decision);
        board_summary.decisions.push_back(board_decision);
      }
    } else {
      ++board_summary.batch_acceptance_rejected_score_count;
      if (!residual_health_guard_pass) {
        ++board_summary
              .persistent_incremental_residual_health_guard_rejected_count;
      }
      ++board_summary.rejected_count;
      if (pair_summary != nullptr) {
        ++pair_summary->rejected_count;
      }
      if (!add_exception && incremental_kept) {
        estimator.rejectBatch(batch);
      }
      builder.RestoreState(state);
      for (StereoPairBoardTrialSelectionDecision board_decision :
           board_decisions) {
        board_decision.attempted = true;
        board_decision.accepted = false;
        board_decision.batchAccepted = false;
        board_decision.force = force;
        board_decision.seed = force;
        board_decision.reject_reason = pair_decision.reject_reason;
        CopyPersistentReturnValueToBoardDecision(
            ret, pair_decision.rank_before, normalization_count,
            "persistent_pair_cohesive", false, "rollback",
            &board_decision);
        FillPersistentResidualDiagnostics(before_stats, trial_stats,
                                          &board_decision);
        board_summary.decisions.push_back(board_decision);
      }
    }
    if (pair_summary != nullptr) {
      pair_summary->decisions.push_back(pair_decision);
    }
  }

  board_summary.final_selected_pair_board_count =
      static_cast<int>(selected_pair_boards.size());
  board_summary.selected_pair_board_keys = selected_pair_boards;
  board_summary.success = !selected_pair_boards.empty();
  board_summary.persistent_incremental_elapsed_time_seconds =
      ElapsedSeconds(start_time);
  board_summary.pair_cohesion_under_target_pair_count_before_rescue =
      CountPairsBelowPairCohesionTarget(
          selected_pair_boards, dataset,
          options.pair_cohesion_min_boards_per_pair);
  board_summary.pair_cohesion_under_target_pair_count_after_rescue =
      board_summary.pair_cohesion_under_target_pair_count_before_rescue;
  board_summary.single_board_pair_count_before_rescue =
      CountPairsBelowSelectedBoardMinimum(
          selected_pair_boards, options.pair_cohesion_min_boards_per_pair);
  board_summary.single_board_pair_count_after_rescue =
      board_summary.single_board_pair_count_before_rescue;
  board_summary.single_board_pair_count_after_policy =
      board_summary.single_board_pair_count_after_rescue;
  if (!board_summary.success) {
    board_summary.failure_reason =
        "persistent_incremental_selection_rejected_all_batches";
    return board_summary;
  }
  *scene_state = builder.BuildSceneState(*scene_state);
  *final_selection =
      BuildSelectionSummaryFromSelectedPairBoards(
          base_selection, dataset, selected_pair_boards);
  if (pair_summary != nullptr) {
    pair_summary->success = true;
    pair_summary->final_selected_pair_count =
        static_cast<int>(PairIndicesFromPairBoards(selected_pair_boards).size());
    pair_summary->selected_pair_indices =
        PairIndicesFromPairBoards(selected_pair_boards);
  }
  return board_summary;
}

StereoPairBoardTrialSelectionSummary
RunPersistentIncrementalPairCohesiveSelection(
    const StereoMeasurementDataset& dataset,
    const StereoExtrinsicSolverOptions& options,
    const StereoPairSelectionSummary& base_selection,
    StereoPairTrialSelectionSummary* pair_summary,
    StereoPairSelectionSummary* final_selection,
    StereoSceneState* scene_state) {
  if (scene_state == nullptr) {
    StereoPairBoardTrialSelectionSummary summary;
    summary.failure_reason = "missing_scene_state";
    return summary;
  }
  const std::string cam0_family = scene_state->cam0.camera_model_family;
  const std::string cam1_family = scene_state->cam1.camera_model_family;
  if (cam0_family == "ds-none" && cam1_family == "ds-none") {
    return RunPersistentIncrementalPairCohesiveSelectionTyped
        <DsGeometry, DsGeometry>(
            dataset, options, base_selection, pair_summary, final_selection,
            scene_state);
  }
  if (cam0_family == "omni-none" && cam1_family == "omni-none") {
    return RunPersistentIncrementalPairCohesiveSelectionTyped
        <OmniGeometry, OmniGeometry>(
            dataset, options, base_selection, pair_summary, final_selection,
            scene_state);
  }
  if (cam0_family == "omni-radtan" && cam1_family == "omni-radtan") {
    return RunPersistentIncrementalPairCohesiveSelectionTyped
        <OmniRadtanGeometry, OmniRadtanGeometry>(
            dataset, options, base_selection, pair_summary, final_selection,
            scene_state);
  }
  if (cam0_family == "ds-none" && cam1_family == "eucm-none") {
    return RunPersistentIncrementalPairCohesiveSelectionTyped
        <DsGeometry, EucmGeometry>(
            dataset, options, base_selection, pair_summary, final_selection,
            scene_state);
  }
  if (cam0_family == "eucm-none" && cam1_family == "ds-none") {
    return RunPersistentIncrementalPairCohesiveSelectionTyped
        <EucmGeometry, DsGeometry>(
            dataset, options, base_selection, pair_summary, final_selection,
            scene_state);
  }
  if (cam0_family == "eucm-none" && cam1_family == "eucm-none") {
    return RunPersistentIncrementalPairCohesiveSelectionTyped
        <EucmGeometry, EucmGeometry>(
            dataset, options, base_selection, pair_summary, final_selection,
            scene_state);
  }
  if (cam0_family == "pinhole-equi" && cam1_family == "pinhole-equi") {
    return RunPersistentIncrementalPairCohesiveSelectionTyped
        <PinholeEquiGeometry, PinholeEquiGeometry>(
            dataset, options, base_selection, pair_summary, final_selection,
            scene_state);
  }
  StereoPairBoardTrialSelectionSummary summary;
  summary.enabled = true;
  summary.incremental_estimator_enabled = true;
  summary.persistent_incremental_estimator_used = true;
  summary.failure_reason =
      "unsupported_persistent_incremental_camera_families: " + cam0_family +
      "/" + cam1_family;
  return summary;
}

Stage6IncrementalBatchEstimatorOptions MakeStage6IncrementalEstimatorOptions(
    const StereoExtrinsicSolverOptions& options,
    bool pair_level) {
  Stage6IncrementalBatchEstimatorOptions estimator_options;
  estimator_options.enabled = options.enable_stage6_incremental_estimator ||
                              options.enable_committing_pair_batch_selection;
  estimator_options.info_gain_threshold = options.incremental_mi_tol;
  estimator_options.rank_threshold = options.incremental_rank_threshold;
  estimator_options.info_block = options.incremental_info_block;
  const double legacy_rotation_limit =
      pair_level ? options.pair_selection_max_baseline_rotation_delta_deg
                 : options.pair_board_selection_max_baseline_rotation_delta_deg;
  const double legacy_translation_limit =
      pair_level ? options.pair_selection_max_baseline_translation_delta_m
                 : options.pair_board_selection_max_baseline_translation_delta_m;
  estimator_options.max_baseline_rotation_delta_deg =
      std::max(2.0, 10.0 * legacy_rotation_limit);
  estimator_options.max_baseline_translation_delta_m =
      std::max(0.05, 10.0 * legacy_translation_limit);
  return estimator_options;
}

Stage6IncrementalBatchCandidate MakeStage6IncrementalBatchCandidate(
    const char* batch_type,
    int pair_index,
    const std::set<PairBoardKey>& selected_pair_boards_before,
    const std::set<PairBoardKey>& selected_pair_boards_after,
    bool force = false) {
  Stage6IncrementalBatchCandidate candidate;
  candidate.batch_type = batch_type == nullptr ? "" : batch_type;
  candidate.pair_index = pair_index;
  candidate.selected_pair_boards_before = selected_pair_boards_before;
  candidate.selected_pair_boards_after = selected_pair_boards_after;
  candidate.force = force;
  return candidate;
}

void CopyIncrementalBatchResultToDecision(
    const Stage6IncrementalBatchResult& result,
    const std::string& batch_type,
    StereoPairBoardTrialSelectionDecision* decision) {
  if (decision == nullptr) {
    return;
  }
  decision->incremental_estimator_enabled = true;
  decision->candidate_batch_type = batch_type;
  decision->batchAccepted = result.batchAccepted;
  decision->accept_reason = result.accept_reason;
  decision->solution_valid = result.solution_valid;
  decision->optimization_success = result.optimization_success;
  decision->num_iterations = result.num_iterations;
  decision->objective_before = result.objective_before;
  decision->objective_after = result.objective_after;
  decision->initial_total_rmse = result.rmse_before;
  decision->trial_total_rmse = result.rmse_after;
  decision->total_rmse_delta = result.total_rmse_delta;
  decision->cam0_rmse_delta = result.cam0_rmse_delta;
  decision->cam1_rmse_delta = result.cam1_rmse_delta;
  decision->baseline_rotation_delta_deg = result.baseline_rotation_delta_deg;
  decision->baseline_translation_delta_m = result.baseline_translation_delta_m;
  decision->hard_validity_pass = result.solution_valid;
  decision->marginal_information_gain_proxy =
      result.marginal_information_gain_proxy;
  decision->information_gain_proxy = result.marginal_information_gain_proxy;
  decision->rank_before = result.rank_before;
  decision->rank_after = result.rank_after;
  decision->rank_proxy_increases = result.rank_proxy_increases;
  decision->info_gain_threshold = result.info_gain_threshold;
  decision->committed_or_rollback = result.committed_or_rollback;
  decision->force = result.force;
  decision->accepted = result.batchAccepted;
  decision->accepted_by_batch_acceptance = result.batchAccepted;
  decision->reject_reason = result.reject_reason;
}

void CopyIncrementalBatchResultToDecision(
    const Stage6IncrementalBatchResult& result,
    const std::string& batch_type,
    StereoPairTrialSelectionDecision* decision) {
  if (decision == nullptr) {
    return;
  }
  decision->incremental_estimator_enabled = true;
  decision->candidate_batch_type = batch_type;
  decision->batchAccepted = result.batchAccepted;
  decision->accept_reason = result.accept_reason;
  decision->solution_valid = result.solution_valid;
  decision->optimization_success = result.optimization_success;
  decision->num_iterations = result.num_iterations;
  decision->objective_before = result.objective_before;
  decision->objective_after = result.objective_after;
  decision->initial_total_rmse = result.rmse_before;
  decision->trial_total_rmse = result.rmse_after;
  decision->total_rmse_delta = result.total_rmse_delta;
  decision->cam0_rmse_delta = result.cam0_rmse_delta;
  decision->cam1_rmse_delta = result.cam1_rmse_delta;
  decision->baseline_rotation_delta_deg = result.baseline_rotation_delta_deg;
  decision->baseline_translation_delta_m = result.baseline_translation_delta_m;
  decision->marginal_information_gain_proxy =
      result.marginal_information_gain_proxy;
  decision->rank_before = result.rank_before;
  decision->rank_after = result.rank_after;
  decision->rank_proxy_increases = result.rank_proxy_increases;
  decision->info_gain_threshold = result.info_gain_threshold;
  decision->committed_or_rollback = result.committed_or_rollback;
  decision->force = result.force;
  decision->accepted = result.batchAccepted;
  decision->reject_reason = result.reject_reason;
}

StereoPairBoardTrialSelectionSummary RunKalibrStyleCommittingPairBatchSelection(
    const StereoMeasurementDataset& dataset,
    const StereoExtrinsicSolverOptions& options,
    const StereoPairSelectionSummary& base_selection,
    StereoPairTrialSelectionSummary* pair_summary,
    StereoPairSelectionSummary* final_selection,
    StereoSceneState* scene_state) {
  StereoPairBoardTrialSelectionSummary board_summary;
  board_summary.enabled = true;
  board_summary.pairboard_selection_mode = options.pairboard_selection_mode;
  board_summary.incremental_estimator_enabled = true;
  board_summary.marginal_information_gain_proxy_enabled = true;
  board_summary.rmse_delta_diagnostics_only = true;
  board_summary.incremental_mi_tol = options.incremental_mi_tol;
  board_summary.incremental_rank_threshold =
      options.incremental_rank_threshold;
  board_summary.incremental_info_block = ToString(options.incremental_info_block);
  if (pair_summary != nullptr) {
    *pair_summary = StereoPairTrialSelectionSummary();
    pair_summary->enabled = true;
    pair_summary->requested_seed_count = 0;
    pair_summary->warnings.push_back("committing_pair_batch_selection");
  }
  if (final_selection == nullptr || scene_state == nullptr) {
    board_summary.failure_reason = "missing_output_pointer";
    return board_summary;
  }
  if (options.solver_mode != StereoSolverMode::GlobalSparseBa &&
      options.solver_mode != StereoSolverMode::SharedOnlyGlobalSparseBa) {
    board_summary.failure_reason =
        "committing_pair_batch_selection_requires_global_sparse_ba_solver_mode";
    return board_summary;
  }

  std::vector<const StereoPairSelectionRow*> ranked_rows;
  for (const StereoPairSelectionRow& row : base_selection.rows) {
    if (!row.eligible || row.shared_board_count <= 0) {
      continue;
    }
    if (dataset.pair_shared_board_ids.count(row.pair_index) == 0) {
      continue;
    }
    ranked_rows.push_back(&row);
  }
  std::sort(ranked_rows.begin(), ranked_rows.end(),
            [](const StereoPairSelectionRow* lhs,
               const StereoPairSelectionRow* rhs) {
              return IsSelectionRowBetter(*lhs, *rhs);
            });
  board_summary.candidate_count = static_cast<int>(ranked_rows.size());
  board_summary.valid_candidate_count = board_summary.candidate_count;
  board_summary.budget_mode = options.pair_selection_budget_mode;
  board_summary.runtime_safety_ceiling = options.pair_selection_runtime_safety_ceiling;
  board_summary.max_candidate_additions_effective =
      options.pair_selection_budget_mode == StereoCandidateBudgetMode::KalibrStyle
          ? "ignored_in_kalibr_style_batch"
          : std::to_string(options.pair_selection_max_candidate_additions);
  if (pair_summary != nullptr) {
    pair_summary->candidate_count = static_cast<int>(ranked_rows.size());
    pair_summary->valid_candidate_count = static_cast<int>(ranked_rows.size());
    pair_summary->budget_mode = options.pair_selection_budget_mode;
    pair_summary->runtime_safety_ceiling =
        options.pair_selection_runtime_safety_ceiling;
    pair_summary->max_candidate_additions_effective =
        board_summary.max_candidate_additions_effective;
  }
  if (ranked_rows.empty()) {
    board_summary.failure_reason = "no_valid_committing_pair_batch_candidates";
    return board_summary;
  }

  StereoExtrinsicSolverOptions trial_options = options;
  trial_options.final_ba_residual_mode = options.selection_ba_residual_mode;
  trial_options.final_ba_optimize_intrinsics = false;
  trial_options.final_ba_optimize_stereo_extrinsic = true;
  trial_options.final_ba_optimize_pair_poses = true;
  trial_options.final_ba_optimize_board_poses = true;
  trial_options.ba_max_iterations = std::max(1, std::min(options.ba_max_iterations, 10));
  ApplySelectionCoObsFactorBaOptions(&trial_options);
  const bool commit_trial_state =
      options.selection_optimization_mode ==
      StereoSelectionOptimizationMode::TrialBaCommit;
  const bool information_gain_only =
      options.selection_optimization_mode ==
      StereoSelectionOptimizationMode::InformationGainOnly;
  const Stage6IncrementalBatchEstimator pair_estimator(
      MakeStage6IncrementalEstimatorOptions(options, true));
  const Stage6IncrementalBatchEstimator pair_board_estimator(
      MakeStage6IncrementalEstimatorOptions(options, false));
  StereoSceneState current_scene = *scene_state;
  std::set<PairBoardKey> selected_pair_boards;
  StereoPairSelectionSummary current_selection =
      BuildSelectionSummaryFromSelectedPairBoards(
          base_selection, dataset, selected_pair_boards);
  StereoMeasurementDataset current_dataset =
      MakePairBoardMaskedDataset(dataset, selected_pair_boards);
  StereoResidualEvaluator evaluator(
      StereoResidualEvaluationOptions{
          false,
          options.pair_pose_refit_mode,
          options.symmetric_refit_max_iterations,
          options.symmetric_refit_step,
          false});
  StereoResidualSummary current_residual;
  current_residual.success = true;
  current_residual.total_stereo_rmse = 0.0;
  current_residual.cam0_rmse = 0.0;
  current_residual.cam1_rmse = 0.0;
  std::set<int> pair_diversity_rescue_pair_indices;
  board_summary.initial_seed_rmse = 0.0;
  if (pair_summary != nullptr) {
    pair_summary->initial_seed_rmse = 0.0;
  }

  const StereoCandidateTraversalBudget pair_budget =
      ComputeStereoCandidateTraversalBudget(
          options.pair_selection_budget_mode,
          static_cast<int>(ranked_rows.size()),
          options.pair_selection_max_candidate_additions,
          options.pair_selection_adaptive_budget_ratio,
          options.pair_selection_adaptive_budget_min,
          options.pair_selection_adaptive_budget_max,
          options.pair_selection_runtime_safety_ceiling);
  board_summary.budget_mode = pair_budget.mode;
  board_summary.runtime_safety_ceiling = pair_budget.runtime_safety_ceiling;
  board_summary.max_candidate_additions_effective =
      pair_budget.max_candidate_additions_effective;
  if (pair_summary != nullptr) {
    pair_summary->budget_mode = pair_budget.mode;
    pair_summary->runtime_safety_ceiling = pair_budget.runtime_safety_ceiling;
    pair_summary->max_candidate_additions_effective =
        pair_budget.max_candidate_additions_effective;
  }

  auto try_pair_board =
      [&](StereoPairBoardTrialSelectionDecision* decision,
          const char* residual_label) -> bool {
        if (decision == nullptr) {
          return false;
        }
        const PairBoardKey key(decision->pair_index, decision->board_id);
        if (selected_pair_boards.count(key) > 0) {
          return false;
        }
        decision->pairboard_selection_mode = options.pairboard_selection_mode;
        decision->pair_cohesion_candidate = true;
        decision->selected_pair_board_count_before =
            static_cast<int>(selected_pair_boards.size());
        decision->selected_pair_count_before =
            static_cast<int>(PairIndicesFromPairBoards(selected_pair_boards).size());
        decision->selected_board_count_before =
            static_cast<int>(BoardIdsFromPairBoards(selected_pair_boards).size());
        decision->coverage_gain =
            ComputePairBoardCoverageGain(selected_pair_boards,
                                         decision->pair_index,
                                         decision->board_id);
        decision->attempted = true;
        std::set<PairBoardKey> trial_pair_boards = selected_pair_boards;
        trial_pair_boards.insert(key);
        StereoPairSelectionSummary trial_selection =
            BuildSelectionSummaryFromSelectedPairBoards(
                base_selection, dataset, trial_pair_boards);
        StereoMeasurementDataset trial_dataset =
            MakePairBoardMaskedDataset(dataset, trial_pair_boards);
        StereoSceneState trial_scene = current_scene;
        const Eigen::Isometry3d baseline_before =
            ToIsometry3d(current_scene.T_cam1_cam0);
        if (information_gain_only) {
          const Stage6IncrementalBatchCandidate candidate =
              MakeStage6IncrementalBatchCandidate(
                  "pair_board_fallback_information_gain_only",
                  decision->pair_index, selected_pair_boards,
                  trial_pair_boards);
          Stage6IncrementalBatchResult batch_result =
              pair_board_estimator.EvaluateInformationGainOnly(
                  dataset, current_scene, candidate);
          if (!batch_result.batchAccepted &&
              options.enable_incremental_pair_diversity_rescue &&
              pair_diversity_rescue_pair_indices.count(decision->pair_index) >
                  0 &&
              CountSelectedPairBoardForPair(selected_pair_boards,
                                            decision->pair_index) <
                  std::max(1,
                           options
                               .incremental_pair_diversity_rescue_min_boards) &&
              batch_result.reject_reason == "marginal_information_gain_gate") {
            batch_result.batchAccepted = true;
            batch_result.accept_reason = "pair_diversity_rescue";
            batch_result.reject_reason.clear();
            batch_result.committed_or_rollback = "selection_only";
          }
          ++board_summary.batch_acceptance_attempted_count;
          CopyIncrementalBatchResultToDecision(
              batch_result, candidate.batch_type, decision);
          decision->selection_optimization_mode =
              ToString(options.selection_optimization_mode);
          if (!batch_result.batchAccepted) {
            ++board_summary.batch_acceptance_rejected_score_count;
            return false;
          }
          ++board_summary.batch_acceptance_accepted_count;
          selected_pair_boards.insert(key);
          current_selection = trial_selection;
          current_dataset = trial_dataset;
          return true;
        }
        StereoGlobalSparseBaSummary trial_ba_summary;
        if (!RunGlobalSparseBa(trial_dataset, trial_options, trial_selection,
                               &trial_ba_summary, &trial_scene)) {
          ++board_summary.batch_acceptance_attempted_count;
          ++board_summary.batch_acceptance_rejected_hard_validity_count;
          decision->reject_reason =
              trial_ba_summary.failure_reason.empty() ? "hard_validity_gate"
                                                      : trial_ba_summary.failure_reason;
          return false;
        }
        CopyCoObsTrialDiagnostics(trial_ba_summary, decision);
        const StereoResidualSummary trial_residual = evaluator.Evaluate(
            trial_dataset, trial_scene, trial_selection.selected_pair_indices,
            residual_label);
        const Stage6IncrementalBatchCandidate candidate =
            MakeStage6IncrementalBatchCandidate(
                "pair_board_fallback", decision->pair_index,
                selected_pair_boards, trial_pair_boards);
        Stage6IncrementalBatchResult batch_result =
            pair_board_estimator.AddBatch(
                dataset, current_scene, trial_scene, current_residual,
                trial_residual, trial_ba_summary, candidate);
        if (!batch_result.batchAccepted &&
            options.enable_incremental_pair_diversity_rescue &&
            pair_diversity_rescue_pair_indices.count(decision->pair_index) >
                0 &&
            CountSelectedPairBoardForPair(selected_pair_boards,
                                          decision->pair_index) <
                std::max(1,
                         options
                             .incremental_pair_diversity_rescue_min_boards) &&
            batch_result.solution_valid &&
            batch_result.reject_reason == "marginal_information_gain_gate") {
          batch_result.batchAccepted = true;
          batch_result.accept_reason = "pair_diversity_rescue";
          batch_result.reject_reason.clear();
          batch_result.committed_or_rollback = "committed";
        }
	        TryAcceptByCoObsAwareScore(
	            trial_ba_summary, options, batch_result.solution_valid,
	            CountSelectedPairBoardForPair(selected_pair_boards,
	                                          decision->pair_index) > 0,
	            &batch_result, decision);
        ++board_summary.batch_acceptance_attempted_count;
        CopyIncrementalBatchResultToDecision(
            batch_result, candidate.batch_type, decision);
        if (!batch_result.batchAccepted) {
          if (batch_result.reject_reason == "hard_validity_gate") {
            ++board_summary.batch_acceptance_rejected_hard_validity_count;
          } else {
            ++board_summary.batch_acceptance_rejected_score_count;
          }
        } else {
          ++board_summary.batch_acceptance_accepted_count;
          if (batch_result.accept_reason == "marginal_information_gain" ||
              batch_result.accept_reason == "rank_proxy_increase") {
            ++board_summary
                  .batch_acceptance_rescued_from_legacy_rmse_gate_count;
          }
        }
        const bool accepted = batch_result.batchAccepted;
        if (!accepted) {
          return false;
        }
        selected_pair_boards.insert(key);
        current_selection = trial_selection;
        current_dataset = trial_dataset;
        if (commit_trial_state) {
          current_scene = trial_scene;
        }
        current_residual = trial_residual;
        return true;
      };

  int added_pair_count = 0;
  for (const StereoPairSelectionRow* row_ptr : ranked_rows) {
    if (row_ptr == nullptr) {
      continue;
    }
    if (pair_budget.mode == StereoCandidateBudgetMode::Fixed &&
        added_pair_count >= pair_budget.traversal_limit) {
      break;
    }
    if (pair_budget.mode != StereoCandidateBudgetMode::Fixed &&
        board_summary.valid_candidate_traversed_count >=
            pair_budget.traversal_limit) {
      board_summary.safety_ceiling_hit =
          pair_budget.mode == StereoCandidateBudgetMode::KalibrStyle;
      if (pair_summary != nullptr) {
        pair_summary->safety_ceiling_hit = board_summary.safety_ceiling_hit;
        pair_summary->warnings.push_back(
            "runtime_safety_ceiling_hit; result is runtime-capped");
      }
      board_summary.warnings.push_back(
          pair_budget.mode == StereoCandidateBudgetMode::KalibrStyle
              ? "runtime_safety_ceiling_hit; result is runtime-capped"
              : "candidate_budget_limit_hit");
      break;
    }
    ++board_summary.valid_candidate_traversed_count;
    if (pair_summary != nullptr) {
      ++pair_summary->valid_candidate_traversed_count;
    }

    const StereoPairSelectionRow& row = *row_ptr;
    const auto shared_it = dataset.pair_shared_board_ids.find(row.pair_index);
    if (shared_it == dataset.pair_shared_board_ids.end()) {
      continue;
    }
    std::vector<StereoPairBoardTrialSelectionDecision> batch_board_decisions;
    for (int board_id : shared_it->second) {
      batch_board_decisions.push_back(
          MakePairBoardDecision(dataset, current_scene, options,
                                row.pair_index, board_id));
    }

    StereoPairTrialSelectionDecision pair_decision =
        MakeTrialDecisionFromRow(dataset, row, BoardIdsFromPairBoards(selected_pair_boards));
    pair_decision.seed = false;
    pair_decision.attempted = true;
    AccumulatePairBatchDiagnostics(&pair_decision, batch_board_decisions);
    if (pair_summary != nullptr) {
      ++pair_summary->attempted_count;
    }

    std::set<PairBoardKey> trial_pair_boards = selected_pair_boards;
    for (const StereoPairBoardTrialSelectionDecision& board_decision :
         batch_board_decisions) {
      trial_pair_boards.insert(
          PairBoardKey(board_decision.pair_index, board_decision.board_id));
    }
    StereoPairSelectionSummary trial_selection =
        BuildSelectionSummaryFromSelectedPairBoards(
            base_selection, dataset, trial_pair_boards);
    StereoMeasurementDataset trial_dataset =
        MakePairBoardMaskedDataset(dataset, trial_pair_boards);
    StereoSceneState trial_scene = current_scene;
    StereoGlobalSparseBaSummary trial_ba_summary;
    bool pair_accepted = false;
    if (information_gain_only) {
      const Stage6IncrementalBatchCandidate candidate =
          MakeStage6IncrementalBatchCandidate(
              "pair_level_information_gain_only", row.pair_index,
              selected_pair_boards, trial_pair_boards);
      const Stage6IncrementalBatchResult batch_result =
          pair_estimator.EvaluateInformationGainOnly(
              dataset, current_scene, candidate);
      ++board_summary.batch_acceptance_attempted_count;
      CopyIncrementalBatchResultToDecision(
          batch_result, candidate.batch_type, &pair_decision);
      pair_decision.selection_optimization_mode =
          ToString(options.selection_optimization_mode);
      pair_accepted = batch_result.batchAccepted;
      if (pair_accepted) {
        ++board_summary.batch_acceptance_accepted_count;
      } else {
        ++board_summary.batch_acceptance_rejected_score_count;
      }
      if (pair_accepted) {
        selected_pair_boards = trial_pair_boards;
        current_selection = trial_selection;
        current_dataset = trial_dataset;
        for (StereoPairBoardTrialSelectionDecision board_decision :
             batch_board_decisions) {
          board_decision.attempted = true;
          board_decision.accepted = true;
          board_decision.reject_reason.clear();
          CopyIncrementalBatchResultToDecision(
              batch_result, candidate.batch_type, &board_decision);
          board_decision.selection_optimization_mode =
              ToString(options.selection_optimization_mode);
          board_decision.accepted = true;
          board_decision.reject_reason.clear();
          board_summary.decisions.push_back(board_decision);
        }
      }
    } else if (!RunGlobalSparseBa(trial_dataset, trial_options, trial_selection,
                                  &trial_ba_summary, &trial_scene)) {
      ++board_summary.batch_acceptance_attempted_count;
      ++board_summary.batch_acceptance_rejected_hard_validity_count;
      pair_decision.reject_reason =
          trial_ba_summary.failure_reason.empty() ? "hard_validity_gate"
                                                  : trial_ba_summary.failure_reason;
    } else {
      CopyCoObsTrialDiagnostics(trial_ba_summary, &pair_decision);
      const StereoResidualSummary trial_residual = evaluator.Evaluate(
          trial_dataset, trial_scene, trial_selection.selected_pair_indices,
          "stage6_committing_pair_batch_candidate");
      const Stage6IncrementalBatchCandidate candidate =
          MakeStage6IncrementalBatchCandidate(
              "pair_level", row.pair_index, selected_pair_boards,
              trial_pair_boards);
      Stage6IncrementalBatchResult batch_result =
          pair_estimator.AddBatch(dataset, current_scene, trial_scene,
                                  current_residual, trial_residual,
                                  trial_ba_summary, candidate);
      TryAcceptByCoObsAwareScore(
          trial_ba_summary, options, batch_result.solution_valid,
          &batch_result, &pair_decision);
      ++board_summary.batch_acceptance_attempted_count;
      CopyIncrementalBatchResultToDecision(
          batch_result, candidate.batch_type, &pair_decision);
      pair_accepted = batch_result.batchAccepted;
      if (pair_accepted) {
        ++board_summary.batch_acceptance_accepted_count;
      } else if (batch_result.reject_reason == "hard_validity_gate") {
        ++board_summary.batch_acceptance_rejected_hard_validity_count;
      } else {
        ++board_summary.batch_acceptance_rejected_score_count;
      }
      if (pair_accepted) {
        selected_pair_boards = trial_pair_boards;
        current_selection = trial_selection;
        current_dataset = trial_dataset;
        if (commit_trial_state) {
          current_scene = trial_scene;
        }
        current_residual = trial_residual;
        for (StereoPairBoardTrialSelectionDecision board_decision :
             batch_board_decisions) {
          board_decision.attempted = true;
          board_decision.accepted = true;
          board_decision.reject_reason.clear();
          CopyIncrementalBatchResultToDecision(
              batch_result, candidate.batch_type, &board_decision);
          board_decision.accepted = true;
          board_decision.reject_reason.clear();
          board_summary.decisions.push_back(board_decision);
        }
      }
    }
    if (pair_summary != nullptr) {
      if (pair_accepted) {
        ++pair_summary->accepted_count;
      } else {
        ++pair_summary->rejected_count;
      }
      pair_summary->decisions.push_back(pair_decision);
    }
    if (pair_accepted) {
      ++board_summary.attempted_count;
      board_summary.accepted_count += static_cast<int>(batch_board_decisions.size());
      ++added_pair_count;
      continue;
    }

    if (options.enable_incremental_pair_diversity_rescue &&
        pair_decision.solution_valid &&
        pair_decision.reject_reason == "marginal_information_gain_gate") {
      pair_diversity_rescue_pair_indices.insert(row.pair_index);
      std::sort(batch_board_decisions.begin(), batch_board_decisions.end(),
                [](const StereoPairBoardTrialSelectionDecision& lhs,
                   const StereoPairBoardTrialSelectionDecision& rhs) {
                  if (lhs.candidate_score != rhs.candidate_score) {
                    return lhs.candidate_score > rhs.candidate_score;
                  }
                  if (lhs.shared_point_count != rhs.shared_point_count) {
                    return lhs.shared_point_count > rhs.shared_point_count;
                  }
                  return lhs.board_id < rhs.board_id;
                });
    }

    for (StereoPairBoardTrialSelectionDecision board_decision :
         batch_board_decisions) {
      ++board_summary.pair_cohesion_candidate_count;
      const bool accepted =
          try_pair_board(&board_decision,
                         "stage6_committing_pair_batch_fallback_pair_board");
      if (board_decision.attempted) {
        ++board_summary.attempted_count;
        ++board_summary.pair_cohesion_attempted_count;
      }
      if (accepted) {
        ++board_summary.accepted_count;
        ++board_summary.pair_cohesion_accepted_count;
      } else {
        ++board_summary.rejected_count;
        ++board_summary.pair_cohesion_rejected_count;
      }
      board_summary.decisions.push_back(board_decision);
    }
  }

  if (ApplyAblationExcludedPairBoards(
          options.ablation_excluded_pair_boards,
          &selected_pair_boards,
          &board_summary.warnings) > 0) {
    RefitAfterPairBoardAblation(
        dataset,
        options,
        trial_options,
        base_selection,
        selected_pair_boards,
        "stage6_committing_pair_batch_ablation",
        &current_scene,
        &current_residual,
        &board_summary.warnings);
  }

  board_summary.final_selected_pair_board_count =
      static_cast<int>(selected_pair_boards.size());
  board_summary.final_selected_rmse = current_residual.total_stereo_rmse;
  board_summary.selected_pair_board_keys = selected_pair_boards;
  board_summary.success = !selected_pair_boards.empty();
  board_summary.single_board_pair_count_before_rescue =
      CountPairsBelowSelectedBoardMinimum(
          selected_pair_boards, options.pair_cohesion_min_boards_per_pair);
  board_summary.single_board_pair_count_after_rescue =
      board_summary.single_board_pair_count_before_rescue;
  board_summary.single_board_pair_count_after_policy =
      board_summary.single_board_pair_count_after_rescue;
  board_summary.pair_cohesion_under_target_pair_count_before_rescue =
      CountPairsBelowPairCohesionTarget(
          selected_pair_boards, dataset, options.pair_cohesion_min_boards_per_pair);
  board_summary.pair_cohesion_under_target_pair_count_after_rescue =
      board_summary.pair_cohesion_under_target_pair_count_before_rescue;
  if (pair_summary != nullptr) {
    pair_summary->final_selected_pair_count =
        static_cast<int>(PairIndicesFromPairBoards(selected_pair_boards).size());
    pair_summary->final_selected_rmse = current_residual.total_stereo_rmse;
    pair_summary->selected_pair_indices = PairIndicesFromPairBoards(selected_pair_boards);
    pair_summary->success = !selected_pair_boards.empty();
  }
  *scene_state = current_scene;
  *final_selection =
      BuildSelectionSummaryFromSelectedPairBoards(
          base_selection, dataset, selected_pair_boards);
  return board_summary;
}

StereoExtrinsicUncertaintySummary ComputeExtrinsicUncertaintySummary(
    const StereoPairOnlyBaInitSummary& pair_init_summary,
    const StereoSceneState& scene_state) {
  StereoExtrinsicUncertaintySummary summary;
  summary.enabled = true;
  const Eigen::Isometry3d reference_baseline =
      ToIsometry3d(scene_state.T_cam1_cam0);
  std::vector<double> rotation_deltas;
  std::vector<double> translation_deltas;
  std::vector<double> baseline_lengths;
  std::vector<TransformCandidate> accepted_candidates;
  for (const StereoPairInitCandidateRow& row : pair_init_summary.candidates) {
    if (!row.consistency_accepted) {
      continue;
    }
    const Eigen::Isometry3d candidate = ToIsometry3d(row.T_cam1_cam0_candidate);
    StereoExtrinsicCandidateDispersionRow dispersion_row;
    dispersion_row.pair_index = row.pair_index;
    dispersion_row.board_id = row.board_id;
    dispersion_row.consistency_accepted = true;
    dispersion_row.rotation_delta_deg =
        RotationDistanceRadians(reference_baseline, candidate) * 180.0 / M_PI;
    dispersion_row.translation_delta_m =
        (reference_baseline.translation() - candidate.translation()).norm();
    dispersion_row.baseline_length = candidate.translation().norm();
    summary.candidate_rows.push_back(dispersion_row);
    rotation_deltas.push_back(dispersion_row.rotation_delta_deg);
    translation_deltas.push_back(dispersion_row.translation_delta_m);
    baseline_lengths.push_back(dispersion_row.baseline_length);
    TransformCandidate transform_candidate;
    transform_candidate.transform = candidate;
    transform_candidate.weight = 1.0;
    accepted_candidates.push_back(transform_candidate);
  }
  summary.candidate_count =
      static_cast<int>(pair_init_summary.candidates.size());
  summary.accepted_candidate_count =
      static_cast<int>(summary.candidate_rows.size());
  if (summary.accepted_candidate_count <= 0) {
    summary.failure_reason = "no_consistency_accepted_candidates";
    return summary;
  }
  summary.rotation_delta_mean_deg = ComputeMean(rotation_deltas);
  summary.rotation_delta_median_deg = ComputeMedian(rotation_deltas);
  summary.translation_delta_mean_m = ComputeMean(translation_deltas);
  summary.translation_delta_median_m = ComputeMedian(translation_deltas);
  summary.baseline_length_mean = ComputeMean(baseline_lengths);
  summary.baseline_length_std =
      ComputeStddev(baseline_lengths, summary.baseline_length_mean);

  double worst_rotation = 0.0;
  double worst_translation = 0.0;
  int worst_pair = -1;
  for (const StereoExtrinsicCandidateDispersionRow& row : summary.candidate_rows) {
    std::vector<TransformCandidate> jackknife_candidates;
    for (const StereoExtrinsicCandidateDispersionRow& other : summary.candidate_rows) {
      if (other.pair_index == row.pair_index && other.board_id == row.board_id) {
        continue;
      }
      TransformCandidate candidate;
      candidate.transform = ToIsometry3d(
          pair_init_summary.candidates[0].T_cam1_cam0_candidate);
      jackknife_candidates.push_back(candidate);
    }
    jackknife_candidates.clear();
    for (const StereoPairInitCandidateRow& candidate_row : pair_init_summary.candidates) {
      if (!candidate_row.consistency_accepted) {
        continue;
      }
      if (candidate_row.pair_index == row.pair_index) {
        continue;
      }
      TransformCandidate candidate;
      candidate.transform = ToIsometry3d(candidate_row.T_cam1_cam0_candidate);
      candidate.weight = 1.0;
      jackknife_candidates.push_back(candidate);
    }
    StereoExtrinsicJackknifeRow jackknife_row;
    jackknife_row.excluded_pair_index = row.pair_index;
    jackknife_row.remaining_candidate_count =
        static_cast<int>(jackknife_candidates.size());
    if (jackknife_candidates.empty()) {
      jackknife_row.rotation_delta_deg =
          std::numeric_limits<double>::infinity();
      jackknife_row.translation_delta_m =
          std::numeric_limits<double>::infinity();
      jackknife_row.baseline_length = std::numeric_limits<double>::infinity();
      summary.jackknife_rows.push_back(jackknife_row);
      continue;
    }
    const Eigen::Isometry3d jackknife_baseline =
        AverageTransforms(jackknife_candidates);
    jackknife_row.rotation_delta_deg =
        RotationDistanceRadians(reference_baseline, jackknife_baseline) *
        180.0 / M_PI;
    jackknife_row.translation_delta_m =
        (reference_baseline.translation() - jackknife_baseline.translation()).norm();
    jackknife_row.baseline_length = jackknife_baseline.translation().norm();
    summary.jackknife_rows.push_back(jackknife_row);
    if (jackknife_row.rotation_delta_deg > worst_rotation ||
        jackknife_row.translation_delta_m > worst_translation) {
      worst_rotation =
          std::max(worst_rotation, jackknife_row.rotation_delta_deg);
      worst_translation =
          std::max(worst_translation, jackknife_row.translation_delta_m);
      worst_pair = jackknife_row.excluded_pair_index;
    }
  }
  summary.jackknife_rotation_max_deg = worst_rotation;
  summary.jackknife_translation_max_m = worst_translation;
  summary.worst_jackknife_pair_index = worst_pair;
  summary.success = true;
  return summary;
}

}  // namespace

StereoExtrinsicCalibrationRunner::StereoExtrinsicCalibrationRunner(
    StereoExtrinsicSolverOptions options)
    : options_(std::move(options)) {}

StereoExtrinsicCalibrationResult StereoExtrinsicCalibrationRunner::Run(
    const StereoExtrinsicProblemInput& input) const {
  const Clock::time_point total_start = Clock::now();
  StereoExtrinsicCalibrationResult result;
  result.problem_input = input;
  result.optimized_scene = input.initial_scene;
  RuntimeCounters runtime_counters;
  const Clock::time_point initialization_start = Clock::now();
  std::cout << "[Stage6] runner: initialization started." << std::endl;
  result.initialization = InitializeStereoScene(
      input.measurement_dataset, options_, &result.pair_init_summary,
      &result.optimized_scene);
  result.runtime_summary.initialization_runtime_seconds =
      ElapsedSeconds(initialization_start);
  if (!result.initialization.success) {
    result.failure_reason = result.initialization.failure_reason;
    result.runtime_summary.total_runtime_seconds = ElapsedSeconds(total_start);
    return result;
  }
  result.post_initialization_scene = result.optimized_scene;

  if (options_.board_masking_use_local_board_pose_ba) {
    result.pair_selection_summary = SelectStereoLocalBoardPosePairs(
        input.measurement_dataset, options_, &result.optimized_scene);
  } else {
    result.pair_selection_summary =
        SelectStereoPairs(input.measurement_dataset, result.optimized_scene,
                          result.initialization, options_);
  }
  if (!options_.board_masking_use_local_board_pose_ba &&
      options_.enable_persistent_incremental_stereo_ba) {
    StereoPairSelectionSummary final_selection = result.pair_selection_summary;
    result.pair_board_trial_selection_summary =
        RunPersistentIncrementalPairCohesiveSelection(
            input.measurement_dataset, options_, result.pair_selection_summary,
            &result.pair_trial_selection_summary, &final_selection,
            &result.optimized_scene);
    if (result.pair_board_trial_selection_summary.success) {
      result.pair_selection_summary = final_selection;
      result.pair_selection_summary.selected_pair_board_keys =
          result.pair_board_trial_selection_summary.selected_pair_board_keys;
    } else {
      result.warnings.push_back(
          "persistent_incremental_pair_cohesive_selection_failed: " +
          result.pair_board_trial_selection_summary.failure_reason);
      if (!options_.allow_legacy_selection_fallback_after_persistent_failure) {
        result.failure_reason =
            "persistent_incremental_pair_cohesive_selection_failed: " +
            result.pair_board_trial_selection_summary.failure_reason;
        result.runtime_summary.total_runtime_seconds = ElapsedSeconds(total_start);
        return result;
      }
    }
  }
  if (!options_.board_masking_use_local_board_pose_ba &&
      !result.pair_board_trial_selection_summary.success &&
      (!options_.enable_persistent_incremental_stereo_ba ||
       options_.allow_legacy_selection_fallback_after_persistent_failure) &&
      options_.enable_committing_pair_batch_selection) {
    StereoPairSelectionSummary final_selection = result.pair_selection_summary;
    result.pair_board_trial_selection_summary =
        RunKalibrStyleCommittingPairBatchSelection(
            input.measurement_dataset, options_, result.pair_selection_summary,
            &result.pair_trial_selection_summary, &final_selection,
            &result.optimized_scene);
    if (result.pair_board_trial_selection_summary.success) {
      result.pair_selection_summary = final_selection;
      result.pair_selection_summary.selected_pair_board_keys =
          result.pair_board_trial_selection_summary.selected_pair_board_keys;
    } else {
      result.warnings.push_back(
          "committing_pair_batch_selection_failed: " +
          result.pair_board_trial_selection_summary.failure_reason);
    }
  } else if (!options_.board_masking_use_local_board_pose_ba &&
             !result.pair_board_trial_selection_summary.success &&
             (!options_.enable_persistent_incremental_stereo_ba ||
              options_.allow_legacy_selection_fallback_after_persistent_failure) &&
             options_.enable_kalibr_style_pair_selection) {
    StereoPairSelectionSummary final_selection = result.pair_selection_summary;
    result.pair_trial_selection_summary = RunKalibrStylePairTrialSelection(
        input.measurement_dataset, options_, result.pair_selection_summary,
        &final_selection, &result.optimized_scene);
    if (result.pair_trial_selection_summary.success) {
      result.pair_selection_summary = final_selection;
    } else {
      result.warnings.push_back(
          "kalibr_style_pair_selection_failed: " +
          result.pair_trial_selection_summary.failure_reason);
    }
  }
  if (!options_.board_masking_use_local_board_pose_ba &&
      !result.pair_board_trial_selection_summary.success &&
      !options_.enable_persistent_incremental_stereo_ba &&
      !options_.enable_committing_pair_batch_selection &&
      options_.enable_pair_board_trial_selection) {
    StereoPairSelectionSummary final_selection = result.pair_selection_summary;
    result.pair_board_trial_selection_summary =
        RunKalibrStylePairBoardTrialSelection(
            input.measurement_dataset, options_, result.pair_selection_summary,
            &final_selection, &result.optimized_scene);
    if (result.pair_board_trial_selection_summary.success) {
      result.pair_selection_summary = final_selection;
      result.pair_selection_summary.selected_pair_board_keys =
          result.pair_board_trial_selection_summary.selected_pair_board_keys;
    } else {
      result.warnings.push_back(
          "pair_board_trial_selection_failed: " +
          result.pair_board_trial_selection_summary.failure_reason);
    }
  }

  const Clock::time_point training_optimization_start = Clock::now();
  if (options_.solver_mode == StereoSolverMode::GlobalSparseBa ||
      options_.solver_mode == StereoSolverMode::SharedOnlyGlobalSparseBa) {
    std::cout << "[Stage6] runner: global sparse BA started." << std::endl;
    const std::set<int> selected_pairs(
        result.pair_selection_summary.selected_pair_indices.begin(),
        result.pair_selection_summary.selected_pair_indices.end());
    StereoResidualEvaluator selected_evaluator(
        StereoResidualEvaluationOptions{
            false,
            options_.pair_pose_refit_mode,
            options_.symmetric_refit_max_iterations,
            options_.symmetric_refit_step,
            options_.persistent_pose_structure ==
                StereoPersistentPoseStructure::IndependentPairBoard});
    const StereoMeasurementDataset selected_eval_dataset =
        MakePairBoardMaskedDataset(
            input.measurement_dataset,
            result.pair_selection_summary.selected_pair_board_keys);
    const StereoResidualSummary initial_selected_summary =
        selected_evaluator.Evaluate(selected_eval_dataset, result.optimized_scene,
                                    selected_pairs, "training_selected_initial");
    result.training_selected_initial_residual_summary = initial_selected_summary;
    result.pre_global_sparse_ba_scene = result.optimized_scene;
    result.global_sparse_ba_summary.solver_mode = options_.solver_mode;
    result.global_sparse_ba_summary.residual_mode =
        options_.final_ba_residual_mode;
    result.global_sparse_ba_summary.optimize_intrinsics =
        options_.final_ba_optimize_intrinsics;
    result.global_sparse_ba_summary.eligible_pair_count =
        result.pair_selection_summary.eligible_pair_count;
    result.global_sparse_ba_summary.selected_pair_count =
        result.pair_selection_summary.selected_pair_count;
    result.global_sparse_ba_summary.max_iterations = options_.ba_max_iterations;
    result.global_sparse_ba_summary.convergence_threshold =
        options_.ba_convergence_threshold;
    result.global_sparse_ba_summary.rig_param_mode = options_.rig_param_mode;
    result.global_sparse_ba_summary.rig_camera_prior_translation_weight =
        options_.rig_camera_prior_translation_weight;
    result.global_sparse_ba_summary.rig_camera_prior_rotation_weight =
        options_.rig_camera_prior_rotation_weight;
    result.global_sparse_ba_summary.rig_stereo_relative_prior_weight =
        options_.rig_stereo_relative_prior_weight;
    result.global_sparse_ba_summary.board_masking_use_local_board_pose_ba =
        options_.board_masking_use_local_board_pose_ba;
    result.global_sparse_ba_summary.initial_selected_rmse =
        initial_selected_summary.total_stereo_rmse;
    result.global_sparse_ba_summary.initial_selected_cam0_rmse =
        initial_selected_summary.cam0_rmse;
    result.global_sparse_ba_summary.initial_selected_cam1_rmse =
        initial_selected_summary.cam1_rmse;
    if (options_.skip_final_global_ba) {
      result.global_sparse_ba_summary.success = true;
    result.global_sparse_ba_summary.failure_reason =
          options_.enable_persistent_incremental_stereo_ba
              ? "skip_final_global_ba_after_persistent_incremental_stereo_ba"
              : options_.enable_committing_pair_batch_selection
              ? "skip_final_global_ba_after_incremental_batch_acceptance"
              : "skip_final_global_ba_after_trial_selection";
      result.global_sparse_ba_summary.iterations = 0;
      result.global_sparse_ba_summary.objective_start = 0.0;
      result.global_sparse_ba_summary.objective_final = 0.0;
      result.global_sparse_ba_summary.spherical_weight =
          options_.spherical_weight;
      result.global_sparse_ba_summary.spherical_polar_weighting =
          options_.spherical_polar_weighting;
      result.global_sparse_ba_summary.spherical_min_polar_deg =
          options_.spherical_min_polar_deg;
      result.global_sparse_ba_summary.spherical_max_weight =
          options_.spherical_max_weight;
      result.global_sparse_ba_summary.spherical_uncertainty_mode =
          options_.spherical_uncertainty_mode;
      result.global_sparse_ba_summary.spherical_pixel_sigma_px =
          options_.spherical_pixel_sigma_px;
      result.global_sparse_ba_summary.spherical_model_sigma =
          options_.spherical_model_sigma;
      result.global_sparse_ba_summary.spherical_covariance_damping =
          options_.spherical_covariance_damping;
      result.global_sparse_ba_summary.spherical_min_sigma_rad =
          options_.spherical_min_sigma_rad;
      result.global_sparse_ba_summary.spherical_max_whitening_weight =
          options_.spherical_max_whitening_weight;
      result.global_sparse_ba_summary.spherical_use_normalize_jacobian =
          options_.spherical_use_normalize_jacobian;
      const Eigen::Isometry3d initial_T_cam1_cam0 =
          ToIsometry3d(result.pre_global_sparse_ba_scene.T_cam1_cam0);
      const Eigen::Isometry3d final_T_cam1_cam0 =
          ToIsometry3d(result.optimized_scene.T_cam1_cam0);
      result.global_sparse_ba_summary.rig_stereo_relative_rotation_drift_deg =
          RotationDistanceRadians(initial_T_cam1_cam0, final_T_cam1_cam0) *
          180.0 / M_PI;
      result.global_sparse_ba_summary.rig_stereo_relative_translation_drift_m =
          (final_T_cam1_cam0.translation() -
           initial_T_cam1_cam0.translation()).norm();
    } else if (!RunGlobalSparseBa(input.measurement_dataset, options_,
                                  result.pair_selection_summary,
                                  &result.global_sparse_ba_summary,
                                  &result.optimized_scene)) {
      result.failure_reason =
          result.global_sparse_ba_summary.failure_reason.empty()
              ? "Stage6 global sparse BA failed."
              : result.global_sparse_ba_summary.failure_reason;
      result.runtime_summary.total_runtime_seconds = ElapsedSeconds(total_start);
      return result;
    }
    const StereoResidualSummary final_selected_summary =
        selected_evaluator.Evaluate(selected_eval_dataset, result.optimized_scene,
                                    selected_pairs, "training_selected_final");
    result.training_selected_final_residual_summary = final_selected_summary;
    result.global_sparse_ba_summary.final_selected_rmse =
        final_selected_summary.total_stereo_rmse;
    result.global_sparse_ba_summary.final_selected_cam0_rmse =
        final_selected_summary.cam0_rmse;
    result.global_sparse_ba_summary.final_selected_cam1_rmse =
        final_selected_summary.cam1_rmse;
    result.global_sparse_ba_summary.success = true;
    result.runtime_summary.global_sparse_ba_runtime_seconds =
        ElapsedSeconds(training_optimization_start);
    result.runtime_summary.training_optimization_runtime_seconds =
        result.runtime_summary.global_sparse_ba_runtime_seconds;
  } else {
    std::cout << "[Stage6] runner: alternating optimization started." << std::endl;
    if (!OptimizeSceneAlternating(input.measurement_dataset, options_,
                                  &runtime_counters,
                                  &result.optimized_scene)) {
      result.failure_reason = "Stereo alternating optimization failed.";
      result.runtime_summary.total_runtime_seconds = ElapsedSeconds(total_start);
      return result;
    }
    result.runtime_summary.training_optimization_runtime_seconds =
        ElapsedSeconds(training_optimization_start);
  }

  StereoResidualEvaluator training_evaluator(
      StereoResidualEvaluationOptions{
          false,
          options_.pair_pose_refit_mode,
          options_.symmetric_refit_max_iterations,
          options_.symmetric_refit_step,
          options_.persistent_pose_structure ==
              StereoPersistentPoseStructure::IndependentPairBoard,
          options_.persistent_pose_structure ==
              StereoPersistentPoseStructure::IndependentPairBoard});
  const bool holdout_uses_local_stereo_board_refit =
      options_.board_masking_use_local_board_pose_ba;
  StereoResidualEvaluator holdout_evaluator(
      StereoResidualEvaluationOptions{
          !holdout_uses_local_stereo_board_refit,
          options_.pair_pose_refit_mode,
          options_.symmetric_refit_max_iterations,
          options_.symmetric_refit_step,
          holdout_uses_local_stereo_board_refit});
  StereoResidualEvaluator holdout_extrinsic_only_evaluator(
      StereoResidualEvaluationOptions{
          false,
          options_.pair_pose_refit_mode,
          options_.symmetric_refit_max_iterations,
          options_.symmetric_refit_step,
          true});
  const StereoMeasurementDataset training_eval_dataset =
      MakePairBoardMaskedDataset(
          input.measurement_dataset,
          result.pair_selection_summary.selected_pair_board_keys);
  result.training_residual_summary =
      training_evaluator.Evaluate(
          training_eval_dataset, result.optimized_scene,
          std::set<int>(training_eval_dataset.training_pair_indices.begin(),
                        training_eval_dataset.training_pair_indices.end()),
          "training");
  std::cout << "[Stage6] runner: training evaluation done. total_rmse="
            << result.training_residual_summary.total_stereo_rmse
            << ", used_pairs="
            << result.training_residual_summary.used_pair_count << std::endl;
  const Clock::time_point holdout_start = Clock::now();
  std::cout << "[Stage6] runner: holdout evaluation started." << std::endl;
  result.holdout_residual_summary =
      holdout_evaluator.Evaluate(
          input.measurement_dataset, result.optimized_scene,
          std::set<int>(input.measurement_dataset.holdout_pair_indices.begin(),
                        input.measurement_dataset.holdout_pair_indices.end()),
          "holdout");
  result.holdout_extrinsic_only_residual_summary =
      holdout_extrinsic_only_evaluator.Evaluate(
          input.measurement_dataset, result.optimized_scene,
          std::set<int>(input.measurement_dataset.holdout_pair_indices.begin(),
                        input.measurement_dataset.holdout_pair_indices.end()),
          "holdout_extrinsic_only");
  result.runtime_summary.holdout_evaluation_runtime_seconds =
      ElapsedSeconds(holdout_start);
  std::cout << "[Stage6] runner: holdout extrinsic-only evaluation done. total_rmse="
            << result.holdout_extrinsic_only_residual_summary.total_stereo_rmse
            << ", used_pairs="
            << result.holdout_extrinsic_only_residual_summary.used_pair_count
            << std::endl;
  result.pair_board_consistency_summary =
      BuildPairBoardConsistencySummary(input.measurement_dataset,
                                       result.optimized_scene, options_);
  result.runtime_summary.symmetric_refit_call_count =
      runtime_counters.symmetric_refit_call_count;
  result.runtime_summary.symmetric_refit_improved_count =
      runtime_counters.symmetric_refit_improved_count;
  result.runtime_summary.symmetric_refit_fallback_count =
      runtime_counters.symmetric_refit_fallback_count;
  result.runtime_summary.max_graph_propagation_iterations =
      input.solver_options.max_graph_propagation_iterations;
  result.runtime_summary.graph_propagation_iteration_count =
      result.initialization.graph_propagation_iteration_count;
  result.runtime_summary.graph_propagation_new_pair_count =
      result.initialization.graph_propagation_new_pair_count;
  result.runtime_summary.graph_propagation_new_board_count =
      result.initialization.graph_propagation_new_board_count;
  result.runtime_summary.graph_propagation_stopped_by_no_progress =
      result.initialization.graph_propagation_stopped_by_no_progress;
  result.runtime_summary.graph_propagation_stopped_by_iteration_limit =
      result.initialization.graph_propagation_stopped_by_iteration_limit;
  result.runtime_summary.runtime_guard_trigger_count =
      runtime_counters.runtime_guard_trigger_count;

  result.success = result.training_residual_summary.success;
  if (!result.success) {
    result.failure_reason = result.training_residual_summary.failure_reason;
  }
  if (options_.export_extrinsic_uncertainty_diagnostics) {
    result.extrinsic_uncertainty_summary =
        ComputeExtrinsicUncertaintySummary(result.pair_init_summary,
                                           result.optimized_scene);
  }
  result.runtime_summary.total_runtime_seconds = ElapsedSeconds(total_start);
  return result;
}

void WriteStereoExtrinsicYaml(const std::string& path,
                              const StereoExtrinsicCalibrationResult& result) {
  std::ofstream output(path.c_str());
  const Eigen::Isometry3d T = ToIsometry3d(result.optimized_scene.T_cam1_cam0);
  const Eigen::Quaterniond q(T.linear());
  output << "cam0_is_reference: 1\n";
  output << "camera0_model_family: " << result.optimized_scene.cam0.camera_model_family << "\n";
  output << "camera1_model_family: " << result.optimized_scene.cam1.camera_model_family << "\n";
  output << "gauge_fixed_board_id: " << result.optimized_scene.gauge_fixed_board_id << "\n";
  output << "translation_xyz: [" << T.translation().x() << ", " << T.translation().y()
         << ", " << T.translation().z() << "]\n";
  output << "quaternion_wxyz: [" << q.w() << ", " << q.x() << ", " << q.y()
         << ", " << q.z() << "]\n";
  output << "baseline_length: " << T.translation().norm() << "\n";
  output << "rotation_matrix:\n";
  for (int row = 0; row < 3; ++row) {
    output << "  - [" << T.matrix()(row, 0) << ", " << T.matrix()(row, 1) << ", "
           << T.matrix()(row, 2) << "]\n";
  }
  output << "solver_mode: " << ToString(result.problem_input.solver_options.solver_mode)
         << "\n";
  output << "selected_pair_count: "
         << result.pair_selection_summary.selected_pair_count << "\n";
}

void WriteStereoFinalCameraYaml(
    const std::string& path,
    const StereoExtrinsicCalibrationResult& result,
    int camera_index) {
  if (camera_index != 0 && camera_index != 1) {
    throw std::invalid_argument("camera_index must be 0 (left) or 1 (right)");
  }
  const StereoCameraFixedCalibration& camera =
      camera_index == 0 ? result.optimized_scene.cam0 : result.optimized_scene.cam1;
  if (!camera.IsValid()) {
    throw std::runtime_error("cannot export invalid Stage6 final camera calibration");
  }

  std::ofstream output(path.c_str());
  if (!output.is_open()) {
    throw std::runtime_error("cannot open Stage6 final camera YAML: " + path);
  }
  output << std::setprecision(17);
  output << "# Stage6 final in-process calibration. Do not combine with an external "
            "stereo_extrinsic.yaml.\n";
  output << "# camera_role: " << (camera_index == 0 ? "left" : "right") << "\n";
  output << "# seed_source: " << camera.source_label << "\n";
  output << "cam0:\n";
  output << "  cam_overlaps: []\n";
  output << "  camera_model: " << camera.camera_model << "\n";
  output << "  distortion_model: " << camera.distortion_model << "\n";
  output << "  intrinsics: [";
  for (std::size_t index = 0; index < camera.intrinsics.size(); ++index) {
    if (index != 0u) {
      output << ", ";
    }
    output << camera.intrinsics[index];
  }
  output << "]\n";
  output << "  distortion_coeffs: [";
  for (std::size_t index = 0; index < camera.distortion_coeffs.size(); ++index) {
    if (index != 0u) {
      output << ", ";
    }
    output << camera.distortion_coeffs[index];
  }
  output << "]\n";
  output << "  resolution: [" << camera.resolution[0] << ", "
         << camera.resolution[1] << "]\n";
}

void WriteStereoExtrinsicSummary(const std::string& path,
                                 const StereoExtrinsicCalibrationResult& result) {
  std::ofstream output(path.c_str());
  const Eigen::Isometry3d T = ToIsometry3d(result.optimized_scene.T_cam1_cam0);
  const Eigen::Quaterniond q(T.linear());
  const double rotation_angle_deg =
      2.0 * std::acos(std::max(-1.0, std::min(1.0, q.w()))) * 180.0 / M_PI;
  output << "success: " << (result.success ? 1 : 0) << "\n";
  output << "failure_reason: " << result.failure_reason << "\n";
  output << "left_image_path: " << result.problem_input.left_image_path << "\n";
  output << "right_image_path: " << result.problem_input.right_image_path << "\n";
  output << "paired_frame_count: "
         << result.problem_input.measurement_dataset.paired_frame_count << "\n";
  output << "unmatched_left_count: "
         << result.problem_input.measurement_dataset.unmatched_left_count << "\n";
  output << "unmatched_right_count: "
         << result.problem_input.measurement_dataset.unmatched_right_count << "\n";
  output << "shared_board_observation_count: "
         << result.problem_input.measurement_dataset.shared_board_observation_count << "\n";
  output << "cam0_only_board_observation_count: "
         << result.problem_input.measurement_dataset.cam0_only_board_observation_count << "\n";
  output << "cam1_only_board_observation_count: "
         << result.problem_input.measurement_dataset.cam1_only_board_observation_count << "\n";
  output << "initial_candidate_count: " << result.initialization.candidate_count << "\n";
  output << "excluded_candidate_count: "
         << result.initialization.excluded_candidate_count << "\n";
  output << "graph_seed_pair_count: "
         << result.initialization.graph_seed_pair_count << "\n";
  output << "reachable_training_pair_count: "
         << result.initialization.reachable_training_pair_count << "\n";
  output << "unreachable_training_pair_count: "
         << result.initialization.unreachable_training_pair_count << "\n";
  output << "excluded_training_pair_count: "
         << result.initialization.excluded_training_pair_count << "\n";
  output << "initialized_training_pair_count: "
         << result.initialization.initialized_training_pair_count << "\n";
  output << "initialized_board_count: "
         << result.initialization.initialized_board_count << "\n";
  output << "uninitialized_training_pair_count: "
         << result.initialization.uninitialized_training_pair_count << "\n";
  output << "graph_propagation_iteration_count: "
         << result.initialization.graph_propagation_iteration_count << "\n";
  output << "pair_pose_refit_mode: "
         << StereoPairPoseRefitModeToString(
                result.problem_input.solver_options.pair_pose_refit_mode) << "\n";
  output << "pose_fit_guard_threshold_px: "
         << result.problem_input.solver_options.pose_fit_guard_threshold_px << "\n";
  output << "shared_board_quality_gate_enabled: "
         << (result.problem_input.solver_options.enable_shared_board_quality_gate ? 1 : 0)
         << "\n";
  output << "shared_board_quality_hard_gate_enabled: "
         << (result.problem_input.solver_options.enable_shared_board_quality_hard_gate ? 1 : 0)
         << "\n";
  output << "shared_board_quality_max_outer_rmse_px: "
         << result.problem_input.solver_options.shared_board_quality_max_outer_rmse_px
         << "\n";
  output << "shared_board_quality_min_outer_points_per_camera: "
         << result.problem_input.solver_options
                .shared_board_quality_min_outer_points_per_camera
         << "\n";
  output << "shared_board_quality_min_good_shared_boards: "
         << result.problem_input.solver_options.shared_board_quality_min_good_shared_boards
         << "\n";
  output << "shared_board_quality_filter_final_ba: "
         << (result.problem_input.solver_options.shared_board_quality_filter_final_ba ? 1 : 0)
         << "\n";
  output << "candidate_consistency_max_rotation_deg: "
         << result.problem_input.solver_options.candidate_consistency_max_rotation_deg << "\n";
  output << "candidate_consistency_max_translation_m: "
         << result.problem_input.solver_options.candidate_consistency_max_translation_m << "\n";
  output << "gauge_fixed_board_id: "
         << result.optimized_scene.gauge_fixed_board_id << "\n";
  output << "cam0_is_reference: 1\n";
  output << "translation_xyz: [" << T.translation().x() << ", "
         << T.translation().y() << ", " << T.translation().z() << "]\n";
  output << "quaternion_wxyz: [" << q.w() << ", " << q.x() << ", " << q.y()
         << ", " << q.z() << "]\n";
  output << "rotation_angle_deg: " << rotation_angle_deg << "\n";
  output << "baseline_length: " << T.translation().norm() << "\n";
  output << "solver_mode: " << ToString(result.problem_input.solver_options.solver_mode)
         << "\n";
  output << "selected_pair_count: "
         << result.pair_selection_summary.selected_pair_count << "\n";
  output << "eligible_pair_count: "
         << result.pair_selection_summary.eligible_pair_count << "\n";
}

void WriteStereoReprojectionSummary(const std::string& path,
                                    const StereoExtrinsicCalibrationResult& result) {
  std::ofstream output(path.c_str());
  const auto write_split = [&output](const StereoResidualSummary& summary,
                                     const std::string& prefix) {
    output << prefix << "_success: " << (summary.success ? 1 : 0) << "\n";
    output << prefix << "_failure_reason: " << summary.failure_reason << "\n";
    output << prefix << "_pair_count: " << summary.pair_count << "\n";
    output << prefix << "_used_pair_count: " << summary.used_pair_count << "\n";
    output << prefix << "_unevaluable_pair_count: " << summary.unevaluable_pair_count << "\n";
    output << prefix << "_shared_board_pair_count: "
           << summary.shared_board_pair_count << "\n";
    output << prefix << "_single_camera_only_pair_count: "
           << summary.single_camera_only_pair_count << "\n";
    output << prefix << "_total_stereo_rmse: " << summary.total_stereo_rmse << "\n";
    output << prefix << "_cam0_rmse: " << summary.cam0_rmse << "\n";
    output << prefix << "_cam1_rmse: " << summary.cam1_rmse << "\n";
    output << prefix << "_cam1_over_cam0_rmse_ratio: "
           << summary.cam1_over_cam0_rmse_ratio << "\n";
    output << prefix << "_cam_residual_balance_gap: "
           << summary.cam_residual_balance_gap << "\n";
    output << prefix << "_outer_only_rmse: " << summary.outer_only_rmse << "\n";
    output << prefix << "_internal_only_rmse: " << summary.internal_only_rmse << "\n";
    output << prefix << "_shared_point_count: " << summary.shared_point_count << "\n";
    output << prefix << "_shared_outer_point_count: "
           << summary.shared_outer_point_count << "\n";
    output << prefix << "_shared_internal_point_count: "
           << summary.shared_internal_point_count << "\n";
    output << prefix << "_shared_total_rmse: " << summary.shared_total_rmse << "\n";
    output << prefix << "_shared_cam0_rmse: " << summary.shared_cam0_rmse << "\n";
    output << prefix << "_shared_cam1_rmse: " << summary.shared_cam1_rmse << "\n";
    output << prefix << "_cam0_only_point_count: " << summary.cam0_only_point_count << "\n";
    output << prefix << "_cam1_only_point_count: " << summary.cam1_only_point_count << "\n";
    output << prefix << "_cam0_only_total_rmse: " << summary.cam0_only_total_rmse << "\n";
    output << prefix << "_cam1_only_total_rmse: " << summary.cam1_only_total_rmse << "\n";
    output << prefix << "_mean_residual_x: " << summary.mean_residual_x << "\n";
    output << prefix << "_mean_residual_y: " << summary.mean_residual_y << "\n";
    output << prefix << "_std_residual_x: " << summary.std_residual_x << "\n";
    output << prefix << "_std_residual_y: " << summary.std_residual_y << "\n";
  };
  write_split(result.training_residual_summary, "training");
  write_split(result.holdout_extrinsic_only_residual_summary,
              "holdout_extrinsic_only");
  output << "holdout_extrinsic_only_mode: local_stereo_board_pose_refit\n";
  output << "holdout_extrinsic_only_diagnostic: local_per_pair_board_pose_refit_tests_stereo_extrinsic_with_layout_removed\n";
  output << "solver_mode: " << ToString(result.problem_input.solver_options.solver_mode)
         << "\n";
  output << "selected_pair_count: "
         << result.pair_selection_summary.selected_pair_count << "\n";
  output << "eligible_pair_count: "
         << result.pair_selection_summary.eligible_pair_count << "\n";
}

void WriteStereoPerCameraResidualsCsv(const std::string& path,
                                      const StereoExtrinsicCalibrationResult& result) {
  std::ofstream output(path.c_str());
  output << "split,camera_index,point_count,rmse,shared_point_count,shared_rmse,cam0_only_point_count,cam0_only_rmse,cam1_only_point_count,cam1_only_rmse\n";
  for (const StereoCameraResidualSummary& summary :
       result.training_residual_summary.camera_summaries) {
    output << "training," << summary.camera_index << "," << summary.point_count
           << "," << summary.rmse
           << "," << summary.shared_point_count
           << "," << summary.shared_rmse
           << "," << summary.cam0_only_point_count
           << "," << summary.cam0_only_rmse
           << "," << summary.cam1_only_point_count
           << "," << summary.cam1_only_rmse << "\n";
  }
  for (const StereoCameraResidualSummary& summary :
       result.holdout_extrinsic_only_residual_summary.camera_summaries) {
    output << "holdout_extrinsic_only," << summary.camera_index << ","
           << summary.point_count
           << "," << summary.rmse
           << "," << summary.shared_point_count
           << "," << summary.shared_rmse
           << "," << summary.cam0_only_point_count
           << "," << summary.cam0_only_rmse
           << "," << summary.cam1_only_point_count
           << "," << summary.cam1_only_rmse << "\n";
  }
}

void WriteStereoPerFrameResidualsCsv(const std::string& path,
                                     const StereoExtrinsicCalibrationResult& result) {
  std::ofstream output(path.c_str());
  output << "split,pair_index,left_frame_label,right_frame_label,is_training,used_in_metrics,"
         << "pose_refit_mode,pose_refit_success,used_symmetric_refit,refit_fell_back_to_seed,"
         << "shared_board_count,cam0_only_board_count,cam1_only_board_count,pose_source,"
         << "point_count,outer_point_count,internal_point_count,cam0_point_count,cam1_point_count,"
         << "shared_point_count,shared_outer_point_count,shared_internal_point_count,"
         << "overall_rmse,cam0_rmse,cam1_rmse,outer_rmse,internal_rmse,"
         << "shared_cam0_rmse,shared_cam1_rmse,shared_outer_rmse,shared_internal_rmse,"
         << "cam0_only_rmse,cam1_only_rmse,"
         << "mean_residual_x,mean_residual_y,std_residual_x,std_residual_y,failure_reason\n";
  for (const StereoPairResidualSummary& summary :
       result.training_residual_summary.pair_summaries) {
    output << "training," << summary.pair_index << "," << summary.left_frame_label << ","
           << summary.right_frame_label << "," << (summary.is_training ? 1 : 0) << ","
           << (summary.used_in_metrics ? 1 : 0) << ","
           << StereoPairPoseRefitModeToString(
                  result.problem_input.solver_options.pair_pose_refit_mode) << ","
           << (summary.pose_refit_success ? 1 : 0) << ","
           << (summary.used_symmetric_refit ? 1 : 0) << ","
           << (summary.refit_fell_back_to_seed ? 1 : 0) << ","
           << summary.shared_board_count << ","
           << summary.cam0_only_board_count << ","
           << summary.cam1_only_board_count << ","
           << summary.pose_source << ","
           << summary.point_count << ","
           << summary.outer_point_count << "," << summary.internal_point_count << ","
           << summary.cam0_point_count << "," << summary.cam1_point_count << ","
           << summary.shared_point_count << "," << summary.shared_outer_point_count << ","
           << summary.shared_internal_point_count << ","
           << summary.overall_rmse << "," << summary.cam0_rmse << ","
           << summary.cam1_rmse << "," << summary.outer_rmse << ","
           << summary.internal_rmse << "," << summary.shared_cam0_rmse << ","
           << summary.shared_cam1_rmse << "," << summary.shared_outer_rmse << ","
           << summary.shared_internal_rmse << "," << summary.cam0_only_rmse << ","
           << summary.cam1_only_rmse << "," << summary.mean_residual_x << ","
           << summary.mean_residual_y << "," << summary.std_residual_x << ","
           << summary.std_residual_y << "," << summary.failure_reason << "\n";
  }
  for (const StereoPairResidualSummary& summary :
       result.holdout_extrinsic_only_residual_summary.pair_summaries) {
    output << "holdout_extrinsic_only," << summary.pair_index << ","
           << summary.left_frame_label << ","
           << summary.right_frame_label << "," << (summary.is_training ? 1 : 0)
           << ","
           << (summary.used_in_metrics ? 1 : 0) << ","
           << StereoPairPoseRefitModeToString(
                  result.problem_input.solver_options.pair_pose_refit_mode)
           << ","
           << (summary.pose_refit_success ? 1 : 0) << ","
           << (summary.used_symmetric_refit ? 1 : 0) << ","
           << (summary.refit_fell_back_to_seed ? 1 : 0) << ","
           << summary.shared_board_count << ","
           << summary.cam0_only_board_count << ","
           << summary.cam1_only_board_count << ","
           << summary.pose_source << ","
           << summary.point_count << ","
           << summary.outer_point_count << "," << summary.internal_point_count
           << ","
           << summary.cam0_point_count << "," << summary.cam1_point_count
           << ","
           << summary.shared_point_count << ","
           << summary.shared_outer_point_count << ","
           << summary.shared_internal_point_count << ","
           << summary.overall_rmse << "," << summary.cam0_rmse << ","
           << summary.cam1_rmse << "," << summary.outer_rmse << ","
           << summary.internal_rmse << "," << summary.shared_cam0_rmse << ","
           << summary.shared_cam1_rmse << "," << summary.shared_outer_rmse
           << ","
           << summary.shared_internal_rmse << "," << summary.cam0_only_rmse
           << ","
           << summary.cam1_only_rmse << "," << summary.mean_residual_x << ","
           << summary.mean_residual_y << "," << summary.std_residual_x << ","
           << summary.std_residual_y << "," << summary.failure_reason << "\n";
  }
}

void WriteStereoHoldoutLayoutTransferGapCsv(
    const std::string& path,
    const StereoExtrinsicCalibrationResult& result) {
  std::ofstream output(path.c_str());
  output << "pair_index,left_frame_label,right_frame_label,shared_board_count,"
         << "normal_pose_source,normal_used_in_metrics,"
         << "extrinsic_only_pose_source,extrinsic_only_used_in_metrics,"
         << "normal_rmse,extrinsic_only_rmse,layout_transfer_gap_rmse,"
         << "normal_outer_rmse,extrinsic_only_outer_rmse,"
         << "layout_transfer_gap_outer_rmse,"
         << "normal_internal_rmse,extrinsic_only_internal_rmse,"
         << "layout_transfer_gap_internal_rmse,"
         << "normal_point_count,extrinsic_only_point_count,"
         << "normal_failure_reason,extrinsic_only_failure_reason\n";
  std::map<int, const StereoPairResidualSummary*> extrinsic_by_pair;
  for (const StereoPairResidualSummary& summary :
       result.holdout_extrinsic_only_residual_summary.pair_summaries) {
    extrinsic_by_pair[summary.pair_index] = &summary;
  }
  for (const StereoPairResidualSummary& normal :
       result.holdout_residual_summary.pair_summaries) {
    const auto extrinsic_it = extrinsic_by_pair.find(normal.pair_index);
    const StereoPairResidualSummary* extrinsic =
        extrinsic_it == extrinsic_by_pair.end() ? nullptr : extrinsic_it->second;
    const auto gap = [extrinsic](const StereoPairResidualSummary& lhs,
                                 double StereoPairResidualSummary::*member) {
      if (extrinsic == nullptr) {
        return 0.0;
      }
      const double lhs_value = lhs.*member;
      const double rhs_value = extrinsic->*member;
      if (!std::isfinite(lhs_value) || !std::isfinite(rhs_value)) {
        return 0.0;
      }
      return lhs_value - rhs_value;
    };
    output << normal.pair_index << "," << normal.left_frame_label << ","
           << normal.right_frame_label << "," << normal.shared_board_count
           << "," << normal.pose_source << ","
           << (normal.used_in_metrics ? 1 : 0) << ",";
    if (extrinsic != nullptr) {
      output << extrinsic->pose_source << ","
             << (extrinsic->used_in_metrics ? 1 : 0) << ","
             << normal.overall_rmse << "," << extrinsic->overall_rmse << ","
             << gap(normal, &StereoPairResidualSummary::overall_rmse) << ","
             << normal.outer_rmse << "," << extrinsic->outer_rmse << ","
             << gap(normal, &StereoPairResidualSummary::outer_rmse) << ","
             << normal.internal_rmse << "," << extrinsic->internal_rmse << ","
             << gap(normal, &StereoPairResidualSummary::internal_rmse) << ","
             << normal.point_count << "," << extrinsic->point_count << ","
             << normal.failure_reason << ","
             << extrinsic->failure_reason << "\n";
    } else {
      output << "missing,0," << normal.overall_rmse << ",0,0,"
             << normal.outer_rmse << ",0,0," << normal.internal_rmse
             << ",0,0," << normal.point_count << ",0,"
             << normal.failure_reason << ",missing_extrinsic_only_pair\n";
    }
  }
}

void WriteStereoHoldoutLocalLayoutDriftCsv(
    const std::string& path,
    const StereoExtrinsicCalibrationResult& result) {
  std::ofstream output(path.c_str());
  output << std::setprecision(12);
  output << "pair_index,left_frame_label,right_frame_label,board_id,"
         << "shared_board_count,normal_pair_pose_success,"
         << "local_board_pose_success,global_board_pose_available,"
         << "normal_pose_source,local_pose_source,"
         << "translation_drift_m,rotation_drift_deg,"
         << "global_outer_rmse_px,local_outer_rmse_px,"
         << "global_outer_point_count,local_outer_point_count,"
         << "failure_reason\n";

  const StereoMeasurementDataset& dataset =
      result.problem_input.measurement_dataset;
  const StereoSceneState& scene_state = result.optimized_scene;
  std::map<int, const StereoPairResidualSummary*> holdout_summary_by_pair;
  for (const StereoPairResidualSummary& summary :
       result.holdout_residual_summary.pair_summaries) {
    holdout_summary_by_pair[summary.pair_index] = &summary;
  }
  const DoubleSphereCameraModel cam0 =
      DoubleSphereCameraModel::FromConfig(MakeCameraConfig(scene_state.cam0));
  const DoubleSphereCameraModel cam1 =
      DoubleSphereCameraModel::FromConfig(MakeCameraConfig(scene_state.cam1));
  const Eigen::Isometry3d T_cam1_cam0 =
      ToIsometry3d(scene_state.T_cam1_cam0);
  const auto evaluate_global_outer_rmse_with_pose =
      [&dataset, &scene_state, &cam0, &cam1, &T_cam1_cam0](
          int pair_index, int board_id,
          const Eigen::Isometry3d& T_cam0_world,
          int* point_count) -> double {
    if (point_count != nullptr) {
      *point_count = 0;
    }
    const auto board_pose_it = scene_state.T_world_board_by_id.find(board_id);
    if (board_pose_it == scene_state.T_world_board_by_id.end() ||
        !cam0.IsValid() || !cam1.IsValid()) {
      return std::numeric_limits<double>::infinity();
    }
    const Eigen::Isometry3d T_world_board =
        ToIsometry3d(board_pose_it->second);
    double squared_error_sum = 0.0;
    int count = 0;
    for (const StereoObservation& observation : dataset.observations) {
      if (observation.pair_index != pair_index ||
          observation.board_id != board_id ||
          observation.point_type != JointPointType::Outer ||
          !observation.used_in_solver) {
        continue;
      }
      const Eigen::Vector3d point_cam0 =
          T_cam0_world * (T_world_board * observation.target_point_board);
      Eigen::Vector2d predicted = Eigen::Vector2d::Zero();
      const bool ok =
          observation.camera_index == 0
              ? cam0.vsEuclideanToKeypoint(point_cam0, &predicted)
              : cam1.vsEuclideanToKeypoint(T_cam1_cam0 * point_cam0,
                                           &predicted);
      if (!ok) {
        continue;
      }
      const Eigen::Vector2d residual =
          predicted - observation.observed_image_xy;
      if (!residual.allFinite()) {
        continue;
      }
      squared_error_sum += residual.squaredNorm();
      ++count;
    }
    if (point_count != nullptr) {
      *point_count = count;
    }
    return count > 0 ? std::sqrt(squared_error_sum /
                                 static_cast<double>(count))
                     : std::numeric_limits<double>::infinity();
  };

  for (int pair_index : dataset.holdout_pair_indices) {
    const StereoFramePair* pair = FindPair(dataset, pair_index);
    const std::string left_frame_label =
        pair == nullptr ? std::string() : pair->left_frame_label;
    const std::string right_frame_label =
        pair == nullptr ? std::string() : pair->right_frame_label;
    const auto shared_it = dataset.pair_shared_board_ids.find(pair_index);
    if (shared_it == dataset.pair_shared_board_ids.end()) {
      continue;
    }
    const auto summary_it = holdout_summary_by_pair.find(pair_index);
    const StereoPairResidualSummary* pair_summary =
        summary_it == holdout_summary_by_pair.end() ? nullptr
                                                    : summary_it->second;

    Eigen::Matrix4d T_cam0_world_matrix = Eigen::Matrix4d::Identity();
    StereoRefitDiagnostics diagnostics;
    RuntimeCounters runtime_counters;
    std::string normal_failure_reason;
    bool normal_pair_pose_success = RefitPairPoseFromStereoOuterObservations(
        dataset, scene_state, pair_index, result.problem_input.solver_options,
        &runtime_counters, &diagnostics, &T_cam0_world_matrix);
    if (!normal_pair_pose_success) {
      normal_failure_reason = "pair_refit_failed";
      const auto pair_pose_it = scene_state.T_cam0_world_by_pair.find(pair_index);
      if (pair_pose_it != scene_state.T_cam0_world_by_pair.end()) {
        T_cam0_world_matrix = pair_pose_it->second;
        normal_failure_reason = "pair_refit_failed_using_optimized_scene_seed:" +
                                normal_failure_reason;
      }
    }

    const Eigen::Isometry3d T_cam0_world =
        ToIsometry3d(T_cam0_world_matrix);
    for (int board_id : shared_it->second) {
      const auto board_pose_it = scene_state.T_world_board_by_id.find(board_id);
      const bool have_global_board_pose =
          board_pose_it != scene_state.T_world_board_by_id.end();

      Eigen::Matrix4d T_cam0_board_local_matrix = Eigen::Matrix4d::Identity();
      const bool local_board_pose_success = RefitStereoBoardPoseForVisualization(
          dataset, scene_state, pair_index, board_id,
          result.problem_input.solver_options.symmetric_refit_max_iterations,
          result.problem_input.solver_options.symmetric_refit_step,
          &T_cam0_board_local_matrix);

      double translation_drift_m =
          std::numeric_limits<double>::quiet_NaN();
      double rotation_drift_deg =
          std::numeric_limits<double>::quiet_NaN();
      double global_outer_rmse_px =
          std::numeric_limits<double>::quiet_NaN();
      double local_outer_rmse_px =
          std::numeric_limits<double>::quiet_NaN();
      int global_outer_point_count = 0;
      int local_outer_point_count = 0;
      std::string failure_reason;

      if (!normal_pair_pose_success) {
        failure_reason += "normal_pair_pose_refit_failed;";
      }
      if (!have_global_board_pose) {
        failure_reason += "missing_global_board_pose;";
      }
      if (!local_board_pose_success) {
        failure_reason += "local_board_pose_refit_failed;";
      }

      if (have_global_board_pose) {
        global_outer_rmse_px = evaluate_global_outer_rmse_with_pose(
            pair_index, board_id, T_cam0_world, &global_outer_point_count);
      }
      if (local_board_pose_success) {
        const Eigen::Isometry3d T_cam0_board_local =
            ToIsometry3d(T_cam0_board_local_matrix);
        const std::vector<StereoBoardPoseObservation> local_observations =
            CollectStereoBoardPoseObservations(dataset, pair_index, board_id);
        local_outer_rmse_px = EvaluateStereoBoardPoseRmse(
            local_observations, scene_state, T_cam0_board_local);
        local_outer_point_count = static_cast<int>(local_observations.size());
      }
      if (have_global_board_pose && local_board_pose_success) {
        const Eigen::Isometry3d T_cam0_board_global =
            T_cam0_world * ToIsometry3d(board_pose_it->second);
        const Eigen::Isometry3d T_cam0_board_local =
            ToIsometry3d(T_cam0_board_local_matrix);
        translation_drift_m =
            (T_cam0_board_local.translation() -
             T_cam0_board_global.translation()).norm();
        rotation_drift_deg =
            RotationDistanceRadians(T_cam0_board_global,
                                    T_cam0_board_local) *
            180.0 / M_PI;
      }

      output << pair_index << "," << left_frame_label << ","
             << right_frame_label << "," << board_id << ","
             << (pair_summary == nullptr ? 0 : pair_summary->shared_board_count)
             << "," << (normal_pair_pose_success ? 1 : 0) << ","
             << (local_board_pose_success ? 1 : 0) << ","
             << (have_global_board_pose ? 1 : 0) << ","
             << (pair_summary == nullptr ? std::string() : pair_summary->pose_source)
             << ",local_stereo_board_pose_refit,"
             << translation_drift_m << "," << rotation_drift_deg << ","
             << global_outer_rmse_px << "," << local_outer_rmse_px << ","
             << global_outer_point_count << "," << local_outer_point_count
             << ",";
      if (failure_reason.empty()) {
        output << normal_failure_reason;
      } else {
        output << failure_reason << normal_failure_reason;
      }
      output << "\n";
    }
  }
}

void WriteStereoBaFrameFactorTraceCsv(
    const std::string& path,
    const StereoExtrinsicCalibrationResult& result) {
  struct BoardFactorStats {
    int cam0_factor_count = 0;
    int cam1_factor_count = 0;
    int outer_factor_count = 0;
    int internal_factor_count = 0;
  };

  struct FactorStats {
    int cam0_factor_count = 0;
    int cam1_factor_count = 0;
    int outer_factor_count = 0;
    int internal_factor_count = 0;
    std::map<int, BoardFactorStats> board_factor_stats;
  };

  struct PairBoardTraceResidualStats {
    int point_count = 0;
    int cam0_point_count = 0;
    int cam1_point_count = 0;
    double squared_error_sum = 0.0;
    double cam0_squared_error_sum = 0.0;
    double cam1_squared_error_sum = 0.0;

    double Rmse() const {
      return point_count <= 0
                 ? std::numeric_limits<double>::quiet_NaN()
                 : std::sqrt(squared_error_sum /
                             static_cast<double>(point_count));
    }
    double Cam0Rmse() const {
      return cam0_point_count <= 0
                 ? std::numeric_limits<double>::quiet_NaN()
                 : std::sqrt(cam0_squared_error_sum /
                             static_cast<double>(cam0_point_count));
    }
    double Cam1Rmse() const {
      return cam1_point_count <= 0
                 ? std::numeric_limits<double>::quiet_NaN()
                 : std::sqrt(cam1_squared_error_sum /
                             static_cast<double>(cam1_point_count));
    }
  };

  const auto evaluate_pair_board_residual =
      [](const StereoMeasurementDataset& dataset,
         const StereoSceneState& scene_state,
         int pair_index,
         int board_id) -> PairBoardTraceResidualStats {
    PairBoardTraceResidualStats stats;
    const auto board_pose_it = scene_state.T_world_board_by_id.find(board_id);
    const auto pair_pose_it = scene_state.T_cam0_world_by_pair.find(pair_index);
    if (pair_pose_it == scene_state.T_cam0_world_by_pair.end() ||
        board_pose_it == scene_state.T_world_board_by_id.end()) {
      return stats;
    }
    const DoubleSphereCameraModel cam0 =
        DoubleSphereCameraModel::FromConfig(MakeCameraConfig(scene_state.cam0));
    const DoubleSphereCameraModel cam1 =
        DoubleSphereCameraModel::FromConfig(MakeCameraConfig(scene_state.cam1));
    if (!cam0.IsValid() || !cam1.IsValid()) {
      return stats;
    }

    const Eigen::Isometry3d T_cam0_world = ToIsometry3d(pair_pose_it->second);
    const Eigen::Isometry3d T_world_board = ToIsometry3d(board_pose_it->second);
    const Eigen::Isometry3d T_cam1_cam0 = ToIsometry3d(scene_state.T_cam1_cam0);
    for (const StereoObservation& observation : dataset.observations) {
      if (observation.pair_index != pair_index ||
          observation.board_id != board_id ||
          !observation.used_in_solver) {
        continue;
      }
      const Eigen::Vector3d point_cam0 =
          T_cam0_world * (T_world_board * observation.target_point_board);
      Eigen::Vector2d predicted = Eigen::Vector2d::Zero();
      bool valid_projection = false;
      if (observation.camera_index == 0) {
        valid_projection = cam0.vsEuclideanToKeypoint(point_cam0, &predicted);
      } else if (observation.camera_index == 1) {
        valid_projection =
            cam1.vsEuclideanToKeypoint(T_cam1_cam0 * point_cam0, &predicted);
      }
      if (!valid_projection || !predicted.allFinite()) {
        continue;
      }
      const Eigen::Vector2d residual =
          predicted - observation.observed_image_xy;
      if (!residual.allFinite()) {
        continue;
      }
      const double squared_error = residual.squaredNorm();
      stats.squared_error_sum += squared_error;
      ++stats.point_count;
      if (observation.camera_index == 0) {
        stats.cam0_squared_error_sum += squared_error;
        ++stats.cam0_point_count;
      } else if (observation.camera_index == 1) {
        stats.cam1_squared_error_sum += squared_error;
        ++stats.cam1_point_count;
      }
    }
    return stats;
  };

  const auto pose_rotation_delta_deg =
      [](const Eigen::Matrix4d* from,
         const Eigen::Matrix4d* to) -> double {
    if (from == nullptr || to == nullptr) {
      return std::numeric_limits<double>::quiet_NaN();
    }
    return RotationDistanceRadians(ToIsometry3d(*from), ToIsometry3d(*to)) *
           180.0 / M_PI;
  };

  const auto pose_translation_delta_m =
      [](const Eigen::Matrix4d* from,
         const Eigen::Matrix4d* to) -> double {
    if (from == nullptr || to == nullptr) {
      return std::numeric_limits<double>::quiet_NaN();
    }
    return (ToIsometry3d(*to).translation() -
            ToIsometry3d(*from).translation()).norm();
  };

  const auto find_pose =
      [](const std::map<int, Eigen::Matrix4d>& poses,
         int id) -> const Eigen::Matrix4d* {
    const auto it = poses.find(id);
    return it == poses.end() ? nullptr : &it->second;
  };

  std::map<int, const StereoPairResidualSummary*> initial_by_pair;
  for (const StereoPairResidualSummary& summary :
       result.training_selected_initial_residual_summary.pair_summaries) {
    initial_by_pair[summary.pair_index] = &summary;
  }
  std::map<int, const StereoPairResidualSummary*> final_by_pair;
  for (const StereoPairResidualSummary& summary :
       result.training_selected_final_residual_summary.pair_summaries) {
    final_by_pair[summary.pair_index] = &summary;
  }

  std::map<int, std::vector<int> > selected_boards_by_pair;
  if (!result.pair_selection_summary.selected_pair_board_keys.empty()) {
    for (const PairBoardKey& key :
         result.pair_selection_summary.selected_pair_board_keys) {
      selected_boards_by_pair[key.first].push_back(key.second);
    }
  } else {
    for (int pair_index : result.pair_selection_summary.selected_pair_indices) {
      const auto boards_it =
          result.problem_input.measurement_dataset.training_pair_board_ids.find(
              pair_index);
      if (boards_it !=
          result.problem_input.measurement_dataset.training_pair_board_ids.end()) {
        selected_boards_by_pair[pair_index] =
            std::vector<int>(boards_it->second.begin(), boards_it->second.end());
      }
    }
  }
  for (auto& entry : selected_boards_by_pair) {
    std::sort(entry.second.begin(), entry.second.end());
    entry.second.erase(std::unique(entry.second.begin(), entry.second.end()),
                       entry.second.end());
  }

  std::map<int, FactorStats> factor_stats_by_pair;
  for (const StereoObservation& observation :
       result.problem_input.measurement_dataset.observations) {
    if (!observation.used_in_solver ||
        result.pair_selection_summary.selected_pair_indices.count(
            observation.pair_index) == 0 ||
        !PairBoardSelected(result.pair_selection_summary, observation.pair_index,
                           observation.board_id)) {
      continue;
    }
    FactorStats& stats = factor_stats_by_pair[observation.pair_index];
    if (observation.camera_index == 0) {
      ++stats.cam0_factor_count;
      ++stats.board_factor_stats[observation.board_id].cam0_factor_count;
    } else if (observation.camera_index == 1) {
      ++stats.cam1_factor_count;
      ++stats.board_factor_stats[observation.board_id].cam1_factor_count;
    }
    if (observation.point_type == JointPointType::Outer) {
      ++stats.outer_factor_count;
      ++stats.board_factor_stats[observation.board_id].outer_factor_count;
    } else if (observation.point_type == JointPointType::Internal) {
      ++stats.internal_factor_count;
      ++stats.board_factor_stats[observation.board_id].internal_factor_count;
    }
  }

  std::ofstream output(path.c_str());
  output << "pair_index,left_frame_label,right_frame_label,entered_ba,"
         << "selected_board_count,selected_board_ids,"
         << "cam0_reprojection_factor_count,cam1_reprojection_factor_count,"
         << "t_1_0_reprojection_factor_count,total_reprojection_factor_count,"
         << "outer_factor_count,internal_factor_count,"
         << "board_factor_counts,initial_overall_rmse,final_overall_rmse,"
         << "overall_rmse_delta,initial_cam0_rmse,final_cam0_rmse,cam0_rmse_delta,"
         << "initial_cam1_rmse,final_cam1_rmse,cam1_rmse_delta,"
         << "residual_state,t_1_0_constrained_by_frame,"
         << "t_1_0_sensitivity_proxy,board_residual_stats,"
         << "t_1_0_init_to_final_rotation_deg,"
         << "t_1_0_init_to_final_translation_m,"
         << "t_1_0_final_ba_rotation_deg,"
         << "t_1_0_final_ba_translation_m,"
         << "frame_pose_init_to_final_rotation_deg,"
         << "frame_pose_init_to_final_translation_m,"
         << "frame_pose_final_ba_rotation_deg,"
         << "frame_pose_final_ba_translation_m,"
         << "optimized_variable_summary,"
         << "factor_count_source\n";

  const double t_init_to_final_rotation_deg = pose_rotation_delta_deg(
      &result.post_initialization_scene.T_cam1_cam0,
      &result.optimized_scene.T_cam1_cam0);
  const double t_init_to_final_translation_m = pose_translation_delta_m(
      &result.post_initialization_scene.T_cam1_cam0,
      &result.optimized_scene.T_cam1_cam0);
  const double t_final_ba_rotation_deg = pose_rotation_delta_deg(
      &result.pre_global_sparse_ba_scene.T_cam1_cam0,
      &result.optimized_scene.T_cam1_cam0);
  const double t_final_ba_translation_m = pose_translation_delta_m(
      &result.pre_global_sparse_ba_scene.T_cam1_cam0,
      &result.optimized_scene.T_cam1_cam0);

  for (int pair_index : result.pair_selection_summary.selected_pair_indices) {
    const auto final_it = final_by_pair.find(pair_index);
    const auto initial_it = initial_by_pair.find(pair_index);
    const StereoPairResidualSummary* final_summary =
        final_it == final_by_pair.end() ? nullptr : final_it->second;
    const StereoPairResidualSummary* initial_summary =
        initial_it == initial_by_pair.end() ? nullptr : initial_it->second;
    const StereoPairResidualSummary* label_summary =
        final_summary != nullptr ? final_summary : initial_summary;
    const FactorStats& stats = factor_stats_by_pair[pair_index];
    const int total_factor_count =
        stats.cam0_factor_count + stats.cam1_factor_count;
    const bool t_1_0_constrained = stats.cam1_factor_count > 0;
    const Eigen::Matrix4d* initial_frame_pose = find_pose(
        result.post_initialization_scene.T_cam0_world_by_pair, pair_index);
    const Eigen::Matrix4d* pre_ba_frame_pose = find_pose(
        result.pre_global_sparse_ba_scene.T_cam0_world_by_pair, pair_index);
    const Eigen::Matrix4d* final_frame_pose = find_pose(
        result.optimized_scene.T_cam0_world_by_pair, pair_index);
    const double frame_pose_init_to_final_rotation_deg =
        pose_rotation_delta_deg(initial_frame_pose, final_frame_pose);
    const double frame_pose_init_to_final_translation_m =
        pose_translation_delta_m(initial_frame_pose, final_frame_pose);
    const double frame_pose_final_ba_rotation_deg =
        pose_rotation_delta_deg(pre_ba_frame_pose, final_frame_pose);
    const double frame_pose_final_ba_translation_m =
        pose_translation_delta_m(pre_ba_frame_pose, final_frame_pose);

    const double initial_overall =
        initial_summary == nullptr ? std::numeric_limits<double>::quiet_NaN()
                                   : initial_summary->overall_rmse;
    const double final_overall =
        final_summary == nullptr ? std::numeric_limits<double>::quiet_NaN()
                                 : final_summary->overall_rmse;
    const double overall_delta = final_overall - initial_overall;
    const double initial_cam0 =
        initial_summary == nullptr ? std::numeric_limits<double>::quiet_NaN()
                                   : initial_summary->cam0_rmse;
    const double final_cam0 =
        final_summary == nullptr ? std::numeric_limits<double>::quiet_NaN()
                                 : final_summary->cam0_rmse;
    const double cam0_delta = final_cam0 - initial_cam0;
    const double initial_cam1 =
        initial_summary == nullptr ? std::numeric_limits<double>::quiet_NaN()
                                   : initial_summary->cam1_rmse;
    const double final_cam1 =
        final_summary == nullptr ? std::numeric_limits<double>::quiet_NaN()
                                 : final_summary->cam1_rmse;
    const double cam1_delta = final_cam1 - initial_cam1;

    std::string residual_state = "unknown";
    if (std::isfinite(overall_delta)) {
      if (overall_delta < -1e-6) {
        residual_state = "improved";
      } else if (overall_delta > 1e-6) {
        residual_state = "worse";
      } else {
        residual_state = "stable";
      }
    }

    std::ostringstream board_factor_counts;
    std::ostringstream board_residual_stats;
    bool first_board = true;
    double frame_t_sensitivity_proxy = 0.0;
    for (const auto& board_entry : stats.board_factor_stats) {
      if (!first_board) {
        board_factor_counts << ";";
        board_residual_stats << ";";
      }
      first_board = false;
      const BoardFactorStats& board_stats = board_entry.second;
      const int board_factor_count =
          board_stats.cam0_factor_count + board_stats.cam1_factor_count;
      const double outer_ratio =
          board_factor_count <= 0
              ? 0.0
              : static_cast<double>(board_stats.outer_factor_count) /
                    static_cast<double>(board_factor_count);
      const double t_sensitivity_proxy =
          std::sqrt(static_cast<double>(
              std::max(0, board_stats.cam1_factor_count))) *
          (0.5 + 0.5 * outer_ratio);
      frame_t_sensitivity_proxy += t_sensitivity_proxy;
      const Eigen::Matrix4d* initial_board_pose = find_pose(
          result.post_initialization_scene.T_world_board_by_id,
          board_entry.first);
      const Eigen::Matrix4d* pre_ba_board_pose = find_pose(
          result.pre_global_sparse_ba_scene.T_world_board_by_id,
          board_entry.first);
      const Eigen::Matrix4d* final_board_pose = find_pose(
          result.optimized_scene.T_world_board_by_id,
          board_entry.first);
      const double board_init_to_final_rotation_deg =
          pose_rotation_delta_deg(initial_board_pose, final_board_pose);
      const double board_init_to_final_translation_m =
          pose_translation_delta_m(initial_board_pose, final_board_pose);
      const double board_final_ba_rotation_deg =
          pose_rotation_delta_deg(pre_ba_board_pose, final_board_pose);
      const double board_final_ba_translation_m =
          pose_translation_delta_m(pre_ba_board_pose, final_board_pose);

      const PairBoardTraceResidualStats initial_board_residual =
          evaluate_pair_board_residual(
              result.problem_input.measurement_dataset,
              result.pre_global_sparse_ba_scene,
              pair_index,
              board_entry.first);
      const PairBoardTraceResidualStats final_board_residual =
          evaluate_pair_board_residual(
              result.problem_input.measurement_dataset,
              result.optimized_scene,
              pair_index,
              board_entry.first);
      const double initial_board_rmse = initial_board_residual.Rmse();
      const double final_board_rmse = final_board_residual.Rmse();
      const double initial_board_cam0_rmse = initial_board_residual.Cam0Rmse();
      const double final_board_cam0_rmse = final_board_residual.Cam0Rmse();
      const double initial_board_cam1_rmse = initial_board_residual.Cam1Rmse();
      const double final_board_cam1_rmse = final_board_residual.Cam1Rmse();

      board_factor_counts << board_entry.first << ":"
                          << board_stats.cam0_factor_count << "/"
                          << board_stats.cam1_factor_count;
      board_residual_stats
          << board_entry.first << ":"
          << board_stats.cam0_factor_count << "/"
          << board_stats.cam1_factor_count << "/"
          << board_stats.outer_factor_count << "/"
          << board_stats.internal_factor_count << "|"
          << initial_board_rmse << "/" << final_board_rmse << "/"
          << (final_board_rmse - initial_board_rmse) << "|"
          << initial_board_cam0_rmse << "/" << final_board_cam0_rmse << "/"
          << (final_board_cam0_rmse - initial_board_cam0_rmse) << "|"
          << initial_board_cam1_rmse << "/" << final_board_cam1_rmse << "/"
          << (final_board_cam1_rmse - initial_board_cam1_rmse) << "|"
          << t_sensitivity_proxy << "|"
          << board_init_to_final_rotation_deg << "/"
          << board_init_to_final_translation_m << "|"
          << board_final_ba_rotation_deg << "/"
          << board_final_ba_translation_m;
    }

    const auto selected_boards_it = selected_boards_by_pair.find(pair_index);
    const std::vector<int> empty_boards;
    const std::vector<int>& selected_boards =
        selected_boards_it == selected_boards_by_pair.end()
            ? empty_boards
            : selected_boards_it->second;

    output << pair_index << ","
           << (label_summary == nullptr ? "" : label_summary->left_frame_label)
           << ","
           << (label_summary == nullptr ? "" : label_summary->right_frame_label)
           << ",1,"
           << selected_boards.size() << ","
           << JoinInts(selected_boards, ';') << ","
           << stats.cam0_factor_count << ","
           << stats.cam1_factor_count << ","
           << stats.cam1_factor_count << ","
           << total_factor_count << ","
           << stats.outer_factor_count << ","
           << stats.internal_factor_count << ","
           << board_factor_counts.str() << ","
           << initial_overall << "," << final_overall << ","
           << overall_delta << ","
           << initial_cam0 << "," << final_cam0 << "," << cam0_delta << ","
           << initial_cam1 << "," << final_cam1 << "," << cam1_delta << ","
           << residual_state << ","
           << (t_1_0_constrained ? 1 : 0) << ","
           << frame_t_sensitivity_proxy << ","
           << board_residual_stats.str() << ","
           << t_init_to_final_rotation_deg << ","
           << t_init_to_final_translation_m << ","
           << t_final_ba_rotation_deg << ","
           << t_final_ba_translation_m << ","
           << frame_pose_init_to_final_rotation_deg << ","
           << frame_pose_init_to_final_translation_m << ","
           << frame_pose_final_ba_rotation_deg << ","
           << frame_pose_final_ba_translation_m << ","
           << "K0_fixed|K1_fixed|T_1_0_optimized|T_cam0_world_frame_optimized|T_world_board_optimized,"
           << "selected_pair_board_used_in_solver\n";
  }
}

namespace {

struct StereoJacobianDiagnosticScope {
  std::string scope;
  int pair_index = -1;
  int board_id = -1;
  std::string variable_block;
  bool active_in_solver = true;
  std::vector<const StereoObservation*> observations;
};

struct StereoJacobianDiagnosticStats {
  int residual_count = 0;
  int residual_dimension = 0;
  double residual_norm = std::numeric_limits<double>::quiet_NaN();
  double rmse_like = std::numeric_limits<double>::quiet_NaN();
  double hessian_trace = std::numeric_limits<double>::quiet_NaN();
  double hessian_frobenius_norm = std::numeric_limits<double>::quiet_NaN();
  double hessian_logdet = std::numeric_limits<double>::quiet_NaN();
  int hessian_rank_proxy = 0;
  double gradient_norm = std::numeric_limits<double>::quiet_NaN();
  double gradient_over_residual_norm = std::numeric_limits<double>::quiet_NaN();
  double condition_number = std::numeric_limits<double>::quiet_NaN();
  double min_singular_value = std::numeric_limits<double>::quiet_NaN();
  double max_singular_value = std::numeric_limits<double>::quiet_NaN();
};

Eigen::Isometry3d PerturbPoseLeft(const Eigen::Isometry3d& pose,
                                  int axis,
                                  double epsilon) {
  Eigen::Isometry3d delta = Eigen::Isometry3d::Identity();
  if (axis < 3) {
    Eigen::Vector3d rotation_axis = Eigen::Vector3d::Zero();
    rotation_axis(axis) = 1.0;
    delta.linear() =
        Eigen::AngleAxisd(epsilon, rotation_axis).toRotationMatrix();
  } else {
    delta.translation()(axis - 3) = epsilon;
  }
  return delta * pose;
}

double Stage6SphericalDiagnosticWeight(
    const StereoExtrinsicSolverOptions& options,
    double observation_weight,
    double polar_angle_deg) {
  double weight =
      std::max(0.0, options.spherical_weight) *
      std::sqrt(std::max(0.0, observation_weight));
  if (options.spherical_polar_weighting &&
      std::isfinite(polar_angle_deg)) {
    const double polar_scale = ComputePolarContinuousAngularWeight(
        polar_angle_deg, options.spherical_min_polar_deg, 5.0);
    const double max_scale = std::max(1.0, options.spherical_max_weight);
    weight *= 1.0 + (max_scale - 1.0) * polar_scale;
  }
  return weight;
}

bool EvaluateStage6ObservationResidual(
    const StereoSceneState& scene_state,
    const DoubleSphereCameraModel& cam0,
    const DoubleSphereCameraModel& cam1,
    const StereoExtrinsicSolverOptions& options,
    StereoFinalBaResidualMode residual_mode,
    const StereoObservation& observation,
    Eigen::VectorXd* residual) {
  if (residual == nullptr) {
    return false;
  }
  const auto pair_pose_it =
      scene_state.T_cam0_world_by_pair.find(observation.pair_index);
  const auto board_pose_it =
      scene_state.T_world_board_by_id.find(observation.board_id);
  if (pair_pose_it == scene_state.T_cam0_world_by_pair.end() ||
      board_pose_it == scene_state.T_world_board_by_id.end()) {
    return false;
  }
  const Eigen::Isometry3d T_cam0_world = ToIsometry3d(pair_pose_it->second);
  const Eigen::Isometry3d T_world_board = ToIsometry3d(board_pose_it->second);
  const Eigen::Isometry3d T_cam1_cam0 = ToIsometry3d(scene_state.T_cam1_cam0);
  Eigen::Vector3d point_camera =
      T_cam0_world * (T_world_board * observation.target_point_board);
  const DoubleSphereCameraModel* camera = &cam0;
  if (observation.camera_index == 1) {
    point_camera = T_cam1_cam0 * point_camera;
    camera = &cam1;
  } else if (observation.camera_index != 0) {
    return false;
  }

  Eigen::Vector2d predicted = Eigen::Vector2d::Zero();
  if (!camera->vsEuclideanToKeypoint(point_camera, &predicted) ||
      !predicted.allFinite()) {
    return false;
  }
  const double pixel_weight =
      std::sqrt(std::max(0.0, observation.weight));

  if (residual_mode == StereoFinalBaResidualMode::Pixel) {
    residual->resize(2);
    *residual = pixel_weight * (predicted - observation.observed_image_xy);
    return residual->allFinite();
  }

  AngularObservationGeometry observation_geometry;
  AngularPredictionGeometry prediction_geometry;
  if (!ComputeAngularObservationGeometry(*camera, observation.observed_image_xy,
                                         &observation_geometry) ||
      !ComputeAngularPredictionGeometry(*camera, point_camera,
                                        &prediction_geometry)) {
    return false;
  }
  const double spherical_weight = Stage6SphericalDiagnosticWeight(
      options, observation.weight, observation_geometry.polar_angle_deg);
  if (!(spherical_weight > 0.0) || !std::isfinite(spherical_weight)) {
    return false;
  }

  if (residual_mode == StereoFinalBaResidualMode::SphericalChordal) {
    residual->resize(3);
    *residual = spherical_weight *
                (prediction_geometry.predicted_ray -
                 observation_geometry.observed_ray);
    return residual->allFinite();
  }

  const Eigen::Vector2d angular_residual =
      spherical_weight *
      ComputeAngularResidualTangent(observation_geometry,
                                    prediction_geometry);
  if (residual_mode == StereoFinalBaResidualMode::SphericalTangent) {
    residual->resize(2);
    *residual = angular_residual;
    return residual->allFinite();
  }

  if (residual_mode == StereoFinalBaResidualMode::HybridPixelSpherical) {
    residual->resize(4);
    residual->head<2>() =
        pixel_weight * (predicted - observation.observed_image_xy);
    residual->tail<2>() = angular_residual;
    return residual->allFinite();
  }

  return false;
}

std::vector<Eigen::VectorXd> EvaluateStage6ResidualBlocks(
    const StereoSceneState& scene_state,
    const DoubleSphereCameraModel& cam0,
    const DoubleSphereCameraModel& cam1,
    const StereoExtrinsicSolverOptions& options,
    StereoFinalBaResidualMode residual_mode,
    const std::vector<const StereoObservation*>& observations) {
  std::vector<Eigen::VectorXd> residuals;
  residuals.reserve(observations.size());
  for (const StereoObservation* observation : observations) {
    if (observation == nullptr) {
      residuals.emplace_back();
      continue;
    }
    Eigen::VectorXd residual;
    if (EvaluateStage6ObservationResidual(scene_state, cam0, cam1, options,
                                          residual_mode, *observation,
                                          &residual)) {
      residuals.push_back(residual);
    } else {
      residuals.emplace_back();
    }
  }
  return residuals;
}

Eigen::VectorXd FlattenValidResiduals(
    const std::vector<Eigen::VectorXd>& residuals,
    const std::vector<int>& valid_indices) {
  int dimension = 0;
  for (int index : valid_indices) {
    if (index >= 0 && index < static_cast<int>(residuals.size())) {
      dimension += static_cast<int>(residuals[index].size());
    }
  }
  Eigen::VectorXd flattened(dimension);
  int offset = 0;
  for (int index : valid_indices) {
    const Eigen::VectorXd& residual = residuals[index];
    flattened.segment(offset, residual.size()) = residual;
    offset += static_cast<int>(residual.size());
  }
  return flattened;
}

void ApplyStage6DiagnosticPosePerturbation(StereoSceneState* scene_state,
                                           const StereoJacobianDiagnosticScope& scope,
                                           int axis,
                                           double epsilon) {
  if (scene_state == nullptr) {
    return;
  }
  if (scope.variable_block == "T_1_0") {
    scene_state->T_cam1_cam0 = ToMatrix4d(PerturbPoseLeft(
        ToIsometry3d(scene_state->T_cam1_cam0), axis, epsilon));
    return;
  }
  if (scope.variable_block == "T_cam0_world") {
    auto pose_it = scene_state->T_cam0_world_by_pair.find(scope.pair_index);
    if (pose_it != scene_state->T_cam0_world_by_pair.end()) {
      pose_it->second = ToMatrix4d(
          PerturbPoseLeft(ToIsometry3d(pose_it->second), axis, epsilon));
    }
    return;
  }
  if (scope.variable_block == "T_world_board") {
    auto pose_it = scene_state->T_world_board_by_id.find(scope.board_id);
    if (pose_it != scene_state->T_world_board_by_id.end()) {
      pose_it->second = ToMatrix4d(
          PerturbPoseLeft(ToIsometry3d(pose_it->second), axis, epsilon));
    }
  }
}

StereoJacobianDiagnosticStats ComputeStage6JacobianDiagnosticStats(
    const StereoSceneState& scene_state,
    const DoubleSphereCameraModel& cam0,
    const DoubleSphereCameraModel& cam1,
    const StereoExtrinsicSolverOptions& options,
    StereoFinalBaResidualMode residual_mode,
    const StereoJacobianDiagnosticScope& scope) {
  StereoJacobianDiagnosticStats stats;
  if (scope.observations.empty()) {
    return stats;
  }

  const std::vector<Eigen::VectorXd> base_blocks =
      EvaluateStage6ResidualBlocks(scene_state, cam0, cam1, options,
                                   residual_mode, scope.observations);
  std::vector<int> valid_indices;
  valid_indices.reserve(base_blocks.size());
  for (int index = 0; index < static_cast<int>(base_blocks.size()); ++index) {
    if (base_blocks[index].size() > 0 && base_blocks[index].allFinite()) {
      valid_indices.push_back(index);
    }
  }
  if (valid_indices.empty()) {
    return stats;
  }

  const Eigen::VectorXd r = FlattenValidResiduals(base_blocks, valid_indices);
  if (r.size() <= 0 || !r.allFinite()) {
    return stats;
  }
  stats.residual_count = static_cast<int>(valid_indices.size());
  stats.residual_dimension = static_cast<int>(r.size());
  stats.residual_norm = r.norm();
  stats.rmse_like =
      std::sqrt(r.squaredNorm() /
                static_cast<double>(std::max<int>(
                    1, static_cast<int>(r.size()))));

  Eigen::MatrixXd jacobian(r.size(), 6);
  jacobian.setZero();
  const double rotation_epsilon = 1e-6;
  const double translation_epsilon = 1e-6;
  for (int axis = 0; axis < 6; ++axis) {
    const double epsilon = axis < 3 ? rotation_epsilon : translation_epsilon;
    StereoSceneState positive_scene = scene_state;
    StereoSceneState negative_scene = scene_state;
    ApplyStage6DiagnosticPosePerturbation(&positive_scene, scope, axis,
                                          epsilon);
    ApplyStage6DiagnosticPosePerturbation(&negative_scene, scope, axis,
                                          -epsilon);
    const std::vector<Eigen::VectorXd> positive_blocks =
        EvaluateStage6ResidualBlocks(positive_scene, cam0, cam1, options,
                                     residual_mode, scope.observations);
    const std::vector<Eigen::VectorXd> negative_blocks =
        EvaluateStage6ResidualBlocks(negative_scene, cam0, cam1, options,
                                     residual_mode, scope.observations);
    int offset = 0;
    for (int valid_index : valid_indices) {
      const Eigen::VectorXd& base = base_blocks[valid_index];
      Eigen::VectorXd positive = base;
      Eigen::VectorXd negative = base;
      if (valid_index < static_cast<int>(positive_blocks.size()) &&
          positive_blocks[valid_index].size() == base.size() &&
          positive_blocks[valid_index].allFinite()) {
        positive = positive_blocks[valid_index];
      }
      if (valid_index < static_cast<int>(negative_blocks.size()) &&
          negative_blocks[valid_index].size() == base.size() &&
          negative_blocks[valid_index].allFinite()) {
        negative = negative_blocks[valid_index];
      }
      jacobian.block(offset, axis, base.size(), 1) =
          (positive - negative) / (2.0 * epsilon);
      offset += static_cast<int>(base.size());
    }
  }

  const Eigen::Matrix<double, 6, 6> hessian =
      jacobian.transpose() * jacobian;
  const Eigen::Matrix<double, 6, 1> gradient = jacobian.transpose() * r;
  stats.hessian_trace = hessian.trace();
  stats.hessian_frobenius_norm = hessian.norm();
  stats.gradient_norm = gradient.norm();
  stats.gradient_over_residual_norm =
      stats.gradient_norm / std::max(1e-12, stats.residual_norm);

  Eigen::JacobiSVD<Eigen::Matrix<double, 6, 6> > svd(hessian);
  const Eigen::Matrix<double, 6, 1> singular_values = svd.singularValues();
  stats.max_singular_value = singular_values.maxCoeff();
  stats.min_singular_value = std::numeric_limits<double>::infinity();
  stats.hessian_logdet = 0.0;
  const double rank_threshold =
      std::max(1e-12, stats.max_singular_value * 1e-9);
  for (int index = 0; index < singular_values.size(); ++index) {
    const double value = singular_values(index);
    stats.hessian_logdet += std::log(std::max(1e-18, value + 1e-12));
    if (value > rank_threshold) {
      ++stats.hessian_rank_proxy;
      stats.min_singular_value = std::min(stats.min_singular_value, value);
    }
  }
  if (!std::isfinite(stats.min_singular_value)) {
    stats.min_singular_value = 0.0;
  }
  stats.condition_number =
      stats.min_singular_value > 0.0
          ? stats.max_singular_value / stats.min_singular_value
          : std::numeric_limits<double>::infinity();
  return stats;
}

std::string Stage6CsvLabelForPair(
    const StereoMeasurementDataset& dataset,
    int pair_index,
    bool left_label) {
  for (const StereoFramePair& pair : dataset.frame_pairs) {
    if (pair.pair_index == pair_index) {
      return left_label ? pair.left_frame_label : pair.right_frame_label;
    }
  }
  return "";
}

}  // namespace

void WriteStereoJacobianBlockDiagnosticsCsv(
    const std::string& path,
    const StereoExtrinsicCalibrationResult& result) {
  const StereoSceneState& scene_state = result.optimized_scene;
  const StereoMeasurementDataset& dataset =
      result.problem_input.measurement_dataset;
  const StereoExtrinsicSolverOptions& options =
      result.problem_input.solver_options;
  const StereoFinalBaResidualMode residual_mode =
      options.skip_final_global_ba ? options.selection_ba_residual_mode
                                   : result.global_sparse_ba_summary.residual_mode;
  const DoubleSphereCameraModel cam0 =
      DoubleSphereCameraModel::FromConfig(MakeCameraConfig(scene_state.cam0));
  const DoubleSphereCameraModel cam1 =
      DoubleSphereCameraModel::FromConfig(MakeCameraConfig(scene_state.cam1));

  std::ofstream output(path.c_str());
  output << "scope,pair_index,board_id,variable_block,active_in_solver,"
         << "residual_mode,residual_count,residual_dimension,jacobian_columns,"
         << "residual_norm,rmse_like,hessian_trace,hessian_frobenius_norm,"
         << "hessian_logdet,hessian_rank_proxy,gradient_norm,"
         << "gradient_over_residual_norm,condition_number,"
         << "min_singular_value,max_singular_value,left_frame_label,"
         << "right_frame_label\n";

  if (!cam0.IsValid() || !cam1.IsValid()) {
    return;
  }

  std::map<int, std::vector<const StereoObservation*> > frame_observations;
  std::map<int, std::vector<const StereoObservation*> > frame_cam1_observations;
  std::map<PairBoardKey, std::vector<const StereoObservation*> >
      pair_board_observations;
  std::map<PairBoardKey, std::vector<const StereoObservation*> >
      pair_board_cam1_observations;

  for (const StereoObservation& observation : dataset.observations) {
    if (!observation.used_in_solver ||
        result.pair_selection_summary.selected_pair_indices.count(
            observation.pair_index) == 0 ||
        !PairBoardSelected(result.pair_selection_summary, observation.pair_index,
                           observation.board_id)) {
      continue;
    }
    frame_observations[observation.pair_index].push_back(&observation);
    pair_board_observations[PairBoardKey(observation.pair_index,
                                         observation.board_id)]
        .push_back(&observation);
    if (observation.camera_index == 1) {
      frame_cam1_observations[observation.pair_index].push_back(&observation);
      pair_board_cam1_observations[PairBoardKey(observation.pair_index,
                                                observation.board_id)]
          .push_back(&observation);
    }
  }

  std::vector<StereoJacobianDiagnosticScope> scopes;
  for (const auto& entry : frame_observations) {
    StereoJacobianDiagnosticScope frame_pose_scope;
    frame_pose_scope.scope = "frame";
    frame_pose_scope.pair_index = entry.first;
    frame_pose_scope.board_id = -1;
    frame_pose_scope.variable_block = "T_cam0_world";
    frame_pose_scope.active_in_solver =
        result.global_sparse_ba_summary.optimize_pair_poses;
    frame_pose_scope.observations = entry.second;
    scopes.push_back(frame_pose_scope);

    StereoJacobianDiagnosticScope stereo_scope;
    stereo_scope.scope = "frame";
    stereo_scope.pair_index = entry.first;
    stereo_scope.board_id = -1;
    stereo_scope.variable_block = "T_1_0";
    stereo_scope.active_in_solver =
        result.global_sparse_ba_summary.optimize_stereo_extrinsic;
    stereo_scope.observations = frame_cam1_observations[entry.first];
    scopes.push_back(stereo_scope);
  }

  for (const auto& entry : pair_board_observations) {
    const int pair_index = entry.first.first;
    const int board_id = entry.first.second;

    StereoJacobianDiagnosticScope frame_pose_scope;
    frame_pose_scope.scope = "pair_board";
    frame_pose_scope.pair_index = pair_index;
    frame_pose_scope.board_id = board_id;
    frame_pose_scope.variable_block = "T_cam0_world";
    frame_pose_scope.active_in_solver =
        result.global_sparse_ba_summary.optimize_pair_poses;
    frame_pose_scope.observations = entry.second;
    scopes.push_back(frame_pose_scope);

    StereoJacobianDiagnosticScope stereo_scope;
    stereo_scope.scope = "pair_board";
    stereo_scope.pair_index = pair_index;
    stereo_scope.board_id = board_id;
    stereo_scope.variable_block = "T_1_0";
    stereo_scope.active_in_solver =
        result.global_sparse_ba_summary.optimize_stereo_extrinsic;
    stereo_scope.observations =
        pair_board_cam1_observations[PairBoardKey(pair_index, board_id)];
    scopes.push_back(stereo_scope);

    StereoJacobianDiagnosticScope board_pose_scope;
    board_pose_scope.scope = "pair_board";
    board_pose_scope.pair_index = pair_index;
    board_pose_scope.board_id = board_id;
    board_pose_scope.variable_block = "T_world_board";
    board_pose_scope.active_in_solver =
        result.global_sparse_ba_summary.optimize_board_poses &&
        board_id != scene_state.gauge_fixed_board_id;
    board_pose_scope.observations = entry.second;
    scopes.push_back(board_pose_scope);
  }

  for (const StereoJacobianDiagnosticScope& scope : scopes) {
    const StereoJacobianDiagnosticStats stats =
        ComputeStage6JacobianDiagnosticStats(scene_state, cam0, cam1, options,
                                             residual_mode, scope);
    output << scope.scope << "," << scope.pair_index << ","
           << scope.board_id << "," << scope.variable_block << ","
           << (scope.active_in_solver ? 1 : 0) << ","
           << ToString(residual_mode) << ","
           << stats.residual_count << "," << stats.residual_dimension << ",6,"
           << stats.residual_norm << "," << stats.rmse_like << ","
           << stats.hessian_trace << "," << stats.hessian_frobenius_norm << ","
           << stats.hessian_logdet << "," << stats.hessian_rank_proxy << ","
           << stats.gradient_norm << ","
           << stats.gradient_over_residual_norm << ","
           << stats.condition_number << "," << stats.min_singular_value << ","
           << stats.max_singular_value << ","
           << Stage6CsvLabelForPair(dataset, scope.pair_index, true) << ","
           << Stage6CsvLabelForPair(dataset, scope.pair_index, false) << "\n";
  }
}

void WriteStereoPerBoardResidualsCsv(const std::string& path,
                                     const StereoExtrinsicCalibrationResult& result) {
  std::ofstream output(path.c_str());
  output << "split,board_id,point_count,cam0_point_count,cam1_point_count,"
         << "shared_pair_count,shared_point_count,shared_cam0_point_count,"
         << "shared_cam1_point_count,shared_outer_point_count,shared_internal_point_count,"
         << "rmse,shared_cam0_rmse,shared_cam1_rmse,shared_outer_rmse,"
         << "shared_internal_rmse\n";
  for (const StereoBoardResidualSummary& summary :
       result.training_residual_summary.board_summaries) {
    output << "training," << summary.board_id << "," << summary.point_count << ","
           << summary.cam0_point_count << "," << summary.cam1_point_count << ","
           << summary.shared_pair_count << "," << summary.shared_point_count << ","
           << summary.shared_cam0_point_count << ","
           << summary.shared_cam1_point_count << ","
           << summary.shared_outer_point_count << ","
           << summary.shared_internal_point_count << ","
           << summary.rmse << "," << summary.shared_cam0_rmse << ","
           << summary.shared_cam1_rmse << "," << summary.shared_outer_rmse << ","
           << summary.shared_internal_rmse << "\n";
  }
  for (const StereoBoardResidualSummary& summary :
       result.holdout_extrinsic_only_residual_summary.board_summaries) {
    output << "holdout_extrinsic_only," << summary.board_id << ","
           << summary.point_count << ","
           << summary.cam0_point_count << "," << summary.cam1_point_count << ","
           << summary.shared_pair_count << "," << summary.shared_point_count
           << ","
           << summary.shared_cam0_point_count << ","
           << summary.shared_cam1_point_count << ","
           << summary.shared_outer_point_count << ","
           << summary.shared_internal_point_count << ","
           << summary.rmse << "," << summary.shared_cam0_rmse << ","
           << summary.shared_cam1_rmse << "," << summary.shared_outer_rmse
           << ","
           << summary.shared_internal_rmse << "\n";
  }
}

void WriteStereoIntrinsicsSanitySummary(
    const std::string& path,
    const StereoExtrinsicCalibrationResult& result) {
  std::ofstream output(path.c_str());
  const StereoCameraFixedCalibration& initial_cam0 =
      result.problem_input.initial_scene.cam0;
  const StereoCameraFixedCalibration& initial_cam1 =
      result.problem_input.initial_scene.cam1;
  const StereoCameraFixedCalibration& cam0 = result.optimized_scene.cam0;
  const StereoCameraFixedCalibration& cam1 = result.optimized_scene.cam1;
  const bool same_model = cam0.camera_model == cam1.camera_model &&
                          cam0.camera_model_family == cam1.camera_model_family &&
                          cam0.distortion_model == cam1.distortion_model;
  const bool same_intrinsics = VectorsEqual(cam0.intrinsics, cam1.intrinsics) &&
                               VectorsEqual(cam0.distortion_coeffs,
                                            cam1.distortion_coeffs);
  const bool same_resolution = VectorsEqual(cam0.resolution, cam1.resolution);
  const bool likely_intrinsics_shared_scale_issue =
      same_intrinsics &&
      result.training_residual_summary.shared_cam0_rmse > 0.0 &&
      result.training_residual_summary.shared_cam1_rmse > 0.0 &&
      std::abs(result.training_residual_summary.shared_cam1_rmse /
                   result.training_residual_summary.shared_cam0_rmse -
               1.0) > 0.15;
  const StereoIntrinsicsPolicyDecision intrinsics_policy =
      EvaluateStereoIntrinsicsPolicy(
          result.problem_input.measurement_dataset,
          result.problem_input.solver_options);
  const bool has_separate_distortion_dv =
      !initial_cam0.distortion_coeffs.empty() ||
      !initial_cam1.distortion_coeffs.empty();

  output << "left_camera_seed_source: " << cam0.source_label << "\n";
  output << "right_camera_seed_source: " << cam1.source_label << "\n";
  output << "stage6_uses_external_intrinsics: 0\n";
  output << "left_camera_model_family: " << cam0.camera_model_family << "\n";
  output << "right_camera_model_family: " << cam1.camera_model_family << "\n";
  output << "left_camera_model: " << cam0.camera_model << "\n";
  output << "right_camera_model: " << cam1.camera_model << "\n";
  output << "left_distortion_model: " << cam0.distortion_model << "\n";
  output << "right_distortion_model: " << cam1.distortion_model << "\n";
  output << "left_resolution: " << FormatIntVector(cam0.resolution) << "\n";
  output << "right_resolution: " << FormatIntVector(cam1.resolution) << "\n";
  output << "left_intrinsics: " << FormatDoubleVector(cam0.intrinsics) << "\n";
  output << "right_intrinsics: " << FormatDoubleVector(cam1.intrinsics) << "\n";
  output << "stage6_intrinsics_mode: "
         << StereoIntrinsicsModeToString(intrinsics_policy.requested_mode)
         << "\n";
  output << "stage6_requested_intrinsics_mode: "
         << StereoIntrinsicsModeToString(intrinsics_policy.requested_mode)
         << "\n";
  output << "stage6_effective_intrinsics_mode: "
         << StereoIntrinsicsModeToString(intrinsics_policy.effective_mode)
         << "\n";
  output << "stage6_projection_intrinsics_active: "
         << (intrinsics_policy.projection_active ? 1 : 0) << "\n";
  output << "stage6_projection_prior_enabled: "
         << (intrinsics_policy.projection_prior_enabled ? 1 : 0) << "\n";
  output << "stage6_distortion_intrinsics_active: "
         << (intrinsics_policy.distortion_active && has_separate_distortion_dv
                 ? 1 : 0)
         << "\n";
  output << "stage6_distortion_prior_enabled: "
         << (intrinsics_policy.distortion_prior_enabled &&
                     has_separate_distortion_dv
                 ? 1 : 0)
         << "\n";
  output << "stage6_distortion_prior_sigma: "
         << result.problem_input.solver_options
                .persistent_incremental_distortion_prior_sigma
         << "\n";
  output << "stage6_projection_release_reason: "
         << intrinsics_policy.reason << "\n";
  output << "left_initial_intrinsics: "
         << FormatDoubleVector(initial_cam0.intrinsics) << "\n";
  output << "right_initial_intrinsics: "
         << FormatDoubleVector(initial_cam1.intrinsics) << "\n";
  output << "left_intrinsics_changed: "
         << (!VectorsEqual(initial_cam0.intrinsics, cam0.intrinsics) ? 1 : 0)
         << "\n";
  output << "right_intrinsics_changed: "
         << (!VectorsEqual(initial_cam1.intrinsics, cam1.intrinsics) ? 1 : 0)
         << "\n";
  output << "left_distortion_coeffs: "
         << FormatDoubleVector(cam0.distortion_coeffs) << "\n";
  output << "right_distortion_coeffs: "
         << FormatDoubleVector(cam1.distortion_coeffs) << "\n";
  output << "left_initial_distortion_coeffs: "
         << FormatDoubleVector(initial_cam0.distortion_coeffs) << "\n";
  output << "right_initial_distortion_coeffs: "
         << FormatDoubleVector(initial_cam1.distortion_coeffs) << "\n";
  output << "left_distortion_changed: "
         << (!VectorsEqual(initial_cam0.distortion_coeffs,
                           cam0.distortion_coeffs)
                 ? 1 : 0)
         << "\n";
  output << "right_distortion_changed: "
         << (!VectorsEqual(initial_cam1.distortion_coeffs,
                           cam1.distortion_coeffs)
                 ? 1 : 0)
         << "\n";
  output << "same_camera_model: " << (same_model ? 1 : 0) << "\n";
  output << "same_intrinsics_parameters: " << (same_intrinsics ? 1 : 0) << "\n";
  output << "same_resolution: " << (same_resolution ? 1 : 0) << "\n";
  output << "training_shared_cam0_rmse: "
         << result.training_residual_summary.shared_cam0_rmse << "\n";
  output << "training_shared_cam1_rmse: "
         << result.training_residual_summary.shared_cam1_rmse << "\n";
  output << "training_shared_cam1_over_cam0_rmse_ratio: "
         << (result.training_residual_summary.shared_cam0_rmse > 0.0
                 ? result.training_residual_summary.shared_cam1_rmse /
                       result.training_residual_summary.shared_cam0_rmse
                 : 0.0)
         << "\n";
  output << "likely_intrinsics_shared_scale_issue: "
         << (likely_intrinsics_shared_scale_issue ? 1 : 0) << "\n";
  if (same_intrinsics) {
    output << "warning: left and right in-process camera parameters are identical.\n";
  }
  if (likely_intrinsics_shared_scale_issue) {
    output << "warning: shared residuals are camera-imbalanced while intrinsics are "
              "identical; verify independent monocular camera initialization.\n";
  }
}

void WriteStereoPairingSummary(const std::string& path,
                               const StereoExtrinsicProblemInput& input) {
  std::ofstream output(path.c_str());
  output << "measurement_source_mode: " << input.measurement_source_mode << "\n";
  output << "inherits_stage5_persistent_accepted_set: "
         << (input.measurement_source_mode == "backend_selected_only" ? 1 : 0)
         << "\n";
  output << "left_frame_count: " << input.measurement_dataset.left_frame_count << "\n";
  output << "right_frame_count: " << input.measurement_dataset.right_frame_count << "\n";
  output << "paired_frame_count: " << input.measurement_dataset.paired_frame_count << "\n";
  output << "unmatched_left_count: " << input.measurement_dataset.unmatched_left_count << "\n";
  output << "unmatched_right_count: " << input.measurement_dataset.unmatched_right_count << "\n";
  output << "pairing_mode: " << input.measurement_dataset.pairing_mode << "\n";
  output << "max_pair_timestamp_delta_ns: "
         << input.measurement_dataset.max_pair_timestamp_delta_ns << "\n";
  output << "mean_abs_pair_timestamp_delta_ms: "
         << input.measurement_dataset.mean_abs_pair_timestamp_delta_ms << "\n";
  output << "max_abs_pair_timestamp_delta_ms: "
         << input.measurement_dataset.max_abs_pair_timestamp_delta_ms << "\n";
  for (const std::string& warning : input.measurement_dataset.warnings) {
    output << "warning: " << warning << "\n";
  }
}

void WriteStereoInitializationSummary(const std::string& path,
                                      const StereoExtrinsicCalibrationResult& result) {
  std::ofstream output(path.c_str());
  const StereoCameraFixedCalibration& cam0 = result.problem_input.initial_scene.cam0;
  const StereoCameraFixedCalibration& cam1 = result.problem_input.initial_scene.cam1;
  output << "stage6_uses_external_intrinsics: 0\n";
  output << "left_camera_seed_source: " << cam0.source_label << "\n";
  output << "right_camera_seed_source: " << cam1.source_label << "\n";
  output << "left_camera_seed_model_family: " << cam0.camera_model_family
         << "\n";
  output << "right_camera_seed_model_family: " << cam1.camera_model_family
         << "\n";
  output << "left_camera_seed_intrinsics: "
         << FormatDoubleVector(cam0.intrinsics) << "\n";
  output << "right_camera_seed_intrinsics: "
         << FormatDoubleVector(cam1.intrinsics) << "\n";
  output << "left_camera_seed_distortion: "
         << FormatDoubleVector(cam0.distortion_coeffs) << "\n";
  output << "right_camera_seed_distortion: "
         << FormatDoubleVector(cam1.distortion_coeffs) << "\n";
  output << "success: " << (result.initialization.success ? 1 : 0) << "\n";
  output << "failure_reason: " << result.initialization.failure_reason << "\n";
  output << "candidate_count: " << result.initialization.candidate_count << "\n";
  output << "excluded_candidate_count: "
         << result.initialization.excluded_candidate_count << "\n";
  output << "pair_pose_candidate_count: "
         << result.initialization.pair_pose_candidate_count << "\n";
  output << "board_pose_candidate_count: "
         << result.initialization.board_pose_candidate_count << "\n";
  output << "candidate_rejected_pose_fit_count: "
         << result.initialization.candidate_rejected_pose_fit_count << "\n";
  output << "candidate_rejected_consistency_count: "
         << result.initialization.candidate_rejected_consistency_count << "\n";
  output << "graph_seed_pair_count: "
         << result.initialization.graph_seed_pair_count << "\n";
  output << "reachable_training_pair_count: "
         << result.initialization.reachable_training_pair_count << "\n";
  output << "unreachable_training_pair_count: "
         << result.initialization.unreachable_training_pair_count << "\n";
  output << "excluded_training_pair_count: "
         << result.initialization.excluded_training_pair_count << "\n";
  output << "initialized_training_pair_count: "
         << result.initialization.initialized_training_pair_count << "\n";
  output << "initialized_board_count: "
         << result.initialization.initialized_board_count << "\n";
  output << "uninitialized_training_pair_count: "
         << result.initialization.uninitialized_training_pair_count << "\n";
  output << "graph_propagation_iteration_count: "
         << result.initialization.graph_propagation_iteration_count << "\n";
  output << "graph_propagation_new_pair_count: "
         << result.initialization.graph_propagation_new_pair_count << "\n";
  output << "graph_propagation_new_board_count: "
         << result.initialization.graph_propagation_new_board_count << "\n";
  output << "graph_propagation_stopped_by_no_progress: "
         << (result.initialization.graph_propagation_stopped_by_no_progress ? 1 : 0)
         << "\n";
  output << "graph_propagation_stopped_by_iteration_limit: "
         << (result.initialization.graph_propagation_stopped_by_iteration_limit ? 1 : 0)
         << "\n";
  output << "uninitialized_board_count: "
         << result.initialization.uninitialized_board_count << "\n";
  output << "connected_component_count: "
         << result.initialization.connected_component_count << "\n";
  output << "gauge_connected_component_id: "
         << result.initialization.gauge_connected_component_id << "\n";
  output << "gauge_connected_pair_count: "
         << result.initialization.gauge_connected_pair_count << "\n";
  output << "gauge_connected_board_count: "
         << result.initialization.gauge_connected_board_count << "\n";
  output << "medoid_score: " << result.initialization.medoid_score << "\n";
  output << "pair_only_stereo_ba_init_enabled: "
         << (result.problem_input.solver_options.enable_pair_only_stereo_ba_init ? 1 : 0)
         << "\n";
  output << "pair_only_stereo_ba_init_success: "
         << (result.pair_init_summary.success ? 1 : 0) << "\n";
  output << "pair_only_raw_candidate_count: "
         << result.pair_init_summary.raw_candidate_count << "\n";
  output << "pair_only_consistency_filtered_candidate_count: "
         << result.pair_init_summary.consistency_filtered_candidate_count << "\n";
  output << "pair_only_consistency_rejected_candidate_count: "
         << result.pair_init_summary.consistency_rejected_candidate_count << "\n";
  output << "pair_only_before_shared_rmse: "
         << result.pair_init_summary.before_shared_rmse << "\n";
  output << "pair_only_after_shared_rmse: "
         << result.pair_init_summary.after_shared_rmse << "\n";
  output << "pair_only_used_refined_baseline: "
         << (result.pair_init_summary.used_refined_baseline ? 1 : 0) << "\n";
  output << "pair_only_baseline_rotation_delta_deg: "
         << result.pair_init_summary.baseline_rotation_delta_deg << "\n";
  output << "pair_only_baseline_translation_delta_m: "
         << result.pair_init_summary.baseline_translation_delta_m << "\n";
  output << "pair_only_robust_loss_enabled: "
         << (result.pair_init_summary.robust_loss_enabled ? 1 : 0) << "\n";
  output << "pair_only_objective_finite: "
         << (result.pair_init_summary.objective_finite ? 1 : 0) << "\n";
  output << "pair_only_objective_decreased: "
         << (result.pair_init_summary.objective_decreased ? 1 : 0) << "\n";
  output << "pair_only_linear_solver_failure: "
         << (result.pair_init_summary.linear_solver_failure ? 1 : 0) << "\n";
  output << "pair_only_reached_max_iterations: "
         << (result.pair_init_summary.reached_max_iterations ? 1 : 0) << "\n";
  output << "pair_only_optimization_iterations: "
         << result.pair_init_summary.optimization_iterations << "\n";
  output << "pair_only_objective_start: "
         << result.pair_init_summary.objective_start << "\n";
  output << "pair_only_objective_final: "
         << result.pair_init_summary.objective_final << "\n";
  output << "stage6_initialization_role: seed_only_no_selection\n";
  for (int pair_index : result.initialization.reachable_training_pair_indices) {
    output << "reachable_training_pair: " << pair_index << "\n";
  }
  for (int pair_index : result.initialization.unreachable_training_pair_indices) {
    output << "unreachable_training_pair: " << pair_index << "\n";
  }
  for (int board_id : result.initialization.reachable_board_ids) {
    output << "reachable_board_id: " << board_id << "\n";
  }
  for (int board_id : result.initialization.unreachable_board_ids) {
    output << "unreachable_board_id: " << board_id << "\n";
  }
  for (const std::string& reason : result.initialization.excluded_candidate_reasons) {
    output << "excluded_candidate_reason: " << reason << "\n";
  }
  for (const std::string& warning : result.initialization.warnings) {
    output << "warning: " << warning << "\n";
  }
}

void WriteStereoGraphSummary(const std::string& path,
                             const StereoExtrinsicCalibrationResult& result) {
  std::ofstream output(path.c_str());
  output << "training_pair_count: "
         << result.problem_input.measurement_dataset.training_pair_indices.size() << "\n";
  output << "reachable_training_pair_count: "
         << result.initialization.reachable_training_pair_count << "\n";
  output << "unreachable_training_pair_count: "
         << result.initialization.unreachable_training_pair_count << "\n";
  output << "connected_component_count: "
         << result.initialization.connected_component_count << "\n";
  output << "gauge_connected_component_id: "
         << result.initialization.gauge_connected_component_id << "\n";
  output << "gauge_connected_pair_count: "
         << result.initialization.gauge_connected_pair_count << "\n";
  output << "gauge_connected_board_count: "
         << result.initialization.gauge_connected_board_count << "\n";
  output << "initialized_board_count: "
         << result.initialization.initialized_board_count << "\n";
  output << "uninitialized_board_count: "
         << result.initialization.uninitialized_board_count << "\n";
  output << "graph_seed_pair_count: "
         << result.initialization.graph_seed_pair_count << "\n";
  output << "graph_propagation_iteration_count: "
         << result.initialization.graph_propagation_iteration_count << "\n";
  output << "graph_propagation_new_pair_count: "
         << result.initialization.graph_propagation_new_pair_count << "\n";
  output << "graph_propagation_new_board_count: "
         << result.initialization.graph_propagation_new_board_count << "\n";
  output << "graph_propagation_stopped_by_no_progress: "
         << (result.initialization.graph_propagation_stopped_by_no_progress ? 1 : 0)
         << "\n";
  output << "graph_propagation_stopped_by_iteration_limit: "
         << (result.initialization.graph_propagation_stopped_by_iteration_limit ? 1 : 0)
         << "\n";
  for (int pair_index : result.initialization.reachable_training_pair_indices) {
    output << "reachable_training_pair: " << pair_index << "\n";
  }
  for (int pair_index : result.initialization.unreachable_training_pair_indices) {
    output << "unreachable_training_pair: " << pair_index << "\n";
  }
  for (int board_id : result.initialization.reachable_board_ids) {
    output << "reachable_board_id: " << board_id << "\n";
  }
  for (int board_id : result.initialization.unreachable_board_ids) {
    output << "unreachable_board_id: " << board_id << "\n";
  }
  for (const auto& entry : result.initialization.pair_component_ids) {
    output << "pair_component: " << entry.first << "," << entry.second << "\n";
  }
  for (const auto& entry : result.initialization.board_component_ids) {
    output << "board_component: " << entry.first << "," << entry.second << "\n";
  }
}

void WriteStage6RuntimeSummary(const std::string& path,
                               const StereoExtrinsicCalibrationResult& result) {
  std::ofstream output(path.c_str());
  output << "pair_pose_refit_mode: "
         << StereoPairPoseRefitModeToString(
                result.problem_input.solver_options.pair_pose_refit_mode) << "\n";
  output << "cache_dir: " << result.runtime_summary.cache_dir << "\n";
  output << "cache_enabled: "
         << (result.runtime_summary.cache_enabled ? 1 : 0) << "\n";
  output << "total_runtime_seconds: "
         << result.runtime_summary.total_runtime_seconds << "\n";
  output << "pairing_build_dataset_runtime_seconds: "
         << result.runtime_summary.pairing_build_dataset_runtime_seconds << "\n";
  output << "initialization_runtime_seconds: "
         << result.runtime_summary.initialization_runtime_seconds << "\n";
  output << "training_optimization_runtime_seconds: "
         << result.runtime_summary.training_optimization_runtime_seconds << "\n";
  output << "global_sparse_ba_runtime_seconds: "
         << result.runtime_summary.global_sparse_ba_runtime_seconds << "\n";
  output << "holdout_evaluation_runtime_seconds: "
         << result.runtime_summary.holdout_evaluation_runtime_seconds << "\n";
  output << "cam0_training_detection_cache_hits: "
         << result.runtime_summary.cam0_training_detection_cache_hits << "\n";
  output << "cam0_training_detection_cache_misses: "
         << result.runtime_summary.cam0_training_detection_cache_misses << "\n";
  output << "cam0_training_detection_cache_load_failures: "
         << result.runtime_summary.cam0_training_detection_cache_load_failures
         << "\n";
  output << "cam0_training_detection_cache_store_failures: "
         << result.runtime_summary.cam0_training_detection_cache_store_failures
         << "\n";
  output << "cam1_training_detection_cache_hits: "
         << result.runtime_summary.cam1_training_detection_cache_hits << "\n";
  output << "cam1_training_detection_cache_misses: "
         << result.runtime_summary.cam1_training_detection_cache_misses << "\n";
  output << "cam1_training_detection_cache_load_failures: "
         << result.runtime_summary.cam1_training_detection_cache_load_failures
         << "\n";
  output << "cam1_training_detection_cache_store_failures: "
         << result.runtime_summary.cam1_training_detection_cache_store_failures
         << "\n";
  output << "frontend_pairing_prefilter_enabled: "
         << (result.runtime_summary.frontend_pairing_prefilter_enabled ? 1 : 0)
         << "\n";
  output << "frontend_original_left_frame_count: "
         << result.runtime_summary.frontend_original_left_frame_count << "\n";
  output << "frontend_original_right_frame_count: "
         << result.runtime_summary.frontend_original_right_frame_count << "\n";
  output << "frontend_processed_left_frame_count: "
         << result.runtime_summary.frontend_processed_left_frame_count << "\n";
  output << "frontend_processed_right_frame_count: "
         << result.runtime_summary.frontend_processed_right_frame_count << "\n";
  output << "frontend_skipped_unpaired_left_frame_count: "
         << result.runtime_summary.frontend_skipped_unpaired_left_frame_count
         << "\n";
  output << "frontend_skipped_unpaired_right_frame_count: "
         << result.runtime_summary.frontend_skipped_unpaired_right_frame_count
         << "\n";
  output << "symmetric_refit_call_count: "
         << result.runtime_summary.symmetric_refit_call_count << "\n";
  output << "symmetric_refit_improved_count: "
         << result.runtime_summary.symmetric_refit_improved_count << "\n";
  output << "symmetric_refit_fallback_count: "
         << result.runtime_summary.symmetric_refit_fallback_count << "\n";
  output << "max_graph_propagation_iterations: "
         << result.runtime_summary.max_graph_propagation_iterations << "\n";
  output << "graph_propagation_iteration_count: "
         << result.runtime_summary.graph_propagation_iteration_count << "\n";
  output << "graph_propagation_new_pair_count: "
         << result.runtime_summary.graph_propagation_new_pair_count << "\n";
  output << "graph_propagation_new_board_count: "
         << result.runtime_summary.graph_propagation_new_board_count << "\n";
  output << "graph_propagation_stopped_by_no_progress: "
         << (result.runtime_summary.graph_propagation_stopped_by_no_progress ? 1 : 0)
         << "\n";
  output << "graph_propagation_stopped_by_iteration_limit: "
         << (result.runtime_summary.graph_propagation_stopped_by_iteration_limit ? 1 : 0)
         << "\n";
  output << "runtime_guard_trigger_count: "
         << result.runtime_summary.runtime_guard_trigger_count << "\n";
}

const char* ToString(StereoPairPoseRefitMode mode) {
  return StereoPairPoseRefitModeToString(mode);
}

const char* ToString(StereoViewSelectionMode mode) {
  switch (mode) {
    case StereoViewSelectionMode::Off:
      return "off";
    case StereoViewSelectionMode::TopK:
      return "topk";
    case StereoViewSelectionMode::KalibrStyleTrial:
      return "kalibr_style_trial";
  }
  return "unknown";
}

const char* ToString(StereoSolverMode mode) {
  switch (mode) {
    case StereoSolverMode::Alternating:
      return "alternating";
    case StereoSolverMode::GlobalSparseBa:
      return "global_sparse_ba";
    case StereoSolverMode::SharedOnlyGlobalSparseBa:
      return "shared_only_global_sparse_ba";
  }
  return "unknown";
}

const char* ToString(StereoSingleCameraOnlyWeightMode mode) {
  switch (mode) {
    case StereoSingleCameraOnlyWeightMode::FixedScale:
      return "fixed_scale";
    case StereoSingleCameraOnlyWeightMode::PerSideBudgetCap:
      return "per_side_budget_cap";
    case StereoSingleCameraOnlyWeightMode::AdaptiveIndependentSideCap:
      return "adaptive_independent_side_cap";
  }
  return "unknown";
}

const char* ToString(StereoPairBoardSelectionMode mode) {
  switch (mode) {
    case StereoPairBoardSelectionMode::StrictRmse:
      return "strict_rmse";
    case StereoPairBoardSelectionMode::KalibrStyleBatch:
      return "kalibr_style_batch";
  }
  return "unknown";
}

const char* ToString(StereoFinalBaResidualMode mode) {
  switch (mode) {
    case StereoFinalBaResidualMode::Pixel:
      return "pixel";
    case StereoFinalBaResidualMode::SphericalChordal:
      return "spherical_chordal";
    case StereoFinalBaResidualMode::SphericalTangent:
      return "spherical_tangent";
    case StereoFinalBaResidualMode::HybridPixelSpherical:
      return "hybrid_pixel_spherical";
  }
  return "unknown";
}

const char* ToString(StereoSphericalUncertaintyMode mode) {
  switch (mode) {
    case StereoSphericalUncertaintyMode::None:
      return "none";
    case StereoSphericalUncertaintyMode::Pixel:
      return "pixel";
    case StereoSphericalUncertaintyMode::Model:
      return "model";
    case StereoSphericalUncertaintyMode::PixelModel:
      return "pixel_model";
  }
  return "unknown";
}

const char* ToString(StereoCandidateBudgetMode mode) {
  switch (mode) {
    case StereoCandidateBudgetMode::Fixed:
      return "fixed";
    case StereoCandidateBudgetMode::Adaptive:
      return "adaptive";
    case StereoCandidateBudgetMode::KalibrStyle:
      return "kalibr_style";
  }
  return "unknown";
}

const char* ToString(StereoSelectionOptimizationMode mode) {
  switch (mode) {
    case StereoSelectionOptimizationMode::TrialBaCommit:
      return "trial_ba_commit";
    case StereoSelectionOptimizationMode::TrialBaNoCommit:
      return "trial_ba_no_commit";
    case StereoSelectionOptimizationMode::InformationGainOnly:
      return "information_gain_only";
  }
  return "unknown";
}

const char* ToString(StereoRigParamMode mode) {
  switch (mode) {
    case StereoRigParamMode::Cam0Reference:
      return "cam0_reference";
    case StereoRigParamMode::RigCentricSymmetric:
      return "rig_centric_symmetric";
  }
  return "unknown";
}

const char* ToString(StereoSingleBoardPairPolicy policy) {
  switch (policy) {
    case StereoSingleBoardPairPolicy::Keep:
      return "keep";
    case StereoSingleBoardPairPolicy::Audit:
      return "audit";
    case StereoSingleBoardPairPolicy::Drop:
      return "drop";
    case StereoSingleBoardPairPolicy::LowWeight:
      return "low_weight";
  }
  return "unknown";
}

void WriteStereoPairSelectionSummary(const std::string& path,
                                     const StereoExtrinsicCalibrationResult& result) {
  std::ofstream output(path.c_str());
  output << "success: " << (result.pair_selection_summary.success ? 1 : 0) << "\n";
  output << "failure_reason: " << result.pair_selection_summary.failure_reason << "\n";
  output << "mode: " << ToString(result.pair_selection_summary.mode) << "\n";
  output << "requested_pair_count: "
         << result.pair_selection_summary.requested_pair_count << "\n";
  output << "eligible_pair_count: "
         << result.pair_selection_summary.eligible_pair_count << "\n";
  output << "selected_pair_count: "
         << result.pair_selection_summary.selected_pair_count << "\n";
  output << "selected_shared_board_pair_count: "
         << result.pair_selection_summary.selected_shared_board_pair_count << "\n";
  output << "selected_single_camera_only_pair_count: "
         << result.pair_selection_summary.selected_single_camera_only_pair_count << "\n";
  output << "reachable_pair_count: "
         << result.pair_selection_summary.reachable_pair_count << "\n";
  output << "initialized_pair_count: "
         << result.pair_selection_summary.initialized_pair_count << "\n";
  output << "selected_covered_board_count: "
         << result.pair_selection_summary.selected_covered_board_count << "\n";
  output << "selected_pose_fit_rmse_min: "
         << result.pair_selection_summary.selected_pose_fit_rmse_min << "\n";
  output << "selected_pose_fit_rmse_median: "
         << result.pair_selection_summary.selected_pose_fit_rmse_median << "\n";
  output << "selected_pose_fit_rmse_max: "
         << result.pair_selection_summary.selected_pose_fit_rmse_max << "\n";
  output << "shared_board_quality_gate_enabled: "
         << (result.problem_input.solver_options.enable_shared_board_quality_gate ? 1 : 0)
         << "\n";
  output << "shared_board_quality_hard_gate_enabled: "
         << (result.problem_input.solver_options.enable_shared_board_quality_hard_gate ? 1 : 0)
         << "\n";
  output << "shared_board_quality_max_outer_rmse_px: "
         << result.problem_input.solver_options.shared_board_quality_max_outer_rmse_px
         << "\n";
  output << "shared_board_quality_filter_final_ba: "
         << (result.problem_input.solver_options.shared_board_quality_filter_final_ba ? 1 : 0)
         << "\n";
  output << "shared_board_quality_min_outer_points_per_camera: "
         << result.problem_input.solver_options
                .shared_board_quality_min_outer_points_per_camera
         << "\n";
  output << "shared_board_quality_min_good_shared_boards: "
         << result.problem_input.solver_options.shared_board_quality_min_good_shared_boards
         << "\n";
  for (int board_id : result.pair_selection_summary.covered_board_ids) {
    output << "selected_covered_board_id: " << board_id << "\n";
  }
  for (int pair_index : result.pair_selection_summary.selected_pair_indices) {
    output << "selected_pair: " << pair_index << "\n";
  }
}

void WriteStereoPairSelectionCsv(const std::string& path,
                                 const StereoExtrinsicCalibrationResult& result) {
  std::ofstream output(path.c_str());
  output << "pair_index,eligible,selected,score_shared_board_count,"
         << "reachable,initialized,covered_board_count,missing_board_coverage_count,"
         << "score_shared_outer_point_count,score_pose_fit_rmse,"
         << "score_single_camera_only_board_count,shared_board_count,"
         << "shared_outer_point_count,pose_fit_rmse,single_camera_only_board_count,"
         << "covered_board_ids,rejection_reason\n";
  for (const StereoPairSelectionRow& row : result.pair_selection_summary.rows) {
    std::ostringstream covered_board_ids;
    for (std::size_t i = 0; i < row.covered_board_ids.size(); ++i) {
      if (i > 0) {
        covered_board_ids << "|";
      }
      covered_board_ids << row.covered_board_ids[i];
    }
    output << row.pair_index << "," << (row.eligible ? 1 : 0) << ","
           << (row.selected ? 1 : 0) << ","
           << row.score_shared_board_count << ","
           << (row.reachable ? 1 : 0) << ","
           << (row.initialized ? 1 : 0) << ","
           << row.covered_board_count << ","
           << row.missing_board_coverage_count << ","
           << row.score_shared_outer_point_count << "," << row.score_pose_fit_rmse
           << "," << row.score_single_camera_only_board_count << ","
           << row.shared_board_count << "," << row.shared_outer_point_count << ","
           << row.pose_fit_rmse << "," << row.single_camera_only_board_count << ","
           << covered_board_ids.str() << ","
           << row.rejection_reason << "\n";
  }
}

void WriteStereoPairInitSummary(const std::string& path,
                                const StereoExtrinsicCalibrationResult& result) {
  std::ofstream output(path.c_str());
  output << "enabled: " << (result.pair_init_summary.enabled ? 1 : 0) << "\n";
  output << "success: " << (result.pair_init_summary.success ? 1 : 0) << "\n";
  output << "failure_reason: " << result.pair_init_summary.failure_reason << "\n";
  output << "raw_candidate_count: " << result.pair_init_summary.raw_candidate_count << "\n";
  output << "consistency_filtered_candidate_count: "
         << result.pair_init_summary.consistency_filtered_candidate_count << "\n";
  output << "consistency_rejected_candidate_count: "
         << result.pair_init_summary.consistency_rejected_candidate_count << "\n";
  output << "failed_pair_board_count: "
         << result.pair_init_summary.failed_pair_board_count << "\n";
  output << "medoid_baseline_length: "
         << result.pair_init_summary.medoid_baseline_length << "\n";
  output << "pair_ba_baseline_length: "
         << result.pair_init_summary.pair_ba_baseline_length << "\n";
  output << "before_shared_rmse: " << result.pair_init_summary.before_shared_rmse
         << "\n";
  output << "after_shared_rmse: " << result.pair_init_summary.after_shared_rmse
         << "\n";
  output << "baseline_rotation_delta_deg: "
         << result.pair_init_summary.baseline_rotation_delta_deg << "\n";
  output << "baseline_translation_delta_m: "
         << result.pair_init_summary.baseline_translation_delta_m << "\n";
  output << "used_refined_baseline: "
         << (result.pair_init_summary.used_refined_baseline ? 1 : 0) << "\n";
  output << "robust_loss_enabled: "
         << (result.pair_init_summary.robust_loss_enabled ? 1 : 0) << "\n";
  output << "objective_finite: "
         << (result.pair_init_summary.objective_finite ? 1 : 0) << "\n";
  output << "objective_decreased: "
         << (result.pair_init_summary.objective_decreased ? 1 : 0) << "\n";
  output << "linear_solver_failure: "
         << (result.pair_init_summary.linear_solver_failure ? 1 : 0) << "\n";
  output << "reached_max_iterations: "
         << (result.pair_init_summary.reached_max_iterations ? 1 : 0) << "\n";
  output << "optimization_iterations: "
         << result.pair_init_summary.optimization_iterations << "\n";
  output << "objective_start: "
         << result.pair_init_summary.objective_start << "\n";
  output << "objective_final: "
         << result.pair_init_summary.objective_final << "\n";
  output << "shared_board_quality_gate_enabled: "
         << (result.problem_input.solver_options.enable_shared_board_quality_gate ? 1 : 0)
         << "\n";
  output << "shared_board_quality_hard_gate_enabled: "
         << (result.problem_input.solver_options.enable_shared_board_quality_hard_gate ? 1 : 0)
         << "\n";
  output << "shared_board_quality_max_outer_rmse_px: "
         << result.problem_input.solver_options.shared_board_quality_max_outer_rmse_px
         << "\n";
  output << "shared_board_quality_filter_final_ba: "
         << (result.problem_input.solver_options.shared_board_quality_filter_final_ba ? 1 : 0)
         << "\n";
  for (const std::string& warning : result.pair_init_summary.warnings) {
    output << "warning: " << warning << "\n";
  }
}

void WriteStereoPairInitCandidatesCsv(const std::string& path,
                                      const StereoExtrinsicCalibrationResult& result) {
  std::ofstream output(path.c_str());
  output << "pair_index,board_id,raw_candidate,consistency_accepted,"
         << "cam0_outer_rmse,cam1_outer_rmse,shared_outer_point_count,"
         << "candidate_baseline_length,reject_reason\n";
  for (const StereoPairInitCandidateRow& row : result.pair_init_summary.candidates) {
    output << row.pair_index << "," << row.board_id << ","
           << (row.raw_candidate ? 1 : 0) << ","
           << (row.consistency_accepted ? 1 : 0) << ","
           << row.cam0_outer_rmse << "," << row.cam1_outer_rmse << ","
           << row.shared_outer_point_count << ","
           << row.candidate_baseline_length << ","
           << row.reject_reason << "\n";
  }
}

void WriteStereoPairInitResidualsCsv(const std::string& path,
                                     const StereoExtrinsicCalibrationResult& result) {
  std::ofstream output(path.c_str());
  output << "pair_index,board_id,shared_point_count,before_rmse,after_rmse\n";
  for (const StereoPairInitResidualRow& row : result.pair_init_summary.residual_rows) {
    output << row.pair_index << "," << row.board_id << ","
           << row.shared_point_count << "," << row.before_rmse << ","
           << row.after_rmse << "\n";
  }
}

void WriteStereoPairTrialSelectionSummary(
    const std::string& path,
    const StereoExtrinsicCalibrationResult& result) {
  std::ofstream output(path.c_str());
  output << "enabled: "
         << (result.pair_trial_selection_summary.enabled ? 1 : 0) << "\n";
  output << "selection_method: "
         << (result.problem_input.solver_options
                     .enable_persistent_incremental_stereo_ba
                 ? "persistent_incremental_pair_cohesive_selection"
                 : (result.problem_input.solver_options
                            .enable_committing_pair_batch_selection
                        ? "kalibr_style_incremental_batch_acceptance_pair_level"
                        : "kalibr_style_pair_trial_selection"))
         << "\n";
  output << "incremental_estimator_enabled: "
         << (result.problem_input.solver_options
                     .enable_stage6_incremental_estimator
                 ? 1
                 : 0)
         << "\n";
  output << "marginal_information_gain_proxy_enabled: "
         << (result.problem_input.solver_options
                     .enable_committing_pair_batch_selection
                 ? 1
                 : 0)
         << "\n";
  output << "rmse_delta_diagnostics_only: "
         << (result.problem_input.solver_options
                     .enable_committing_pair_batch_selection ||
             result.problem_input.solver_options
                     .enable_persistent_incremental_stereo_ba
                 ? 1
                 : 0)
         << "\n";
  output << "incremental_mi_tol: "
         << result.problem_input.solver_options.incremental_mi_tol << "\n";
  output << "incremental_rank_threshold: "
         << result.problem_input.solver_options.incremental_rank_threshold
         << "\n";
  output << "incremental_info_block: "
         << ToString(result.problem_input.solver_options.incremental_info_block)
         << "\n";
  output << "selection_ba_residual_mode: "
         << ToString(result.problem_input.solver_options
                         .selection_ba_residual_mode)
         << "\n";
  output << "selection_coobs_factor_ba_enable: "
         << (result.problem_input.solver_options
                     .selection_coobs_factor_ba_enable
                 ? 1
                 : 0)
         << "\n";
  output << "selection_coobs_factor_ba_apply_stereo_factor: "
         << (result.problem_input.solver_options
                     .selection_coobs_factor_ba_apply_stereo_factor
                 ? 1
                 : 0)
         << "\n";
  output << "selection_coobs_factor_ba_apply_layout_factor: "
         << (result.problem_input.solver_options
                     .selection_coobs_factor_ba_apply_layout_factor
                 ? 1
                 : 0)
         << "\n";
  output << "selection_coobs_factor_ba_stereo_weight: "
         << result.problem_input.solver_options
                .selection_coobs_factor_ba_stereo_weight
         << "\n";
  output << "selection_coobs_factor_ba_layout_weight: "
         << result.problem_input.solver_options
                .selection_coobs_factor_ba_layout_weight
         << "\n";
  output << "coobs_aware_acceptance_enable: "
         << (result.problem_input.solver_options
                     .coobs_aware_acceptance_enable
                 ? 1
                 : 0)
         << "\n";
  output << "coobs_aware_acceptance_min_score: "
         << result.problem_input.solver_options
                .coobs_aware_acceptance_min_score
         << "\n";
  output << "coobs_aware_acceptance_max_total_rmse_delta: "
         << result.problem_input.solver_options
                .coobs_aware_acceptance_max_total_rmse_delta
         << "\n";
  output << "coobs_aware_acceptance_max_camera_rmse_delta: "
         << result.problem_input.solver_options
                .coobs_aware_acceptance_max_camera_rmse_delta
         << "\n";
  output << "coobs_aware_acceptance_balance_guard_enable: "
         << (result.problem_input.solver_options
                     .coobs_aware_acceptance_balance_guard_enable
                 ? 1
                 : 0)
         << "\n";
  output << "coobs_aware_acceptance_max_camera_delta_imbalance: "
         << result.problem_input.solver_options
                .coobs_aware_acceptance_max_camera_delta_imbalance
         << "\n";
  output << "coobs_aware_acceptance_max_camera_delta_ratio: "
         << result.problem_input.solver_options
                .coobs_aware_acceptance_max_camera_delta_ratio
         << "\n";
  output << "coobs_aware_acceptance_require_pair_completion: "
         << (result.problem_input.solver_options
                     .coobs_aware_acceptance_require_pair_completion
                 ? 1
                 : 0)
         << "\n";
  output << "success: "
         << (result.pair_trial_selection_summary.success ? 1 : 0) << "\n";
  output << "failure_reason: "
         << result.pair_trial_selection_summary.failure_reason << "\n";
  output << "requested_seed_count: "
         << result.pair_trial_selection_summary.requested_seed_count << "\n";
  output << "seed_count: " << result.pair_trial_selection_summary.seed_count << "\n";
  output << "candidate_count: "
         << result.pair_trial_selection_summary.candidate_count << "\n";
  output << "budget_mode: "
         << ToString(result.pair_trial_selection_summary.budget_mode)
         << "\n";
  output << "valid_candidate_count: "
         << result.pair_trial_selection_summary.valid_candidate_count
         << "\n";
  output << "valid_candidate_traversed_count: "
         << result.pair_trial_selection_summary.valid_candidate_traversed_count
         << "\n";
  output << "safety_ceiling_hit: "
         << (result.pair_trial_selection_summary.safety_ceiling_hit ? 1 : 0)
         << "\n";
  output << "runtime_safety_ceiling: "
         << result.pair_trial_selection_summary.runtime_safety_ceiling
         << "\n";
  output << "max_candidate_additions_effective: "
         << result.pair_trial_selection_summary
                .max_candidate_additions_effective
         << "\n";
  output << "attempted_count: "
         << result.pair_trial_selection_summary.attempted_count << "\n";
  output << "accepted_count: "
         << result.pair_trial_selection_summary.accepted_count << "\n";
  output << "rejected_count: "
         << result.pair_trial_selection_summary.rejected_count << "\n";
  output << "final_selected_pair_count: "
         << result.pair_trial_selection_summary.final_selected_pair_count << "\n";
  output << "initial_seed_rmse: "
         << result.pair_trial_selection_summary.initial_seed_rmse << "\n";
  output << "final_selected_rmse: "
         << result.pair_trial_selection_summary.final_selected_rmse << "\n";
  for (const std::string& warning :
       result.pair_trial_selection_summary.warnings) {
    output << "warning: " << warning << "\n";
  }
}

void WriteStereoPairTrialSelectionDecisionsCsv(
    const std::string& path,
    const StereoExtrinsicCalibrationResult& result) {
  std::ofstream output(path.c_str());
  output << "pair_index,left_frame_label,right_frame_label,shared_board_count,"
         << "cam0_only_board_count,cam1_only_board_count,shared_outer_point_count,"
         << "shared_internal_point_count,candidate_score,coverage_gain,seed,"
         << "attempted,accepted,batchAccepted,initial_total_rmse,"
         << "trial_total_rmse,total_rmse_delta,"
         << "cam0_rmse_delta,cam1_rmse_delta,baseline_rotation_delta_deg,"
         << "baseline_translation_delta_m,incremental_estimator_enabled,"
         << "persistent_incremental_estimator_used,"
         << "candidate_batch_type,accept_reason,solution_valid,"
         << "optimization_success,num_iterations,objective_before,"
         << "objective_after,marginal_information_gain_proxy,rank_before,"
         << "normalized_information_gain,"
         << "information_gain_normalization_count,rank_after,"
         << "rank_psi_after,rank_psi_deficiency_after,"
         << "rank_theta_deficiency_after,rank_proxy_increases,"
         << "svd_tolerance,qr_tolerance,elapsed_time_seconds,"
         << "info_gain_threshold,"
         << "committed_or_rollback,trial_coobs_stereo_factor_count,"
         << "trial_coobs_layout_factor_count,"
         << "trial_coobs_stereo_initial_rot_mean_deg,"
         << "trial_coobs_stereo_initial_rot_max_deg,"
         << "trial_coobs_stereo_initial_trans_mean_m,"
         << "trial_coobs_stereo_initial_trans_max_m,"
         << "trial_coobs_layout_initial_rot_mean_deg,"
         << "trial_coobs_layout_initial_rot_max_deg,"
         << "trial_coobs_layout_initial_trans_mean_m,"
         << "trial_coobs_layout_initial_trans_max_m,"
         << "coobs_acceptance_score,coobs_acceptance_health_pass,"
         << "coobs_acceptance_structure_pass,"
         << "coobs_acceptance_balance_pass,"
         << "coobs_acceptance_camera_delta_imbalance,"
         << "coobs_acceptance_camera_delta_ratio,"
         << "accepted_by_coobs_aware_acceptance,"
         << "force,reject_reason\n";
  for (const StereoPairTrialSelectionDecision& row :
       result.pair_trial_selection_summary.decisions) {
    output << row.pair_index << "," << row.left_frame_label << ","
           << row.right_frame_label << "," << row.shared_board_count << ","
           << row.cam0_only_board_count << "," << row.cam1_only_board_count << ","
           << row.shared_outer_point_count << "," << row.shared_internal_point_count
           << "," << row.candidate_score << "," << row.coverage_gain << ","
           << (row.seed ? 1 : 0) << "," << (row.attempted ? 1 : 0) << ","
           << (row.accepted ? 1 : 0) << "," << (row.batchAccepted ? 1 : 0)
           << "," << row.initial_total_rmse << ","
           << row.trial_total_rmse << "," << row.total_rmse_delta << ","
           << row.cam0_rmse_delta << "," << row.cam1_rmse_delta << ","
           << row.baseline_rotation_delta_deg << ","
           << row.baseline_translation_delta_m << ","
           << (row.incremental_estimator_enabled ? 1 : 0) << ","
           << (row.persistent_incremental_estimator_used ? 1 : 0) << ","
           << row.candidate_batch_type << "," << row.accept_reason << ","
           << (row.solution_valid ? 1 : 0) << ","
           << (row.optimization_success ? 1 : 0) << ","
           << row.num_iterations << "," << row.objective_before << ","
           << row.objective_after << ","
           << row.marginal_information_gain_proxy << ","
           << row.rank_before << "," << row.normalized_information_gain << ","
           << row.information_gain_normalization_count << ","
           << row.rank_after << "," << row.rank_psi_after << ","
           << row.rank_psi_deficiency_after << ","
           << row.rank_theta_deficiency_after << ","
           << (row.rank_proxy_increases ? 1 : 0) << ","
           << row.svd_tolerance << "," << row.qr_tolerance << ","
           << row.elapsed_time_seconds << ","
           << row.info_gain_threshold << "," << row.committed_or_rollback
           << "," << row.trial_coobs_stereo_factor_count << ","
           << row.trial_coobs_layout_factor_count << ","
           << row.trial_coobs_stereo_initial_rot_mean_deg << ","
           << row.trial_coobs_stereo_initial_rot_max_deg << ","
           << row.trial_coobs_stereo_initial_trans_mean_m << ","
           << row.trial_coobs_stereo_initial_trans_max_m << ","
           << row.trial_coobs_layout_initial_rot_mean_deg << ","
           << row.trial_coobs_layout_initial_rot_max_deg << ","
           << row.trial_coobs_layout_initial_trans_mean_m << ","
           << row.trial_coobs_layout_initial_trans_max_m << ","
           << row.coobs_acceptance_score << ","
           << (row.coobs_acceptance_health_pass ? 1 : 0) << ","
           << (row.coobs_acceptance_structure_pass ? 1 : 0) << ","
           << (row.coobs_acceptance_balance_pass ? 1 : 0) << ","
           << row.coobs_acceptance_camera_delta_imbalance << ","
           << row.coobs_acceptance_camera_delta_ratio << ","
           << (row.accepted_by_coobs_aware_acceptance ? 1 : 0) << ","
           << (row.force ? 1 : 0) << "," << row.reject_reason
           << "\n";
  }
}

void WriteStage6PersistentIncrementalBatchDecisionsCsv(
    const std::string& path,
    const StereoExtrinsicCalibrationResult& result) {
  std::ofstream output(path.c_str());
  output << "batch_index,pair_index,left_frame_label,right_frame_label,"
         << "batch_type,seed,force,attempted,accepted,batchAccepted,"
         << "committed_or_rollback,shared_board_count,"
         << "selected_pair_board_count,selected_board_ids,"
         << "shared_outer_point_count,shared_internal_point_count,"
         << "candidate_score,coverage_gain,initial_total_rmse,"
         << "trial_total_rmse,total_rmse_delta,cam0_rmse_delta,"
         << "cam1_rmse_delta,baseline_rotation_delta_deg,"
         << "baseline_translation_delta_m,solution_valid,"
         << "optimization_success,num_iterations,JStart,JFinal,"
         << "information_gain,normalized_information_gain,"
         << "information_gain_normalization_count,rank_before,rankTheta,"
         << "rankPsi,rankPsiDeficiency,rankThetaDeficiency,"
         << "rank_increases,svd_tolerance,qr_tolerance,"
         << "elapsed_time_seconds,info_gain_threshold,accept_reason,"
         << "reject_reason\n";

  std::map<int, std::vector<int> > selected_boards_by_pair;
  for (const PairBoardKey& key :
       result.pair_board_trial_selection_summary.selected_pair_board_keys) {
    selected_boards_by_pair[key.first].push_back(key.second);
  }
  for (auto& entry : selected_boards_by_pair) {
    std::sort(entry.second.begin(), entry.second.end());
  }

  int batch_index = 0;
  for (const StereoPairTrialSelectionDecision& row :
       result.pair_trial_selection_summary.decisions) {
    const auto boards_it = selected_boards_by_pair.find(row.pair_index);
    const std::vector<int> empty_boards;
    const std::vector<int>& boards =
        boards_it == selected_boards_by_pair.end() ? empty_boards
                                                   : boards_it->second;
    output << batch_index++ << "," << row.pair_index << ","
           << row.left_frame_label << "," << row.right_frame_label << ","
           << row.candidate_batch_type << "," << (row.seed ? 1 : 0) << ","
           << (row.force ? 1 : 0) << "," << (row.attempted ? 1 : 0) << ","
           << (row.accepted ? 1 : 0) << "," << (row.batchAccepted ? 1 : 0)
           << "," << row.committed_or_rollback << ","
           << row.shared_board_count << "," << boards.size() << ","
           << JoinInts(boards, ';') << "," << row.shared_outer_point_count
           << "," << row.shared_internal_point_count << ","
           << row.candidate_score << "," << row.coverage_gain << ","
           << row.initial_total_rmse << "," << row.trial_total_rmse << ","
           << row.total_rmse_delta << "," << row.cam0_rmse_delta << ","
           << row.cam1_rmse_delta << ","
           << row.baseline_rotation_delta_deg << ","
           << row.baseline_translation_delta_m << ","
           << (row.solution_valid ? 1 : 0) << ","
           << (row.optimization_success ? 1 : 0) << ","
           << row.num_iterations << "," << row.objective_before << ","
           << row.objective_after << ","
           << row.marginal_information_gain_proxy << ","
           << row.normalized_information_gain << ","
           << row.information_gain_normalization_count << ","
           << row.rank_before << "," << row.rank_after << ","
           << row.rank_psi_after << "," << row.rank_psi_deficiency_after
           << "," << row.rank_theta_deficiency_after << ","
           << (row.rank_proxy_increases ? 1 : 0) << ","
           << row.svd_tolerance << "," << row.qr_tolerance << ","
           << row.elapsed_time_seconds << "," << row.info_gain_threshold
           << "," << row.accept_reason << "," << row.reject_reason << "\n";
  }
}

void WriteStereoPairTrialSelectedPairsCsv(
    const std::string& path,
    const StereoExtrinsicCalibrationResult& result) {
  std::ofstream output(path.c_str());
  output << "pair_index\n";
  for (int pair_index : result.pair_trial_selection_summary.selected_pair_indices) {
    output << pair_index << "\n";
  }
}

void WriteStereoPairBoardTrialSelectionSummary(
    const std::string& path,
    const StereoExtrinsicCalibrationResult& result) {
  std::ofstream output(path.c_str());
  output << "enabled: "
         << (result.pair_board_trial_selection_summary.enabled ? 1 : 0) << "\n";
  output << "selection_method: "
         << (result.problem_input.solver_options
                     .enable_persistent_incremental_stereo_ba
                 ? "persistent_incremental_pair_cohesive_selection"
                 : (result.problem_input.solver_options
                            .enable_committing_pair_batch_selection
                        ? "kalibr_style_ba_coupled_batch_selection"
                        : "kalibr_style_pair_board_trial_selection"))
         << "\n";
  output << "incremental_batch_acceptance_enabled: "
         << (result.problem_input.solver_options
                     .enable_committing_pair_batch_selection
                 ? 1
                 : 0)
         << "\n";
  output << "persistent_incremental_batch_acceptance_enabled: "
         << (result.problem_input.solver_options
                     .enable_persistent_incremental_stereo_ba
                 ? 1
                 : 0)
         << "\n";
  output << "incremental_estimator_enabled: "
         << (result.pair_board_trial_selection_summary
                     .incremental_estimator_enabled
                 ? 1
                 : 0)
         << "\n";
  output << "persistent_incremental_estimator_used: "
         << (result.pair_board_trial_selection_summary
                     .persistent_incremental_estimator_used
                 ? 1
                 : 0)
         << "\n";
  output << "persistent_incremental_default_main_path: "
         << (result.problem_input.solver_options
                     .enable_persistent_incremental_stereo_ba
                 ? 1
                 : 0)
         << "\n";
  output << "persistent_incremental_fail_closed: "
         << (result.problem_input.solver_options
                         .enable_persistent_incremental_stereo_ba &&
                     !result.problem_input.solver_options
                          .allow_legacy_selection_fallback_after_persistent_failure
                 ? 1
                 : 0)
         << "\n";
  output << "legacy_selection_fallback_after_persistent_failure_allowed: "
         << (result.problem_input.solver_options
                     .allow_legacy_selection_fallback_after_persistent_failure
                 ? 1
                 : 0)
         << "\n";
  output << "persistent_incremental_uses_real_incremental_estimator: "
         << (result.pair_board_trial_selection_summary
                     .persistent_incremental_estimator_used
                 ? 1
                 : 0)
         << "\n";
  output << "persistent_incremental_batch_unit: pair_cohesive\n";
  output << "persistent_incremental_pose_structure: "
         << result.pair_board_trial_selection_summary
                .persistent_pose_structure
         << "\n";
  output << "persistent_incremental_layout_updates_extrinsic: "
         << (result.problem_input.solver_options.persistent_pose_structure ==
                     StereoPersistentPoseStructure::SharedFrameLayout
                 ? 1 : 0)
         << "\n";
  output << "pair_board_selection_role: "
         << (result.problem_input.solver_options
                     .enable_persistent_incremental_stereo_ba
                 ? "ablation_fallback_diagnostic"
                 : "main_or_legacy_path")
         << "\n";
  output << "legacy_rmse_score_gates_role: "
         << (result.problem_input.solver_options
                     .enable_persistent_incremental_stereo_ba
                 ? "diagnostics_plus_catastrophic_residual_safety_guard"
                 : "legacy_acceptance_path")
         << "\n";
  output << "selection_acceptance_primary_signal: "
         << (result.problem_input.solver_options
                     .enable_persistent_incremental_stereo_ba
                 ? "incremental_estimator_information_rank_objective_validity"
                 : "legacy_score_rmse_gates")
         << "\n";
  output << "coobs_layout_factor_role: "
         << (result.problem_input.solver_options.selection_coobs_factor_ba_enable
                 ? "explicit_ablation_enabled"
                 : "diagnostic_disabled")
         << "\n";
  output << "persistent_incremental_information_group: "
         << result.pair_board_trial_selection_summary
                .persistent_incremental_information_group
         << "\n";
  Stage6PersistentResidualMetric persistent_metric =
      Stage6PersistentResidualMetric::Pixel;
  try {
    persistent_metric = PersistentResidualMetricForMode(
        result.problem_input.solver_options.selection_ba_residual_mode);
  } catch (const std::exception&) {
    // Preserve a readable failed-run summary; the runtime path records the
    // precise unsupported-mode failure reason.
  }
  output << "persistent_incremental_residual_metric_name: "
         << result.pair_board_trial_selection_summary
                .persistent_incremental_residual_metric_name
         << "\n";
  output << "persistent_incremental_final_selected_metric_units: "
         << PersistentResidualMetricUnits(persistent_metric)
         << "\n";
  output << "stage6_intrinsics_mode: "
         << result.pair_board_trial_selection_summary
                .requested_intrinsics_mode
         << "\n";
  output << "stage6_requested_intrinsics_mode: "
         << result.pair_board_trial_selection_summary
                .requested_intrinsics_mode
         << "\n";
  output << "stage6_effective_intrinsics_mode: "
         << result.pair_board_trial_selection_summary
                .effective_intrinsics_mode
         << "\n";
  output << "persistent_incremental_projection_intrinsics_active: "
         << (result.pair_board_trial_selection_summary
                     .projection_intrinsics_active
                 ? 1 : 0) << "\n";
  output << "persistent_incremental_projection_prior_enabled: "
         << (result.pair_board_trial_selection_summary
                     .projection_prior_enabled
                 ? 1 : 0) << "\n";
  output << "persistent_incremental_projection_release_reason: "
         << result.pair_board_trial_selection_summary
                .projection_release_reason
         << "\n";
  output << "persistent_incremental_projection_policy_training_pair_count: "
         << result.pair_board_trial_selection_summary
                .projection_policy_training_pair_count
         << "\n";
  output << "persistent_incremental_projection_policy_shared_pair_board_count: "
         << result.pair_board_trial_selection_summary
                .projection_policy_shared_pair_board_count
         << "\n";
  output << "persistent_incremental_projection_policy_distinct_board_count: "
         << result.pair_board_trial_selection_summary
                .projection_policy_distinct_board_count
         << "\n";
  output << "persistent_incremental_projection_policy_observation_point_count: "
         << result.pair_board_trial_selection_summary
                .projection_policy_observation_point_count
         << "\n";
  output << "persistent_incremental_projection_prior_shape_sigma: "
         << result.pair_board_trial_selection_summary
                .projection_prior_shape_sigma
         << "\n";
  output << "persistent_incremental_projection_prior_focal_relative_sigma: "
         << result.pair_board_trial_selection_summary
                .projection_prior_focal_relative_sigma
         << "\n";
  output << "persistent_incremental_projection_prior_principal_sigma_px: "
         << result.pair_board_trial_selection_summary
                .projection_prior_principal_sigma_px
         << "\n";
  output << "persistent_incremental_distortion_active: "
         << (result.pair_board_trial_selection_summary
                     .distortion_intrinsics_active
                 ? 1 : 0) << "\n";
  output << "persistent_incremental_distortion_prior_enabled: "
         << (result.pair_board_trial_selection_summary
                     .distortion_prior_enabled
                 ? 1 : 0) << "\n";
  output << "persistent_incremental_distortion_prior_sigma: "
         << result.pair_board_trial_selection_summary.distortion_prior_sigma
         << "\n";
  output << "persistent_incremental_requested_seed_pair_count: "
         << result.problem_input.solver_options
                .persistent_incremental_seed_pair_count
         << "\n";
  output << "legacy_pair_selection_seed_count: "
         << result.problem_input.solver_options.pair_selection_seed_count
         << "\n";
  output << "persistent_incremental_seed_batch_count: "
         << result.pair_board_trial_selection_summary
                .persistent_incremental_seed_batch_count
         << "\n";
  output << "persistent_incremental_seed_pair_count: "
         << result.pair_board_trial_selection_summary
                .persistent_incremental_seed_pair_count
         << "\n";
  output << "persistent_incremental_seed_pair_board_count: "
         << result.pair_board_trial_selection_summary
                .persistent_incremental_seed_pair_board_count
         << "\n";
  output << "persistent_incremental_seed_point_count: "
         << result.pair_board_trial_selection_summary
                .persistent_incremental_seed_point_count
         << "\n";
  output << "persistent_incremental_seed_rank_theta: "
         << result.pair_board_trial_selection_summary
                .persistent_incremental_seed_rank_theta
         << "\n";
  output << "persistent_incremental_seed_information_gain: "
         << result.pair_board_trial_selection_summary
                .persistent_incremental_seed_information_gain
         << "\n";
  output << "persistent_incremental_elapsed_time_seconds: "
         << result.pair_board_trial_selection_summary
                .persistent_incremental_elapsed_time_seconds
         << "\n";
  output << "marginal_information_gain_proxy_enabled: "
         << (result.pair_board_trial_selection_summary
                     .marginal_information_gain_proxy_enabled
                 ? 1
                 : 0)
         << "\n";
  output << "rmse_delta_diagnostics_only: "
         << (result.pair_board_trial_selection_summary
                     .rmse_delta_diagnostics_only
                 ? 1
                 : 0)
         << "\n";
  const bool angular_metric =
      persistent_metric == Stage6PersistentResidualMetric::TangentPlaneAngular;
  const double catastrophic_total_delta = angular_metric
      ? 0.02
      : std::max(1.0, 10.0 * result.problem_input.solver_options
                              .pair_selection_max_rmse_delta);
  const double catastrophic_camera_delta = angular_metric
      ? 0.02
      : std::max(1.0, 10.0 * result.problem_input.solver_options
                              .pair_selection_max_camera_rmse_delta);
  output << "persistent_residual_delta_diagnostics_enabled: "
         << (result.problem_input.solver_options
                     .enable_persistent_incremental_stereo_ba
                 ? 1
                 : 0)
         << "\n";
  output << "persistent_catastrophic_residual_guard_enabled: "
         << (result.problem_input.solver_options
                     .enable_persistent_incremental_stereo_ba
                 ? 1
                 : 0)
         << "\n";
  output << "persistent_catastrophic_residual_guard_units: "
         << PersistentResidualMetricUnits(persistent_metric) << "\n";
  output << "persistent_catastrophic_residual_guard_total_delta: "
         << catastrophic_total_delta << "\n";
  output << "persistent_catastrophic_residual_guard_camera_delta: "
         << catastrophic_camera_delta << "\n";
  output << "incremental_mi_tol: "
         << result.pair_board_trial_selection_summary.incremental_mi_tol
         << "\n";
  output << "incremental_rank_threshold: "
         << result.pair_board_trial_selection_summary
                .incremental_rank_threshold
         << "\n";
  output << "incremental_info_block: "
         << result.pair_board_trial_selection_summary.incremental_info_block
         << "\n";
  output << "batch_acceptance_policy: "
         << (result.problem_input.solver_options
                     .enable_persistent_incremental_stereo_ba
                 ? "persistent_incremental_estimator"
                 : ToString(result.problem_input.solver_options
                                .batch_acceptance_policy))
         << "\n";
  output << "legacy_batch_acceptance_policy: "
         << ToString(result.problem_input.solver_options.batch_acceptance_policy)
         << "\n";
  output << "selection_ba_residual_mode: "
         << ToString(result.problem_input.solver_options
                         .selection_ba_residual_mode)
         << "\n";
  output << "selection_coobs_factor_ba_enable: "
         << (result.problem_input.solver_options
                     .selection_coobs_factor_ba_enable
                 ? 1
                 : 0)
         << "\n";
  output << "selection_coobs_factor_ba_apply_stereo_factor: "
         << (result.problem_input.solver_options
                     .selection_coobs_factor_ba_apply_stereo_factor
                 ? 1
                 : 0)
         << "\n";
  output << "selection_coobs_factor_ba_apply_layout_factor: "
         << (result.problem_input.solver_options
                     .selection_coobs_factor_ba_apply_layout_factor
                 ? 1
                 : 0)
         << "\n";
  output << "selection_coobs_factor_ba_stereo_weight: "
         << result.problem_input.solver_options
                .selection_coobs_factor_ba_stereo_weight
         << "\n";
  output << "selection_coobs_factor_ba_layout_weight: "
         << result.problem_input.solver_options
                .selection_coobs_factor_ba_layout_weight
         << "\n";
  output << "coobs_aware_acceptance_enable: "
         << (result.problem_input.solver_options
                     .coobs_aware_acceptance_enable
                 ? 1
                 : 0)
         << "\n";
  output << "coobs_aware_acceptance_min_score: "
         << result.problem_input.solver_options
                .coobs_aware_acceptance_min_score
         << "\n";
  output << "coobs_aware_acceptance_max_total_rmse_delta: "
         << result.problem_input.solver_options
                .coobs_aware_acceptance_max_total_rmse_delta
         << "\n";
  output << "coobs_aware_acceptance_max_camera_rmse_delta: "
         << result.problem_input.solver_options
                .coobs_aware_acceptance_max_camera_rmse_delta
         << "\n";
  output << "coobs_aware_acceptance_balance_guard_enable: "
         << (result.problem_input.solver_options
                     .coobs_aware_acceptance_balance_guard_enable
                 ? 1
                 : 0)
         << "\n";
  output << "coobs_aware_acceptance_max_camera_delta_imbalance: "
         << result.problem_input.solver_options
                .coobs_aware_acceptance_max_camera_delta_imbalance
         << "\n";
  output << "coobs_aware_acceptance_max_camera_delta_ratio: "
         << result.problem_input.solver_options
                .coobs_aware_acceptance_max_camera_delta_ratio
         << "\n";
  output << "coobs_aware_acceptance_require_pair_completion: "
         << (result.problem_input.solver_options
                     .coobs_aware_acceptance_require_pair_completion
                 ? 1
                 : 0)
         << "\n";
  output << "incremental_pair_diversity_rescue_enabled: "
         << (result.problem_input.solver_options
                     .enable_incremental_pair_diversity_rescue
                 ? 1
                 : 0)
         << "\n";
  output << "incremental_pair_diversity_rescue_min_boards: "
         << result.problem_input.solver_options
                .incremental_pair_diversity_rescue_min_boards
         << "\n";
  output << "success: "
         << (result.pair_board_trial_selection_summary.success ? 1 : 0) << "\n";
  output << "failure_reason: "
         << result.pair_board_trial_selection_summary.failure_reason << "\n";
  output << "seed_count: "
         << result.pair_board_trial_selection_summary.seed_count << "\n";
  output << "candidate_count: "
         << result.pair_board_trial_selection_summary.candidate_count << "\n";
  output << "budget_mode: "
         << ToString(result.pair_board_trial_selection_summary.budget_mode)
         << "\n";
  output << "valid_candidate_count: "
         << result.pair_board_trial_selection_summary.valid_candidate_count
         << "\n";
  output << "valid_candidate_traversed_count: "
         << result.pair_board_trial_selection_summary
                .valid_candidate_traversed_count
         << "\n";
  output << "safety_ceiling_hit: "
         << (result.pair_board_trial_selection_summary.safety_ceiling_hit ? 1
                                                                           : 0)
         << "\n";
  output << "runtime_safety_ceiling: "
         << result.pair_board_trial_selection_summary.runtime_safety_ceiling
         << "\n";
  output << "max_candidate_additions_effective: "
         << result.pair_board_trial_selection_summary
                .max_candidate_additions_effective
         << "\n";
  output << "attempted_count: "
         << result.pair_board_trial_selection_summary.attempted_count << "\n";
  output << "accepted_count: "
         << result.pair_board_trial_selection_summary.accepted_count << "\n";
  output << "rejected_count: "
         << result.pair_board_trial_selection_summary.rejected_count << "\n";
  output << "pairboard_selection_mode: "
         << ToString(result.pair_board_trial_selection_summary
                         .pairboard_selection_mode)
         << "\n";
  output << "batch_acceptance_attempted_count: "
         << result.pair_board_trial_selection_summary
                .batch_acceptance_attempted_count
         << "\n";
  output << "batch_acceptance_accepted_count: "
         << result.pair_board_trial_selection_summary
                .batch_acceptance_accepted_count
         << "\n";
  output << "batch_acceptance_rescued_from_legacy_rmse_gate_count: "
         << result.pair_board_trial_selection_summary
                .batch_acceptance_rescued_from_legacy_rmse_gate_count
         << "\n";
  output << "batch_acceptance_rejected_hard_validity_count: "
         << result.pair_board_trial_selection_summary
                .batch_acceptance_rejected_hard_validity_count
         << "\n";
  output << "batch_acceptance_rejected_catastrophic_residual_count: "
         << result.pair_board_trial_selection_summary
                .batch_acceptance_rejected_catastrophic_residual_count
         << "\n";
  output << "batch_acceptance_rejected_score_count: "
         << result.pair_board_trial_selection_summary
                .batch_acceptance_rejected_score_count
         << "\n";
  output << "persistent_incremental_residual_health_guard_rejected_count: "
         << result.pair_board_trial_selection_summary
                .persistent_incremental_residual_health_guard_rejected_count
         << "\n";
  const StereoExtrinsicSolverOptions& options =
      result.problem_input.solver_options;
  output << "pair_cohesion_enabled: "
         << (options.enable_pair_cohesion ? 1 : 0) << "\n";
  output << "pair_cohesion_min_boards_per_pair: "
         << options.pair_cohesion_min_boards_per_pair << "\n";
  output << "pair_cohesion_max_companions_per_pair: "
         << options.pair_cohesion_max_companions_per_pair << "\n";
  output << "pair_cohesion_auto_target_board_count: "
         << result.pair_board_trial_selection_summary
                .pair_cohesion_auto_target_board_count
         << "\n";
  output << "pair_cohesion_relax_score_gate: "
         << (options.pair_cohesion_relax_score_gate ? 1 : 0) << "\n";
  output << "pair_cohesion_relax_cap_gates: "
         << (options.pair_cohesion_relax_cap_gates ? 1 : 0) << "\n";
  output << "single_board_pair_policy: "
         << ToString(options.single_board_pair_policy) << "\n";
  output << "pair_cohesion_candidate_count: "
         << result.pair_board_trial_selection_summary
                .pair_cohesion_candidate_count
         << "\n";
  output << "pair_cohesion_attempted_count: "
         << result.pair_board_trial_selection_summary
                .pair_cohesion_attempted_count
         << "\n";
  output << "pair_cohesion_accepted_count: "
         << result.pair_board_trial_selection_summary
                .pair_cohesion_accepted_count
         << "\n";
  output << "pair_cohesion_rejected_count: "
         << result.pair_board_trial_selection_summary
                .pair_cohesion_rejected_count
         << "\n";
  output << "pair_cohesion_under_target_pair_count_before_rescue: "
         << result.pair_board_trial_selection_summary
                .pair_cohesion_under_target_pair_count_before_rescue
         << "\n";
  output << "pair_cohesion_under_target_pair_count_after_rescue: "
         << result.pair_board_trial_selection_summary
                .pair_cohesion_under_target_pair_count_after_rescue
         << "\n";
  output << "single_board_pair_count_before_rescue: "
         << result.pair_board_trial_selection_summary
                .single_board_pair_count_before_rescue
         << "\n";
  output << "single_board_pair_count_after_rescue: "
         << result.pair_board_trial_selection_summary
                .single_board_pair_count_after_rescue
         << "\n";
  output << "single_board_pair_count_after_policy: "
         << result.pair_board_trial_selection_summary
                .single_board_pair_count_after_policy
         << "\n";
  output << "dropped_single_board_pair_count: "
         << result.pair_board_trial_selection_summary
                .dropped_single_board_pair_count
         << "\n";
  output << "final_selected_pair_board_count: "
         << result.pair_board_trial_selection_summary
                .final_selected_pair_board_count
         << "\n";
  output << "initial_seed_rmse: "
         << result.pair_board_trial_selection_summary.initial_seed_rmse << "\n";
  output << "final_selected_rmse: "
         << result.pair_board_trial_selection_summary.final_selected_rmse << "\n";
  output << "min_candidate_score: "
         << options.pair_board_selection_min_candidate_score << "\n";
  output << "min_coverage_gain: "
         << options.pair_board_selection_min_coverage_gain << "\n";
  output << "max_accepted_per_pair: "
         << options.pair_board_selection_max_accepted_per_pair << "\n";
  output << "max_accepted_per_board: "
         << options.pair_board_selection_max_accepted_per_board << "\n";
  for (const std::string& warning :
       result.pair_board_trial_selection_summary.warnings) {
    output << "warning: " << warning << "\n";
  }
}

void WriteStereoPairBoardTrialSelectionDecisionsCsv(
    const std::string& path,
    const StereoExtrinsicCalibrationResult& result) {
  std::ofstream output(path.c_str());
  output << "pair_index,board_id,shared_board,pair_cohesion_candidate,"
         << "pair_cohesion_cap_gate_relaxed,cam0_outer_point_count,"
         << "cam1_outer_point_count,shared_point_count,cam0_outer_rmse,"
         << "cam1_outer_rmse,candidate_score,coverage_gain,"
         << "selected_pair_board_count_before,selected_pair_count_before,"
         << "selected_board_count_before,seed,attempted,accepted,"
         << "batchAccepted,"
         << "initial_total_rmse,trial_total_rmse,total_rmse_delta,"
         << "cam0_rmse_delta,cam1_rmse_delta,baseline_rotation_delta_deg,"
         << "baseline_translation_delta_m,pairboard_selection_mode,"
         << "hard_validity_pass,legacy_rmse_pass,catastrophic_residual,"
         << "score_term,coverage_term,pair_completion_bonus,new_board_bonus,"
         << "cap_penalty,information_gain_proxy,residual_overage_penalty,"
         << "batch_acceptance_score,accepted_by_batch_acceptance,"
         << "incremental_estimator_enabled,"
         << "persistent_incremental_estimator_used,candidate_batch_type,"
         << "accept_reason,solution_valid,optimization_success,"
         << "num_iterations,objective_before,objective_after,"
         << "marginal_information_gain_proxy,rank_before,"
         << "normalized_information_gain,"
         << "information_gain_normalization_count,rank_after,"
         << "rank_psi_after,rank_psi_deficiency_after,"
         << "rank_theta_deficiency_after,rank_proxy_increases,"
         << "svd_tolerance,qr_tolerance,elapsed_time_seconds,"
         << "info_gain_threshold,"
         << "committed_or_rollback,trial_coobs_stereo_factor_count,"
         << "trial_coobs_layout_factor_count,"
         << "trial_coobs_stereo_initial_rot_mean_deg,"
         << "trial_coobs_stereo_initial_rot_max_deg,"
         << "trial_coobs_stereo_initial_trans_mean_m,"
         << "trial_coobs_stereo_initial_trans_max_m,"
         << "trial_coobs_layout_initial_rot_mean_deg,"
         << "trial_coobs_layout_initial_rot_max_deg,"
         << "trial_coobs_layout_initial_trans_mean_m,"
         << "trial_coobs_layout_initial_trans_max_m,"
         << "coobs_acceptance_score,coobs_acceptance_health_pass,"
         << "coobs_acceptance_structure_pass,"
         << "coobs_acceptance_balance_pass,"
         << "coobs_acceptance_camera_delta_imbalance,"
         << "coobs_acceptance_camera_delta_ratio,"
         << "accepted_by_coobs_aware_acceptance,"
         << "force,"
         << "reject_reason\n";
  for (const StereoPairBoardTrialSelectionDecision& row :
       result.pair_board_trial_selection_summary.decisions) {
    output << row.pair_index << "," << row.board_id << ","
           << (row.shared_board ? 1 : 0) << ","
           << (row.pair_cohesion_candidate ? 1 : 0) << ","
           << (row.pair_cohesion_cap_gate_relaxed ? 1 : 0) << ","
           << row.cam0_outer_point_count << "," << row.cam1_outer_point_count
           << "," << row.shared_point_count << "," << row.cam0_outer_rmse
           << "," << row.cam1_outer_rmse << "," << row.candidate_score << ","
           << row.coverage_gain << ","
           << row.selected_pair_board_count_before << ","
           << row.selected_pair_count_before << ","
           << row.selected_board_count_before << ","
           << (row.seed ? 1 : 0) << "," << (row.attempted ? 1 : 0) << ","
           << (row.accepted ? 1 : 0) << "," << (row.batchAccepted ? 1 : 0)
           << "," << row.initial_total_rmse << ","
           << row.trial_total_rmse << "," << row.total_rmse_delta << ","
           << row.cam0_rmse_delta << "," << row.cam1_rmse_delta << ","
           << row.baseline_rotation_delta_deg << ","
           << row.baseline_translation_delta_m << ","
           << ToString(row.pairboard_selection_mode) << ","
           << (row.hard_validity_pass ? 1 : 0) << ","
           << (row.legacy_rmse_pass ? 1 : 0) << ","
           << (row.catastrophic_residual ? 1 : 0) << ","
           << row.score_term << "," << row.coverage_term << ","
           << row.pair_completion_bonus << "," << row.new_board_bonus << ","
           << row.cap_penalty << "," << row.information_gain_proxy << ","
           << row.residual_overage_penalty << ","
           << row.batch_acceptance_score << ","
           << (row.accepted_by_batch_acceptance ? 1 : 0) << ","
           << (row.incremental_estimator_enabled ? 1 : 0) << ","
           << (row.persistent_incremental_estimator_used ? 1 : 0) << ","
           << row.candidate_batch_type << "," << row.accept_reason << ","
           << (row.solution_valid ? 1 : 0) << ","
           << (row.optimization_success ? 1 : 0) << ","
           << row.num_iterations << "," << row.objective_before << ","
           << row.objective_after << ","
           << row.marginal_information_gain_proxy << ","
           << row.rank_before << "," << row.normalized_information_gain << ","
           << row.information_gain_normalization_count << ","
           << row.rank_after << "," << row.rank_psi_after << ","
           << row.rank_psi_deficiency_after << ","
           << row.rank_theta_deficiency_after << ","
           << (row.rank_proxy_increases ? 1 : 0) << ","
           << row.svd_tolerance << "," << row.qr_tolerance << ","
           << row.elapsed_time_seconds << ","
           << row.info_gain_threshold << "," << row.committed_or_rollback
           << "," << row.trial_coobs_stereo_factor_count << ","
           << row.trial_coobs_layout_factor_count << ","
           << row.trial_coobs_stereo_initial_rot_mean_deg << ","
           << row.trial_coobs_stereo_initial_rot_max_deg << ","
           << row.trial_coobs_stereo_initial_trans_mean_m << ","
           << row.trial_coobs_stereo_initial_trans_max_m << ","
           << row.trial_coobs_layout_initial_rot_mean_deg << ","
           << row.trial_coobs_layout_initial_rot_max_deg << ","
           << row.trial_coobs_layout_initial_trans_mean_m << ","
           << row.trial_coobs_layout_initial_trans_max_m << ","
           << row.coobs_acceptance_score << ","
           << (row.coobs_acceptance_health_pass ? 1 : 0) << ","
           << (row.coobs_acceptance_structure_pass ? 1 : 0) << ","
           << (row.coobs_acceptance_balance_pass ? 1 : 0) << ","
           << row.coobs_acceptance_camera_delta_imbalance << ","
           << row.coobs_acceptance_camera_delta_ratio << ","
           << (row.accepted_by_coobs_aware_acceptance ? 1 : 0) << ","
           << (row.force ? 1 : 0) << ","
           << row.reject_reason
           << "\n";
  }
}

void WriteStereoPairBoardTrialSelectedBoardsCsv(
    const std::string& path,
    const StereoExtrinsicCalibrationResult& result) {
  std::ofstream output(path.c_str());
  output << "pair_index,board_id\n";
  for (const std::pair<int, int>& key :
       result.pair_board_trial_selection_summary.selected_pair_board_keys) {
    output << key.first << "," << key.second << "\n";
  }
}

void WriteStereoRobustLossSummary(const std::string& path,
                                  const StereoExtrinsicCalibrationResult& result) {
  std::ofstream output(path.c_str());
  output << "robust_loss_type: Huber\n";
  output << "enabled: "
         << (result.problem_input.solver_options.pair_init_use_huber_loss ? 1 : 0)
         << "\n";
  output << "stereo_camera_reprojection_error_huber_delta_formula: sqrt(weight) * 2.0\n";
  output << "pair_only_stereo_ba_init_enabled: "
         << (result.problem_input.solver_options.enable_pair_only_stereo_ba_init ? 1 : 0)
         << "\n";
  output << "pair_only_stereo_ba_init_robust_loss_enabled: "
         << (result.pair_init_summary.robust_loss_enabled ? 1 : 0) << "\n";
  output << "kalibr_style_pair_selection_enabled: "
         << (result.problem_input.solver_options.enable_kalibr_style_pair_selection ? 1 : 0)
         << "\n";
  output << "note: v2 keeps existing Huber loss and does not yet introduce Blake-Zisserman or Cauchy.\n";
}

void WriteStereoExtrinsicUncertaintySummary(
    const std::string& path,
    const StereoExtrinsicCalibrationResult& result) {
  std::ofstream output(path.c_str());
  output << "enabled: "
         << (result.extrinsic_uncertainty_summary.enabled ? 1 : 0) << "\n";
  output << "success: "
         << (result.extrinsic_uncertainty_summary.success ? 1 : 0) << "\n";
  output << "failure_reason: "
         << result.extrinsic_uncertainty_summary.failure_reason << "\n";
  output << "candidate_count: "
         << result.extrinsic_uncertainty_summary.candidate_count << "\n";
  output << "accepted_candidate_count: "
         << result.extrinsic_uncertainty_summary.accepted_candidate_count << "\n";
  output << "rotation_delta_mean_deg: "
         << result.extrinsic_uncertainty_summary.rotation_delta_mean_deg << "\n";
  output << "rotation_delta_median_deg: "
         << result.extrinsic_uncertainty_summary.rotation_delta_median_deg << "\n";
  output << "translation_delta_mean_m: "
         << result.extrinsic_uncertainty_summary.translation_delta_mean_m << "\n";
  output << "translation_delta_median_m: "
         << result.extrinsic_uncertainty_summary.translation_delta_median_m << "\n";
  output << "baseline_length_mean: "
         << result.extrinsic_uncertainty_summary.baseline_length_mean << "\n";
  output << "baseline_length_std: "
         << result.extrinsic_uncertainty_summary.baseline_length_std << "\n";
  output << "jackknife_rotation_max_deg: "
         << result.extrinsic_uncertainty_summary.jackknife_rotation_max_deg << "\n";
  output << "jackknife_translation_max_m: "
         << result.extrinsic_uncertainty_summary.jackknife_translation_max_m << "\n";
  output << "worst_jackknife_pair_index: "
         << result.extrinsic_uncertainty_summary.worst_jackknife_pair_index << "\n";
}

void WriteStereoExtrinsicCandidateDispersionCsv(
    const std::string& path,
    const StereoExtrinsicCalibrationResult& result) {
  std::ofstream output(path.c_str());
  output << "pair_index,board_id,consistency_accepted,rotation_delta_deg,"
         << "translation_delta_m,baseline_length\n";
  for (const StereoExtrinsicCandidateDispersionRow& row :
       result.extrinsic_uncertainty_summary.candidate_rows) {
    output << row.pair_index << "," << row.board_id << ","
           << (row.consistency_accepted ? 1 : 0) << ","
           << row.rotation_delta_deg << "," << row.translation_delta_m << ","
           << row.baseline_length << "\n";
  }
}

void WriteStereoExtrinsicJackknifeCsv(
    const std::string& path,
    const StereoExtrinsicCalibrationResult& result) {
  std::ofstream output(path.c_str());
  output << "excluded_pair_index,remaining_candidate_count,rotation_delta_deg,"
         << "translation_delta_m,baseline_length\n";
  for (const StereoExtrinsicJackknifeRow& row :
       result.extrinsic_uncertainty_summary.jackknife_rows) {
    output << row.excluded_pair_index << "," << row.remaining_candidate_count
           << "," << row.rotation_delta_deg << ","
           << row.translation_delta_m << "," << row.baseline_length << "\n";
  }
}

void WriteStereoPairBoardConsistencySummary(
    const std::string& path,
    const StereoExtrinsicCalibrationResult& result) {
  std::ofstream output(path.c_str());
  const StereoPairBoardConsistencySummary& summary =
      result.pair_board_consistency_summary;
  output << "enabled: " << (summary.enabled ? 1 : 0) << "\n";
  output << "gate_enabled: " << (summary.gate_enabled ? 1 : 0) << "\n";
  output << "local_good_max_outer_rmse_px: "
         << summary.local_good_max_outer_rmse_px << "\n";
  output << "global_bad_min_outer_rmse_px: "
         << summary.global_bad_min_outer_rmse_px << "\n";
  output << "row_count: " << summary.row_count << "\n";
  output << "training_row_count: " << summary.training_row_count << "\n";
  output << "holdout_row_count: " << summary.holdout_row_count << "\n";
  output << "local_good_global_bad_count: "
         << summary.local_good_global_bad_count << "\n";
  output << "gate_rejected_pair_board_count: "
         << summary.gate_rejected_pair_board_count << "\n";
  for (const std::string& warning : summary.warnings) {
    output << "warning: " << warning << "\n";
  }
}

void WriteStereoPairBoardLocalGlobalGapSummary(
    const std::string& path,
    const StereoExtrinsicCalibrationResult& result) {
  struct GapStats {
    std::vector<double> global_outer;
    std::vector<double> local_outer;
    std::vector<double> approx_stereo_local_outer;
    std::vector<double> stereo_local_pose_delta_rotation_deg;
    std::vector<double> stereo_local_pose_delta_translation_m;
    std::vector<double> cam1_from_cam0_rmse;
    std::vector<double> cam0_from_cam1_rmse;
    std::vector<double> stereo_from_cam0_rmse;
    std::vector<double> stereo_from_cam1_rmse;
    std::vector<double> cam1_from_cam0_inverse_direction_rmse;
    std::vector<double> cam0_from_cam1_inverse_direction_rmse;
    int local_good_global_bad_count = 0;
  };

  auto append_stat = [](std::vector<double>* values, double value) {
    if (values != nullptr && std::isfinite(value)) {
      values->push_back(value);
    }
  };
  auto mean_of = [](const std::vector<double>& values) {
    if (values.empty()) {
      return std::numeric_limits<double>::quiet_NaN();
    }
    return std::accumulate(values.begin(), values.end(), 0.0) /
           static_cast<double>(values.size());
  };
  auto median_of = [](std::vector<double> values) {
    if (values.empty()) {
      return std::numeric_limits<double>::quiet_NaN();
    }
    std::sort(values.begin(), values.end());
    const std::size_t mid = values.size() / 2;
    if ((values.size() % 2) == 0U) {
      return 0.5 * (values[mid - 1] + values[mid]);
    }
    return values[mid];
  };
  auto max_of = [](const std::vector<double>& values) {
    if (values.empty()) {
      return std::numeric_limits<double>::quiet_NaN();
    }
    return *std::max_element(values.begin(), values.end());
  };

  GapStats training_stats;
  GapStats holdout_stats;
  double holdout_weighted_local_squared_sum = 0.0;
  int holdout_weighted_local_point_count = 0;
  for (const StereoPairBoardConsistencyRow& row :
       result.pair_board_consistency_summary.rows) {
    GapStats* stats = row.split == "holdout" ? &holdout_stats : &training_stats;
    append_stat(&stats->global_outer, row.global_outer_rmse);
    append_stat(&stats->local_outer, row.local_outer_rmse);
    append_stat(&stats->stereo_local_pose_delta_rotation_deg,
                row.stereo_local_pose_delta_rotation_deg);
    append_stat(&stats->stereo_local_pose_delta_translation_m,
                row.stereo_local_pose_delta_translation_m);
    append_stat(&stats->cam1_from_cam0_rmse,
                row.cam1_outer_rmse_from_cam0_pose);
    append_stat(&stats->cam0_from_cam1_rmse,
                row.cam0_outer_rmse_from_cam1_pose);
    append_stat(&stats->stereo_from_cam0_rmse,
                row.stereo_outer_rmse_from_cam0_pose);
    append_stat(&stats->stereo_from_cam1_rmse,
                row.stereo_outer_rmse_from_cam1_pose);
    append_stat(&stats->cam1_from_cam0_inverse_direction_rmse,
                row.cam1_outer_rmse_from_cam0_pose_inverse_extrinsic);
    append_stat(&stats->cam0_from_cam1_inverse_direction_rmse,
                row.cam0_outer_rmse_from_cam1_pose_inverse_extrinsic);
    if (row.local_good_global_bad) {
      ++stats->local_good_global_bad_count;
    }
    double approx_stereo_local_outer_rmse =
        std::numeric_limits<double>::quiet_NaN();
    double squared_sum = 0.0;
    int point_count = 0;
    if (row.cam0_outer_point_count > 0 &&
        std::isfinite(row.cam0_local_outer_rmse)) {
      squared_sum += static_cast<double>(row.cam0_outer_point_count) *
                     row.cam0_local_outer_rmse * row.cam0_local_outer_rmse;
      point_count += row.cam0_outer_point_count;
    }
    if (row.cam1_outer_point_count > 0 &&
        std::isfinite(row.cam1_local_outer_rmse)) {
      squared_sum += static_cast<double>(row.cam1_outer_point_count) *
                     row.cam1_local_outer_rmse * row.cam1_local_outer_rmse;
      point_count += row.cam1_outer_point_count;
    }
    if (point_count > 0) {
      approx_stereo_local_outer_rmse =
          std::sqrt(squared_sum / static_cast<double>(point_count));
      append_stat(&stats->approx_stereo_local_outer,
                  approx_stereo_local_outer_rmse);
      if (row.split == "holdout") {
        holdout_weighted_local_squared_sum += squared_sum;
        holdout_weighted_local_point_count += point_count;
      }
    }
  }

  const double holdout_approx_aggregate_local_outer_rmse =
      holdout_weighted_local_point_count > 0
          ? std::sqrt(holdout_weighted_local_squared_sum /
                      static_cast<double>(holdout_weighted_local_point_count))
          : std::numeric_limits<double>::quiet_NaN();

  std::ofstream output(path.c_str());
  const auto write_stats = [&output, &mean_of, &median_of, &max_of](
                               const std::string& prefix,
                               const GapStats& stats) {
    output << prefix << "_global_outer_mean_px: "
           << mean_of(stats.global_outer) << "\n";
    output << prefix << "_global_outer_median_px: "
           << median_of(stats.global_outer) << "\n";
    output << prefix << "_global_outer_max_px: "
           << max_of(stats.global_outer) << "\n";
    output << prefix << "_local_outer_mean_px: "
           << mean_of(stats.local_outer) << "\n";
    output << prefix << "_local_outer_median_px: "
           << median_of(stats.local_outer) << "\n";
    output << prefix << "_local_outer_max_px: "
           << max_of(stats.local_outer) << "\n";
    output << prefix << "_approx_stereo_local_outer_mean_px: "
           << mean_of(stats.approx_stereo_local_outer) << "\n";
    output << prefix << "_approx_stereo_local_outer_median_px: "
           << median_of(stats.approx_stereo_local_outer) << "\n";
    output << prefix << "_approx_stereo_local_outer_max_px: "
           << max_of(stats.approx_stereo_local_outer) << "\n";
    output << prefix << "_local_good_global_bad_count: "
           << stats.local_good_global_bad_count << "\n";
    output << prefix << "_stereo_local_pose_delta_rotation_mean_deg: "
           << mean_of(stats.stereo_local_pose_delta_rotation_deg) << "\n";
    output << prefix << "_stereo_local_pose_delta_rotation_median_deg: "
           << median_of(stats.stereo_local_pose_delta_rotation_deg) << "\n";
    output << prefix << "_stereo_local_pose_delta_translation_mean_m: "
           << mean_of(stats.stereo_local_pose_delta_translation_m) << "\n";
    output << prefix << "_stereo_local_pose_delta_translation_median_m: "
           << median_of(stats.stereo_local_pose_delta_translation_m) << "\n";
    output << prefix << "_cam1_outer_rmse_from_cam0_pose_mean_px: "
           << mean_of(stats.cam1_from_cam0_rmse) << "\n";
    output << prefix << "_cam1_outer_rmse_from_cam0_pose_median_px: "
           << median_of(stats.cam1_from_cam0_rmse) << "\n";
    output << prefix << "_cam0_outer_rmse_from_cam1_pose_mean_px: "
           << mean_of(stats.cam0_from_cam1_rmse) << "\n";
    output << prefix << "_cam0_outer_rmse_from_cam1_pose_median_px: "
           << median_of(stats.cam0_from_cam1_rmse) << "\n";
    output << prefix << "_stereo_outer_rmse_from_cam0_pose_mean_px: "
           << mean_of(stats.stereo_from_cam0_rmse) << "\n";
    output << prefix << "_stereo_outer_rmse_from_cam0_pose_median_px: "
           << median_of(stats.stereo_from_cam0_rmse) << "\n";
    output << prefix << "_stereo_outer_rmse_from_cam1_pose_mean_px: "
           << mean_of(stats.stereo_from_cam1_rmse) << "\n";
    output << prefix << "_stereo_outer_rmse_from_cam1_pose_median_px: "
           << median_of(stats.stereo_from_cam1_rmse) << "\n";
    output << prefix
           << "_cam1_outer_rmse_from_cam0_pose_inverse_direction_mean_px: "
           << mean_of(stats.cam1_from_cam0_inverse_direction_rmse) << "\n";
    output << prefix
           << "_cam1_outer_rmse_from_cam0_pose_inverse_direction_median_px: "
           << median_of(stats.cam1_from_cam0_inverse_direction_rmse) << "\n";
    output << prefix
           << "_cam0_outer_rmse_from_cam1_pose_inverse_direction_mean_px: "
           << mean_of(stats.cam0_from_cam1_inverse_direction_rmse) << "\n";
    output << prefix
           << "_cam0_outer_rmse_from_cam1_pose_inverse_direction_median_px: "
           << median_of(stats.cam0_from_cam1_inverse_direction_rmse) << "\n";
  };
  write_stats("training", training_stats);
  write_stats("holdout", holdout_stats);
  output << "holdout_approx_aggregate_local_outer_rmse_px: "
         << holdout_approx_aggregate_local_outer_rmse << "\n";
}

void WriteStereoPairBoardConsistencyCsv(
    const std::string& path,
    const StereoExtrinsicCalibrationResult& result) {
  std::ofstream output(path.c_str());
  output << "split,pair_index,left_frame_label,right_frame_label,is_training,"
         << "board_id,shared_board,cam0_only_board,cam1_only_board,"
         << "cam0_outer_point_count,cam1_outer_point_count,"
         << "global_outer_point_count,global_outer_rmse,"
         << "cam0_local_success,cam1_local_success,"
         << "cam0_local_outer_rmse,cam1_local_outer_rmse,local_outer_rmse,"
         << "cam0_pose_delta_rotation_deg,cam0_pose_delta_translation_m,"
         << "cam1_pose_delta_rotation_deg,cam1_pose_delta_translation_m,"
         << "stereo_local_pose_delta_rotation_deg,"
         << "stereo_local_pose_delta_translation_m,"
         << "cam1_outer_rmse_from_cam0_pose,"
         << "cam0_outer_rmse_from_cam1_pose,"
         << "stereo_outer_rmse_from_cam0_pose,"
         << "stereo_outer_rmse_from_cam1_pose,"
         << "cam1_outer_rmse_from_cam0_pose_inverse_extrinsic,"
         << "cam0_outer_rmse_from_cam1_pose_inverse_extrinsic,"
         << "local_good_global_bad,rejected_by_consistency_gate,"
         << "diagnosis_label\n";
  for (const StereoPairBoardConsistencyRow& row :
       result.pair_board_consistency_summary.rows) {
    output << row.split << "," << row.pair_index << ","
           << row.left_frame_label << "," << row.right_frame_label << ","
           << (row.is_training ? 1 : 0) << "," << row.board_id << ","
           << (row.shared_board ? 1 : 0) << ","
           << (row.cam0_only_board ? 1 : 0) << ","
           << (row.cam1_only_board ? 1 : 0) << ","
           << row.cam0_outer_point_count << ","
           << row.cam1_outer_point_count << ","
           << row.global_outer_point_count << ","
           << row.global_outer_rmse << ","
           << (row.cam0_local_success ? 1 : 0) << ","
           << (row.cam1_local_success ? 1 : 0) << ","
           << row.cam0_local_outer_rmse << ","
           << row.cam1_local_outer_rmse << ","
           << row.local_outer_rmse << ","
           << row.cam0_pose_delta_rotation_deg << ","
           << row.cam0_pose_delta_translation_m << ","
           << row.cam1_pose_delta_rotation_deg << ","
           << row.cam1_pose_delta_translation_m << ","
           << row.stereo_local_pose_delta_rotation_deg << ","
           << row.stereo_local_pose_delta_translation_m << ","
           << row.cam1_outer_rmse_from_cam0_pose << ","
           << row.cam0_outer_rmse_from_cam1_pose << ","
           << row.stereo_outer_rmse_from_cam0_pose << ","
           << row.stereo_outer_rmse_from_cam1_pose << ","
           << row.cam1_outer_rmse_from_cam0_pose_inverse_extrinsic << ","
           << row.cam0_outer_rmse_from_cam1_pose_inverse_extrinsic << ","
           << (row.local_good_global_bad ? 1 : 0) << ","
           << (row.rejected_by_consistency_gate ? 1 : 0) << ","
           << row.diagnosis_label << "\n";
  }
}

void WriteStereoSharedBoardQualityAuditCsv(
    const std::string& path,
    const StereoExtrinsicCalibrationResult& result) {
  std::ofstream output(path.c_str());
  output << "split,pair_index,left_frame_label,right_frame_label,is_training,"
         << "board_id,shared_board,cam0_outer_point_count,"
         << "cam1_outer_point_count,cam0_have_pose,cam1_have_pose,"
         << "cam0_outer_rmse,cam1_outer_rmse,audit_pass,"
         << "would_filter_if_hard_gate,reason\n";

  const StereoMeasurementDataset& dataset =
      result.problem_input.measurement_dataset;
  const StereoExtrinsicSolverOptions& options =
      result.problem_input.solver_options;

  const auto write_rows_for_split =
      [&](const std::vector<int>& pair_indices,
          const std::string& split,
          bool is_training) {
        for (int pair_index : pair_indices) {
          const StereoFramePair* pair = FindPair(dataset, pair_index);
          for (int board_id : SortedBoardIds(dataset, pair_index)) {
            const bool shared_board =
                ContainsBoard(dataset.pair_shared_board_ids, pair_index,
                              board_id);
            StereoSharedBoardQuality quality;
            std::string reason = "not_shared_board";
            if (shared_board) {
              quality = EvaluateSharedBoardQuality(
                  dataset, result.optimized_scene, options, pair_index,
                  board_id);
              if (quality.pass) {
                reason = "pass";
              } else if (!quality.have_cam0 && !quality.have_cam1) {
                reason = "cam0_cam1_pose_unavailable_or_too_few_points";
              } else if (!quality.have_cam0) {
                reason = "cam0_pose_unavailable_or_too_few_points";
              } else if (!quality.have_cam1) {
                reason = "cam1_pose_unavailable_or_too_few_points";
              } else {
                reason = "outer_rmse_above_audit_threshold";
              }
            }
            output << split << "," << pair_index << ","
                   << (pair == nullptr ? "" : pair->left_frame_label) << ","
                   << (pair == nullptr ? "" : pair->right_frame_label) << ","
                   << (is_training ? 1 : 0) << "," << board_id << ","
                   << (shared_board ? 1 : 0) << ","
                   << quality.cam0_outer_point_count << ","
                   << quality.cam1_outer_point_count << ","
                   << (quality.have_cam0 ? 1 : 0) << ","
                   << (quality.have_cam1 ? 1 : 0) << ","
                   << quality.cam0_outer_rmse << ","
                   << quality.cam1_outer_rmse << ","
                   << (quality.pass ? 1 : 0) << ","
                   << (shared_board && !quality.pass ? 1 : 0) << ","
                   << reason << "\n";
          }
        }
      };

  write_rows_for_split(dataset.training_pair_indices, "training", true);
  write_rows_for_split(dataset.holdout_pair_indices, "holdout", false);
}

void WriteStereoAngularFixedKCornerTraceCsv(
    const std::string& path,
    const StereoExtrinsicCalibrationResult& result) {
  std::ofstream output(path.c_str());
  output << std::setprecision(12);
  output << "split,pair_index,frame_id,timestamp,board_id,corner_id,cam_id,"
         << "u_obs_x,u_obs_y,u_pred_x,u_pred_y,"
         << "pixel_error_px,angular_error_rad,angular_error_deg,chordal_error,"
         << "polar_angle_rad,polar_angle_deg,edge_distance_px,"
         << "is_outer_corner,is_inner_corner,detection_source,is_rescued,"
         << "q_obs_x,q_obs_y,q_obs_z,q_pred_x,q_pred_y,q_pred_z,"
         << "valid,invalid_reason\n";

  const auto write_nan = [&output]() {
    output << "nan";
  };
  const auto write_value = [&output, &write_nan](double value) {
    if (std::isfinite(value)) {
      output << value;
    } else {
      write_nan();
    }
  };
  const auto write_row_tail =
      [&](const Eigen::Vector2d& observed,
          const Eigen::Vector2d& predicted,
          double pixel_error_px,
          double angular_error_rad,
          double angular_error_deg,
          double chordal_error,
          double polar_angle_rad,
          double polar_angle_deg,
          double edge_distance_px,
          bool is_outer,
          bool is_inner,
          const Eigen::Vector3d& q_obs,
          const Eigen::Vector3d& q_pred,
          bool valid,
          const std::string& invalid_reason) {
        write_value(observed.x());
        output << ",";
        write_value(observed.y());
        output << ",";
        write_value(predicted.x());
        output << ",";
        write_value(predicted.y());
        output << ",";
        write_value(pixel_error_px);
        output << ",";
        write_value(angular_error_rad);
        output << ",";
        write_value(angular_error_deg);
        output << ",";
        write_value(chordal_error);
        output << ",";
        write_value(polar_angle_rad);
        output << ",";
        write_value(polar_angle_deg);
        output << ",";
        write_value(edge_distance_px);
        output << "," << (is_outer ? 1 : 0) << ","
               << (is_inner ? 1 : 0) << ",unknown,0,";
        write_value(q_obs.x());
        output << ",";
        write_value(q_obs.y());
        output << ",";
        write_value(q_obs.z());
        output << ",";
        write_value(q_pred.x());
        output << ",";
        write_value(q_pred.y());
        output << ",";
        write_value(q_pred.z());
        output << "," << (valid ? 1 : 0) << "," << invalid_reason << "\n";
      };

  const DoubleSphereCameraModel cam0 =
      DoubleSphereCameraModel::FromConfig(MakeCameraConfig(result.optimized_scene.cam0));
  const DoubleSphereCameraModel cam1 =
      DoubleSphereCameraModel::FromConfig(MakeCameraConfig(result.optimized_scene.cam1));
  const Eigen::Isometry3d T_cam1_cam0 =
      ToIsometry3d(result.optimized_scene.T_cam1_cam0);

  const auto edge_distance = [](const StereoCameraFixedCalibration& camera,
                                const Eigen::Vector2d& pixel) {
    if (camera.resolution.size() != 2 ||
        camera.resolution[0] <= 0 ||
        camera.resolution[1] <= 0 ||
        !pixel.allFinite()) {
      return std::numeric_limits<double>::quiet_NaN();
    }
    const double width = static_cast<double>(camera.resolution[0]);
    const double height = static_cast<double>(camera.resolution[1]);
    return std::min(std::min(pixel.x(), width - 1.0 - pixel.x()),
                    std::min(pixel.y(), height - 1.0 - pixel.y()));
  };
  const auto normalize_or_nan = [](const Eigen::Vector3d& value)
      -> Eigen::Vector3d {
    const double norm = value.norm();
    if (!(norm > 0.0) || !std::isfinite(norm)) {
      return Eigen::Vector3d(std::numeric_limits<double>::quiet_NaN(),
                             std::numeric_limits<double>::quiet_NaN(),
                             std::numeric_limits<double>::quiet_NaN());
    }
    return value / norm;
  };
  const auto angular_error = [](const Eigen::Vector3d& a,
                                const Eigen::Vector3d& b) {
    if (!a.allFinite() || !b.allFinite()) {
      return std::numeric_limits<double>::quiet_NaN();
    }
    const double dot = std::max(-1.0, std::min(1.0, a.dot(b)));
    const double cross_norm = a.cross(b).norm();
    return std::atan2(cross_norm, dot);
  };

  const StereoMeasurementDataset& dataset =
      result.problem_input.measurement_dataset;
  for (const StereoObservation& observation : dataset.observations) {
    if (!observation.used_in_solver) {
      continue;
    }
    const StereoFramePair* pair = FindPair(dataset, observation.pair_index);
    if (pair == nullptr || !pair->is_training) {
      continue;
    }
    if (result.pair_selection_summary.selected_pair_indices.count(
            observation.pair_index) == 0) {
      continue;
    }
    if (!PairBoardSelected(result.pair_selection_summary,
                           observation.pair_index,
                           observation.board_id)) {
      continue;
    }

    output << "training_backend," << observation.pair_index << ","
           << observation.frame_index << "," << observation.frame_label << ","
           << observation.board_id << "," << observation.point_id << ","
           << observation.camera_index << ",";

    const bool is_outer = observation.point_type == JointPointType::Outer;
    const bool is_inner = observation.point_type == JointPointType::Internal;
    const StereoCameraFixedCalibration& camera_config =
        observation.camera_index == 0 ? result.optimized_scene.cam0
                                      : result.optimized_scene.cam1;
    const DoubleSphereCameraModel& camera =
        observation.camera_index == 0 ? cam0 : cam1;
    const double edge_distance_px =
        edge_distance(camera_config, observation.observed_image_xy);

    Eigen::Vector3d q_obs_raw = Eigen::Vector3d::Zero();
    if (!camera.keypointToEuclidean(observation.observed_image_xy, &q_obs_raw)) {
      write_row_tail(
          observation.observed_image_xy,
          Eigen::Vector2d(std::numeric_limits<double>::quiet_NaN(),
                          std::numeric_limits<double>::quiet_NaN()),
          std::numeric_limits<double>::quiet_NaN(),
          std::numeric_limits<double>::quiet_NaN(),
          std::numeric_limits<double>::quiet_NaN(),
          std::numeric_limits<double>::quiet_NaN(),
          std::numeric_limits<double>::quiet_NaN(),
          std::numeric_limits<double>::quiet_NaN(),
          edge_distance_px, is_outer, is_inner,
          Eigen::Vector3d(std::numeric_limits<double>::quiet_NaN(),
                          std::numeric_limits<double>::quiet_NaN(),
                          std::numeric_limits<double>::quiet_NaN()),
          Eigen::Vector3d(std::numeric_limits<double>::quiet_NaN(),
                          std::numeric_limits<double>::quiet_NaN(),
                          std::numeric_limits<double>::quiet_NaN()),
          false, "unproject_failed");
      continue;
    }
    const Eigen::Vector3d q_obs = normalize_or_nan(q_obs_raw);
    if (!q_obs.allFinite()) {
      write_row_tail(
          observation.observed_image_xy,
          Eigen::Vector2d(std::numeric_limits<double>::quiet_NaN(),
                          std::numeric_limits<double>::quiet_NaN()),
          std::numeric_limits<double>::quiet_NaN(),
          std::numeric_limits<double>::quiet_NaN(),
          std::numeric_limits<double>::quiet_NaN(),
          std::numeric_limits<double>::quiet_NaN(),
          std::numeric_limits<double>::quiet_NaN(),
          std::numeric_limits<double>::quiet_NaN(),
          edge_distance_px, is_outer, is_inner, q_obs,
          Eigen::Vector3d(std::numeric_limits<double>::quiet_NaN(),
                          std::numeric_limits<double>::quiet_NaN(),
                          std::numeric_limits<double>::quiet_NaN()),
          false, "invalid_observed_ray");
      continue;
    }

    const auto pair_pose_it =
        result.optimized_scene.T_cam0_world_by_pair.find(observation.pair_index);
    if (pair_pose_it == result.optimized_scene.T_cam0_world_by_pair.end()) {
      write_row_tail(
          observation.observed_image_xy,
          Eigen::Vector2d(std::numeric_limits<double>::quiet_NaN(),
                          std::numeric_limits<double>::quiet_NaN()),
          std::numeric_limits<double>::quiet_NaN(),
          std::numeric_limits<double>::quiet_NaN(),
          std::numeric_limits<double>::quiet_NaN(),
          std::numeric_limits<double>::quiet_NaN(),
          std::numeric_limits<double>::quiet_NaN(),
          std::numeric_limits<double>::quiet_NaN(),
          edge_distance_px, is_outer, is_inner, q_obs,
          Eigen::Vector3d(std::numeric_limits<double>::quiet_NaN(),
                          std::numeric_limits<double>::quiet_NaN(),
                          std::numeric_limits<double>::quiet_NaN()),
          false, "missing_pair_pose");
      continue;
    }
    const auto board_pose_it =
        result.optimized_scene.T_world_board_by_id.find(observation.board_id);
    if (board_pose_it == result.optimized_scene.T_world_board_by_id.end()) {
      write_row_tail(
          observation.observed_image_xy,
          Eigen::Vector2d(std::numeric_limits<double>::quiet_NaN(),
                          std::numeric_limits<double>::quiet_NaN()),
          std::numeric_limits<double>::quiet_NaN(),
          std::numeric_limits<double>::quiet_NaN(),
          std::numeric_limits<double>::quiet_NaN(),
          std::numeric_limits<double>::quiet_NaN(),
          std::numeric_limits<double>::quiet_NaN(),
          std::numeric_limits<double>::quiet_NaN(),
          edge_distance_px, is_outer, is_inner, q_obs,
          Eigen::Vector3d(std::numeric_limits<double>::quiet_NaN(),
                          std::numeric_limits<double>::quiet_NaN(),
                          std::numeric_limits<double>::quiet_NaN()),
          false, "missing_board_pose");
      continue;
    }

    const Eigen::Vector4d point_board(observation.target_point_board.x(),
                                      observation.target_point_board.y(),
                                      observation.target_point_board.z(),
                                      1.0);
    const Eigen::Vector4d point_world = board_pose_it->second * point_board;
    const Eigen::Vector3d point_cam0 =
        (pair_pose_it->second * point_world).head<3>();
    const Eigen::Vector3d point_camera =
        observation.camera_index == 0 ? point_cam0
                                      : T_cam1_cam0 * point_cam0;
    const Eigen::Vector3d q_pred = normalize_or_nan(point_camera);
    Eigen::Vector2d predicted =
        Eigen::Vector2d(std::numeric_limits<double>::quiet_NaN(),
                        std::numeric_limits<double>::quiet_NaN());
    if (!q_pred.allFinite() ||
        !camera.vsEuclideanToKeypoint(point_camera, &predicted)) {
      write_row_tail(
          observation.observed_image_xy, predicted,
          std::numeric_limits<double>::quiet_NaN(),
          std::numeric_limits<double>::quiet_NaN(),
          std::numeric_limits<double>::quiet_NaN(),
          std::numeric_limits<double>::quiet_NaN(),
          std::numeric_limits<double>::quiet_NaN(),
          std::numeric_limits<double>::quiet_NaN(),
          edge_distance_px, is_outer, is_inner, q_obs, q_pred, false,
          "projection_failed");
      continue;
    }

    const double pixel_error_px =
        (predicted - observation.observed_image_xy).norm();
    const double angular_error_rad = angular_error(q_obs, q_pred);
    const double angular_error_deg =
        angular_error_rad * 180.0 / M_PI;
    const double chordal_error = (q_pred - q_obs).norm();
    const double polar_angle_rad =
        std::acos(std::max(-1.0, std::min(1.0, q_obs.z())));
    const double polar_angle_deg = polar_angle_rad * 180.0 / M_PI;
    write_row_tail(observation.observed_image_xy, predicted, pixel_error_px,
                   angular_error_rad, angular_error_deg, chordal_error,
                   polar_angle_rad, polar_angle_deg, edge_distance_px,
                   is_outer, is_inner, q_obs, q_pred, true, "");
  }
}

void WriteStereoAngularFixedKSummary(
    const std::string& path,
    const StereoExtrinsicCalibrationResult& result) {
  std::ofstream output(path.c_str());
  output << std::setprecision(12);
  output << "metric,value\n";
  output << "spherical_uncertainty_mode,"
         << ToString(result.global_sparse_ba_summary.spherical_uncertainty_mode)
         << "\n";
  output << "spherical_covariance_valid_count,"
         << result.global_sparse_ba_summary.spherical_covariance_valid_count
         << "\n";
  output << "spherical_covariance_invalid_count,"
         << result.global_sparse_ba_summary.spherical_covariance_invalid_count
         << "\n";
  output << "spherical_covariance_damped_count,"
         << result.global_sparse_ba_summary.spherical_covariance_damped_count
         << "\n";
  output << "spherical_whitening_clamped_count,"
         << result.global_sparse_ba_summary.spherical_whitening_clamped_count
         << "\n";
  output << "spherical_tangent_sigma_mean_rad,"
         << result.global_sparse_ba_summary.spherical_tangent_sigma_mean_rad
         << "\n";
  output << "spherical_tangent_sigma_min_rad,"
         << result.global_sparse_ba_summary.spherical_tangent_sigma_min_rad
         << "\n";
  output << "spherical_tangent_sigma_max_rad,"
         << result.global_sparse_ba_summary.spherical_tangent_sigma_max_rad
         << "\n";
  output << "spherical_whitening_weight_mean,"
         << result.global_sparse_ba_summary.spherical_whitening_weight_mean
         << "\n";
  output << "spherical_whitening_weight_min,"
         << result.global_sparse_ba_summary.spherical_whitening_weight_min
         << "\n";
  output << "spherical_whitening_weight_max,"
         << result.global_sparse_ba_summary.spherical_whitening_weight_max
         << "\n";

  struct AngularAccumulator {
    int count = 0;
    double squared_rad_sum = 0.0;
    double squared_chordal_sum = 0.0;
    double squared_pixel_sum = 0.0;
  };
  const auto add = [](AngularAccumulator* acc,
                      double angular_rad,
                      double chordal,
                      double pixel_error_px) {
    if (acc == nullptr || !std::isfinite(angular_rad) ||
        !std::isfinite(chordal) || !std::isfinite(pixel_error_px)) {
      return;
    }
    ++acc->count;
    acc->squared_rad_sum += angular_rad * angular_rad;
    acc->squared_chordal_sum += chordal * chordal;
    acc->squared_pixel_sum += pixel_error_px * pixel_error_px;
  };
  const auto rmse_rad = [](const AngularAccumulator& acc) {
    return acc.count > 0
               ? std::sqrt(acc.squared_rad_sum /
                           static_cast<double>(acc.count))
               : std::numeric_limits<double>::quiet_NaN();
  };
  const auto rmse_chordal = [](const AngularAccumulator& acc) {
    return acc.count > 0
               ? std::sqrt(acc.squared_chordal_sum /
                           static_cast<double>(acc.count))
               : std::numeric_limits<double>::quiet_NaN();
  };
  const auto rmse_pixel = [](const AngularAccumulator& acc) {
    return acc.count > 0
               ? std::sqrt(acc.squared_pixel_sum /
                           static_cast<double>(acc.count))
               : std::numeric_limits<double>::quiet_NaN();
  };
  const auto write_metric = [&output](const std::string& key, double value) {
    output << key << ",";
    if (std::isfinite(value)) {
      output << value;
    } else {
      output << "nan";
    }
    output << "\n";
  };
  const auto write_accumulator =
      [&](const std::string& prefix, const AngularAccumulator& acc) {
        output << prefix << "_count," << acc.count << "\n";
        write_metric(prefix + "_angular_rmse_rad", rmse_rad(acc));
        write_metric(prefix + "_angular_rmse_deg",
                     rmse_rad(acc) * 180.0 / M_PI);
        write_metric(prefix + "_chordal_rmse", rmse_chordal(acc));
        write_metric(prefix + "_pixel_rmse_px", rmse_pixel(acc));
      };

  const DoubleSphereCameraModel cam0 =
      DoubleSphereCameraModel::FromConfig(MakeCameraConfig(result.optimized_scene.cam0));
  const DoubleSphereCameraModel cam1 =
      DoubleSphereCameraModel::FromConfig(MakeCameraConfig(result.optimized_scene.cam1));
  const Eigen::Isometry3d T_cam1_cam0 =
      ToIsometry3d(result.optimized_scene.T_cam1_cam0);
  const auto normalize_or_nan = [](const Eigen::Vector3d& value)
      -> Eigen::Vector3d {
    const double norm = value.norm();
    if (!(norm > 0.0) || !std::isfinite(norm)) {
      return Eigen::Vector3d(std::numeric_limits<double>::quiet_NaN(),
                             std::numeric_limits<double>::quiet_NaN(),
                             std::numeric_limits<double>::quiet_NaN());
    }
    return value / norm;
  };
  const auto angular_error = [](const Eigen::Vector3d& a,
                                const Eigen::Vector3d& b) {
    if (!a.allFinite() || !b.allFinite()) {
      return std::numeric_limits<double>::quiet_NaN();
    }
    const double dot = std::max(-1.0, std::min(1.0, a.dot(b)));
    return std::atan2(a.cross(b).norm(), dot);
  };
  AngularAccumulator all;
  std::map<std::string, AngularAccumulator> split_accumulators;
  std::map<std::string, std::map<int, AngularAccumulator>>
      split_camera_accumulators;
  std::map<std::string, std::map<std::string, AngularAccumulator>>
      split_polar_bucket_accumulators;
  std::map<int, AngularAccumulator> camera_accumulators;
  std::map<std::string, AngularAccumulator> polar_bucket_accumulators;
  int invalid_unprojection_count = 0;
  int invalid_projection_count = 0;
  int skipped_not_backend_count = 0;
  int skipped_holdout_pose_count = 0;
  std::map<int, Eigen::Matrix4d> holdout_refit_pose_by_pair;
  std::set<int> holdout_refit_failed_pairs;
  std::map<std::pair<int, int>, Eigen::Matrix4d> holdout_board_refit_pose_by_pair_board;
  std::set<std::pair<int, int>> holdout_board_refit_failed_pair_boards;

  const auto bucket_name_for_polar = [](double polar_deg) {
    std::string bucket = "polar_0_30";
    if (polar_deg >= 70.0) {
      bucket = "polar_70_plus";
    } else if (polar_deg >= 50.0) {
      bucket = "polar_50_70";
    } else if (polar_deg >= 30.0) {
      bucket = "polar_30_50";
    }
    return bucket;
  };

  const StereoMeasurementDataset& dataset =
      result.problem_input.measurement_dataset;
  for (const StereoObservation& observation : dataset.observations) {
    if (!observation.used_in_solver) {
      continue;
    }
    const StereoFramePair* pair = FindPair(dataset, observation.pair_index);
    if (pair == nullptr) {
      ++invalid_projection_count;
      continue;
    }
    if (pair->is_training) {
      if (result.pair_selection_summary.selected_pair_indices.count(
              observation.pair_index) == 0) {
        ++skipped_not_backend_count;
        continue;
      }
      if (!PairBoardSelected(result.pair_selection_summary,
                             observation.pair_index,
                             observation.board_id)) {
        ++skipped_not_backend_count;
        continue;
      }
    }
    const DoubleSphereCameraModel& camera =
        observation.camera_index == 0 ? cam0 : cam1;
    Eigen::Vector3d q_obs_raw = Eigen::Vector3d::Zero();
    if (!camera.keypointToEuclidean(observation.observed_image_xy,
                                    &q_obs_raw)) {
      ++invalid_unprojection_count;
      continue;
    }
    const Eigen::Vector3d q_obs = normalize_or_nan(q_obs_raw);
    if (!q_obs.allFinite()) {
      ++invalid_unprojection_count;
      continue;
    }
    Eigen::Matrix4d T_cam0_world = Eigen::Matrix4d::Identity();
    Eigen::Matrix4d T_cam0_board_override = Eigen::Matrix4d::Identity();
    bool use_board_override = false;
    std::string split_name;
    if (pair->is_training) {
      const auto pair_pose_it =
          result.optimized_scene.T_cam0_world_by_pair.find(
              observation.pair_index);
      if (pair_pose_it == result.optimized_scene.T_cam0_world_by_pair.end()) {
        ++invalid_projection_count;
        continue;
      }
      T_cam0_world = pair_pose_it->second;
      split_name = "training_backend";
    } else {
      if (holdout_refit_failed_pairs.count(observation.pair_index) > 0) {
        ++skipped_holdout_pose_count;
        continue;
      }
      auto refit_it = holdout_refit_pose_by_pair.find(observation.pair_index);
      if (refit_it == holdout_refit_pose_by_pair.end()) {
        RuntimeCounters runtime_counters;
        StereoRefitDiagnostics refit_diagnostics;
        Eigen::Matrix4d refit_pose = Eigen::Matrix4d::Identity();
        if (!RefitPairPoseFromStereoOuterObservations(
                dataset, result.optimized_scene, observation.pair_index,
                result.problem_input.solver_options, &runtime_counters,
                &refit_diagnostics, &refit_pose)) {
          holdout_refit_failed_pairs.insert(observation.pair_index);
          ++skipped_holdout_pose_count;
          continue;
        }
        refit_it =
            holdout_refit_pose_by_pair
                .insert(std::make_pair(observation.pair_index, refit_pose))
                .first;
      }
      T_cam0_world = refit_it->second;
      split_name = "holdout_refit";
    }
    const auto board_pose_it =
        result.optimized_scene.T_world_board_by_id.find(observation.board_id);
    if (board_pose_it == result.optimized_scene.T_world_board_by_id.end()) {
      ++invalid_projection_count;
      continue;
    }
    const Eigen::Vector4d point_board(observation.target_point_board.x(),
                                      observation.target_point_board.y(),
                                      observation.target_point_board.z(),
                                      1.0);
    Eigen::Vector3d point_cam0 = Eigen::Vector3d::Zero();
    if (use_board_override) {
      point_cam0 = (T_cam0_board_override * point_board).head<3>();
    } else {
      const Eigen::Vector4d point_world = board_pose_it->second * point_board;
      point_cam0 = (T_cam0_world * point_world).head<3>();
    }
    const Eigen::Vector3d point_camera =
        observation.camera_index == 0 ? point_cam0
                                      : T_cam1_cam0 * point_cam0;
    const Eigen::Vector3d q_pred = normalize_or_nan(point_camera);
    if (!q_pred.allFinite()) {
      ++invalid_projection_count;
      continue;
    }
    const double angular_rad = angular_error(q_obs, q_pred);
    const double chordal = (q_pred - q_obs).norm();
    Eigen::Vector2d predicted =
        Eigen::Vector2d(std::numeric_limits<double>::quiet_NaN(),
                        std::numeric_limits<double>::quiet_NaN());
    double pixel_error_px = std::numeric_limits<double>::quiet_NaN();
    if (camera.vsEuclideanToKeypoint(point_camera, &predicted)) {
      pixel_error_px = (predicted - observation.observed_image_xy).norm();
    }
    const double polar_deg =
        std::acos(std::max(-1.0, std::min(1.0, q_obs.z()))) *
        180.0 / M_PI;
    const std::string bucket = bucket_name_for_polar(polar_deg);
    add(&all, angular_rad, chordal, pixel_error_px);
    add(&split_accumulators[split_name], angular_rad, chordal, pixel_error_px);
    add(&split_camera_accumulators[split_name][observation.camera_index],
        angular_rad, chordal, pixel_error_px);
    add(&split_polar_bucket_accumulators[split_name][bucket], angular_rad,
        chordal, pixel_error_px);
    add(&camera_accumulators[observation.camera_index], angular_rad, chordal,
        pixel_error_px);
    add(&polar_bucket_accumulators[bucket], angular_rad, chordal,
        pixel_error_px);
  }

  for (const StereoObservation& observation : dataset.observations) {
    if (!observation.used_in_solver) {
      continue;
    }
    const StereoFramePair* pair = FindPair(dataset, observation.pair_index);
    if (pair == nullptr || pair->is_training) {
      continue;
    }
    const DoubleSphereCameraModel& camera =
        observation.camera_index == 0 ? cam0 : cam1;
    Eigen::Vector3d q_obs_raw = Eigen::Vector3d::Zero();
    if (!camera.keypointToEuclidean(observation.observed_image_xy,
                                    &q_obs_raw)) {
      ++invalid_unprojection_count;
      continue;
    }
    const Eigen::Vector3d q_obs = normalize_or_nan(q_obs_raw);
    if (!q_obs.allFinite()) {
      ++invalid_unprojection_count;
      continue;
    }
    const std::pair<int, int> key(observation.pair_index,
                                  observation.board_id);
    if (holdout_board_refit_failed_pair_boards.count(key) > 0) {
      ++skipped_holdout_pose_count;
      continue;
    }
    auto board_refit_it = holdout_board_refit_pose_by_pair_board.find(key);
    if (board_refit_it == holdout_board_refit_pose_by_pair_board.end()) {
      Eigen::Matrix4d refit_pose = Eigen::Matrix4d::Identity();
      if (!RefitStereoBoardPoseForVisualization(
              dataset, result.optimized_scene, observation.pair_index,
              observation.board_id,
              result.problem_input.solver_options.symmetric_refit_max_iterations,
              result.problem_input.solver_options.symmetric_refit_step,
              &refit_pose)) {
        holdout_board_refit_failed_pair_boards.insert(key);
        ++skipped_holdout_pose_count;
        continue;
      }
      board_refit_it =
          holdout_board_refit_pose_by_pair_board.insert(std::make_pair(key, refit_pose))
              .first;
    }
    const Eigen::Vector4d point_board(observation.target_point_board.x(),
                                      observation.target_point_board.y(),
                                      observation.target_point_board.z(),
                                      1.0);
    const Eigen::Vector3d point_cam0 =
        (board_refit_it->second * point_board).head<3>();
    const Eigen::Vector3d point_camera =
        observation.camera_index == 0 ? point_cam0
                                      : T_cam1_cam0 * point_cam0;
    const Eigen::Vector3d q_pred = normalize_or_nan(point_camera);
    if (!q_pred.allFinite()) {
      ++invalid_projection_count;
      continue;
    }
    Eigen::Vector2d predicted =
        Eigen::Vector2d(std::numeric_limits<double>::quiet_NaN(),
                        std::numeric_limits<double>::quiet_NaN());
    if (!camera.vsEuclideanToKeypoint(point_camera, &predicted)) {
      ++invalid_projection_count;
      continue;
    }
    const double pixel_error_px =
        (predicted - observation.observed_image_xy).norm();
    const double angular_rad = angular_error(q_obs, q_pred);
    const double chordal = (q_pred - q_obs).norm();
    const double polar_deg =
        std::acos(std::max(-1.0, std::min(1.0, q_obs.z()))) *
        180.0 / M_PI;
    const std::string split_name = "holdout_extrinsic_only";
    const std::string bucket = bucket_name_for_polar(polar_deg);
    add(&split_accumulators[split_name], angular_rad, chordal, pixel_error_px);
    add(&split_camera_accumulators[split_name][observation.camera_index],
        angular_rad, chordal, pixel_error_px);
    add(&split_polar_bucket_accumulators[split_name][bucket], angular_rad,
        chordal, pixel_error_px);
  }

  output << "residual_mode,"
         << ToString(result.global_sparse_ba_summary.residual_mode) << "\n";
  output << "invalid_unprojection_count," << invalid_unprojection_count << "\n";
  output << "invalid_projection_count," << invalid_projection_count << "\n";
  output << "skipped_not_backend_count," << skipped_not_backend_count << "\n";
  output << "skipped_holdout_pose_count," << skipped_holdout_pose_count << "\n";
  output << "holdout_refit_failed_pair_count,"
         << holdout_refit_failed_pairs.size() << "\n";
  output << "holdout_extrinsic_only_refit_failed_pair_board_count,"
         << holdout_board_refit_failed_pair_boards.size() << "\n";
  output << "holdout_angular_mode,stereo_outer_pair_pose_refit\n";
  output << "holdout_extrinsic_only_angular_mode,local_stereo_board_pose_refit\n";
  write_accumulator("all_backend", all);
  write_accumulator("training_backend",
                    split_accumulators["training_backend"]);
  write_accumulator("holdout_refit",
                    split_accumulators["holdout_refit"]);
  write_accumulator("holdout_extrinsic_only",
                    split_accumulators["holdout_extrinsic_only"]);
  write_accumulator("cam0", camera_accumulators[0]);
  write_accumulator("cam1", camera_accumulators[1]);
  write_accumulator("polar_0_30", polar_bucket_accumulators["polar_0_30"]);
  write_accumulator("polar_30_50", polar_bucket_accumulators["polar_30_50"]);
  write_accumulator("polar_50_70", polar_bucket_accumulators["polar_50_70"]);
  write_accumulator("polar_70_plus",
                    polar_bucket_accumulators["polar_70_plus"]);
  const std::vector<std::string> splits = {
      "training_backend", "holdout_refit", "holdout_extrinsic_only"};
  const std::vector<std::string> buckets = {
      "polar_0_30", "polar_30_50", "polar_50_70", "polar_70_plus"};
  for (const std::string& split : splits) {
    write_accumulator(split + "_cam0", split_camera_accumulators[split][0]);
    write_accumulator(split + "_cam1", split_camera_accumulators[split][1]);
    for (const std::string& bucket : buckets) {
      write_accumulator(split + "_" + bucket,
                        split_polar_bucket_accumulators[split][bucket]);
    }
  }
}

void WriteStereoHoldoutBoardPolarRmseCsv(
    const std::string& path,
    const StereoExtrinsicCalibrationResult& result) {
  std::ofstream output(path.c_str());
  if (!output.is_open()) {
    throw std::runtime_error("Failed to open holdout board polar RMSE CSV: " +
                             path);
  }
  output << std::setprecision(12);
  output << "split,board_id,camera_index,point_type,polar_bucket,"
         << "polar_min_deg,polar_max_deg,point_count,"
         << "pixel_rmse_px,angular_rmse_rad,angular_rmse_deg,chordal_rmse,"
         << "mean_pixel_error_px,mean_angular_error_deg,"
         << "max_pixel_error_px,max_angular_error_deg\n";

  struct BucketInfo {
    std::string name;
    double min_deg = 0.0;
    double max_deg = 0.0;
  };
  const std::vector<BucketInfo> buckets = {
      {"polar_0_30", 0.0, 30.0},
      {"polar_30_50", 30.0, 50.0},
      {"polar_50_70", 50.0, 70.0},
      {"polar_70_plus", 70.0,
       std::numeric_limits<double>::infinity()}};

  const auto bucket_for = [&buckets](double polar_deg) -> const BucketInfo* {
    for (const BucketInfo& bucket : buckets) {
      const bool in_bucket =
          polar_deg >= bucket.min_deg &&
          (std::isinf(bucket.max_deg) || polar_deg < bucket.max_deg);
      if (in_bucket) {
        return &bucket;
      }
    }
    return nullptr;
  };

  struct Key {
    std::string split;
    int board_id = -1;
    int camera_index = -1;
    std::string point_type;
    std::string bucket;

    bool operator<(const Key& other) const {
      return std::tie(split, board_id, camera_index, point_type, bucket) <
             std::tie(other.split, other.board_id, other.camera_index,
                      other.point_type, other.bucket);
    }
  };

  struct Accumulator {
    double polar_min_deg = 0.0;
    double polar_max_deg = 0.0;
    int point_count = 0;
    double pixel_sq_sum = 0.0;
    double angular_sq_sum = 0.0;
    double chordal_sq_sum = 0.0;
    double pixel_sum = 0.0;
    double angular_deg_sum = 0.0;
    double max_pixel = 0.0;
    double max_angular_deg = 0.0;
  };

  std::map<Key, Accumulator> accumulators;
  const auto add_point =
      [&](const std::string& split,
          int board_id,
          int camera_index,
          JointPointType point_type,
          const BucketInfo& bucket,
          double pixel_error_px,
          double angular_error_rad,
          double chordal_error) {
        if (!std::isfinite(pixel_error_px) ||
            !std::isfinite(angular_error_rad) ||
            !std::isfinite(chordal_error)) {
          return;
        }
        const std::string typed =
            point_type == JointPointType::Outer ? "outer" : "internal";
        const std::array<std::string, 2> point_types = {{"all", typed}};
        const std::array<int, 2> camera_indices = {{-1, camera_index}};
        for (const std::string& point_type_name : point_types) {
          for (int cam : camera_indices) {
            Key key{split, board_id, cam, point_type_name, bucket.name};
            Accumulator& acc = accumulators[key];
            acc.polar_min_deg = bucket.min_deg;
            acc.polar_max_deg = bucket.max_deg;
            ++acc.point_count;
            acc.pixel_sq_sum += pixel_error_px * pixel_error_px;
            acc.angular_sq_sum += angular_error_rad * angular_error_rad;
            acc.chordal_sq_sum += chordal_error * chordal_error;
            acc.pixel_sum += pixel_error_px;
            const double angular_deg = angular_error_rad * 180.0 / M_PI;
            acc.angular_deg_sum += angular_deg;
            acc.max_pixel = std::max(acc.max_pixel, pixel_error_px);
            acc.max_angular_deg =
                std::max(acc.max_angular_deg, angular_deg);
          }
        }
      };

  const DoubleSphereCameraModel cam0 =
      DoubleSphereCameraModel::FromConfig(MakeCameraConfig(result.optimized_scene.cam0));
  const DoubleSphereCameraModel cam1 =
      DoubleSphereCameraModel::FromConfig(MakeCameraConfig(result.optimized_scene.cam1));
  const Eigen::Isometry3d T_cam1_cam0 =
      ToIsometry3d(result.optimized_scene.T_cam1_cam0);
  const auto normalize_or_nan = [](const Eigen::Vector3d& value)
      -> Eigen::Vector3d {
    const double norm = value.norm();
    if (!(norm > 0.0) || !std::isfinite(norm)) {
      return Eigen::Vector3d(std::numeric_limits<double>::quiet_NaN(),
                             std::numeric_limits<double>::quiet_NaN(),
                             std::numeric_limits<double>::quiet_NaN());
    }
    return value / norm;
  };
  const auto angular_error = [](const Eigen::Vector3d& a,
                                const Eigen::Vector3d& b) {
    if (!a.allFinite() || !b.allFinite()) {
      return std::numeric_limits<double>::quiet_NaN();
    }
    const double dot = std::max(-1.0, std::min(1.0, a.dot(b)));
    return std::atan2(a.cross(b).norm(), dot);
  };

  const StereoMeasurementDataset& dataset =
      result.problem_input.measurement_dataset;
  std::map<int, Eigen::Matrix4d> holdout_refit_pose_by_pair;
  std::set<int> holdout_refit_failed_pairs;
  std::map<std::pair<int, int>, Eigen::Matrix4d>
      holdout_board_refit_pose_by_pair_board;
  std::set<std::pair<int, int>> holdout_board_refit_failed_pair_boards;

  const auto process_observation =
      [&](const StereoObservation& observation,
          const std::string& split,
          const Eigen::Matrix4d& T_cam0_world,
          bool pose_is_cam0_board) {
        const DoubleSphereCameraModel& camera =
            observation.camera_index == 0 ? cam0 : cam1;
        Eigen::Vector3d q_obs_raw = Eigen::Vector3d::Zero();
        if (!camera.keypointToEuclidean(observation.observed_image_xy,
                                        &q_obs_raw)) {
          return;
        }
        const Eigen::Vector3d q_obs = normalize_or_nan(q_obs_raw);
        if (!q_obs.allFinite()) {
          return;
        }
        const double polar_deg =
            std::acos(std::max(-1.0, std::min(1.0, q_obs.z()))) *
            180.0 / M_PI;
        const BucketInfo* bucket = bucket_for(polar_deg);
        if (bucket == nullptr) {
          return;
        }

        const Eigen::Vector4d point_board(observation.target_point_board.x(),
                                          observation.target_point_board.y(),
                                          observation.target_point_board.z(),
                                          1.0);
        Eigen::Vector3d point_cam0 = Eigen::Vector3d::Zero();
        if (pose_is_cam0_board) {
          point_cam0 = (T_cam0_world * point_board).head<3>();
        } else {
          const auto board_pose_it =
              result.optimized_scene.T_world_board_by_id.find(
                  observation.board_id);
          if (board_pose_it ==
              result.optimized_scene.T_world_board_by_id.end()) {
            return;
          }
          const Eigen::Vector4d point_world =
              board_pose_it->second * point_board;
          point_cam0 = (T_cam0_world * point_world).head<3>();
        }
        const Eigen::Vector3d point_camera =
            observation.camera_index == 0 ? point_cam0
                                          : T_cam1_cam0 * point_cam0;
        const Eigen::Vector3d q_pred = normalize_or_nan(point_camera);
        if (!q_pred.allFinite()) {
          return;
        }
        Eigen::Vector2d predicted =
            Eigen::Vector2d(std::numeric_limits<double>::quiet_NaN(),
                            std::numeric_limits<double>::quiet_NaN());
        if (!camera.vsEuclideanToKeypoint(point_camera, &predicted)) {
          return;
        }
        const double pixel_error_px =
            (predicted - observation.observed_image_xy).norm();
        const double angular_error_rad = angular_error(q_obs, q_pred);
        const double chordal_error = (q_pred - q_obs).norm();
        add_point(split, observation.board_id, observation.camera_index,
                  observation.point_type, *bucket, pixel_error_px,
                  angular_error_rad, chordal_error);
      };

  for (const StereoObservation& observation : dataset.observations) {
    if (!observation.used_in_solver) {
      continue;
    }
    const StereoFramePair* pair = FindPair(dataset, observation.pair_index);
    if (pair == nullptr || pair->is_training) {
      continue;
    }

    if (holdout_refit_failed_pairs.count(observation.pair_index) == 0) {
      auto refit_it = holdout_refit_pose_by_pair.find(observation.pair_index);
      if (refit_it == holdout_refit_pose_by_pair.end()) {
        RuntimeCounters runtime_counters;
        StereoRefitDiagnostics refit_diagnostics;
        Eigen::Matrix4d refit_pose = Eigen::Matrix4d::Identity();
        if (!RefitPairPoseFromStereoOuterObservations(
                dataset, result.optimized_scene, observation.pair_index,
                result.problem_input.solver_options, &runtime_counters,
                &refit_diagnostics, &refit_pose)) {
          holdout_refit_failed_pairs.insert(observation.pair_index);
        } else {
          refit_it =
              holdout_refit_pose_by_pair
                  .insert(std::make_pair(observation.pair_index, refit_pose))
                  .first;
        }
      }
      if (refit_it != holdout_refit_pose_by_pair.end()) {
        process_observation(observation, "holdout_refit", refit_it->second,
                            false);
      }
    }

    const std::pair<int, int> pair_board_key(observation.pair_index,
                                             observation.board_id);
    if (holdout_board_refit_failed_pair_boards.count(pair_board_key) > 0) {
      continue;
    }
    auto board_refit_it =
        holdout_board_refit_pose_by_pair_board.find(pair_board_key);
    if (board_refit_it == holdout_board_refit_pose_by_pair_board.end()) {
      Eigen::Matrix4d refit_pose = Eigen::Matrix4d::Identity();
      if (!RefitStereoBoardPoseForVisualization(
              dataset, result.optimized_scene, observation.pair_index,
              observation.board_id,
              result.problem_input.solver_options.symmetric_refit_max_iterations,
              result.problem_input.solver_options.symmetric_refit_step,
              &refit_pose)) {
        holdout_board_refit_failed_pair_boards.insert(pair_board_key);
      } else {
        board_refit_it =
            holdout_board_refit_pose_by_pair_board
                .insert(std::make_pair(pair_board_key, refit_pose))
                .first;
      }
    }
    if (board_refit_it != holdout_board_refit_pose_by_pair_board.end()) {
      process_observation(observation, "holdout_extrinsic_only",
                          board_refit_it->second, true);
    }
  }

  for (const auto& entry : accumulators) {
    const Key& key = entry.first;
    const Accumulator& acc = entry.second;
    if (acc.point_count <= 0) {
      continue;
    }
    const double inv_count = 1.0 / static_cast<double>(acc.point_count);
    const double pixel_rmse = std::sqrt(acc.pixel_sq_sum * inv_count);
    const double angular_rmse_rad = std::sqrt(acc.angular_sq_sum * inv_count);
    const double angular_rmse_deg = angular_rmse_rad * 180.0 / M_PI;
    const double chordal_rmse = std::sqrt(acc.chordal_sq_sum * inv_count);
    output << key.split << "," << key.board_id << ","
           << key.camera_index << "," << key.point_type << ","
           << key.bucket << "," << acc.polar_min_deg << ",";
    if (std::isinf(acc.polar_max_deg)) {
      output << "inf";
    } else {
      output << acc.polar_max_deg;
    }
    output << "," << acc.point_count << "," << pixel_rmse << ","
           << angular_rmse_rad << "," << angular_rmse_deg << ","
           << chordal_rmse << "," << acc.pixel_sum * inv_count << ","
           << acc.angular_deg_sum * inv_count << "," << acc.max_pixel << ","
           << acc.max_angular_deg << "\n";
  }
}

namespace {

double BoardRmseOrNan(const StereoResidualSummary& summary, int board_id) {
  for (const StereoBoardResidualSummary& board : summary.board_summaries) {
    if (board.board_id == board_id) {
      return board.rmse;
    }
  }
  return std::numeric_limits<double>::quiet_NaN();
}

double CameraRmseOrNan(const StereoResidualSummary& summary, int camera_index) {
  for (const StereoCameraResidualSummary& camera : summary.camera_summaries) {
    if (camera.camera_index == camera_index) {
      return camera.rmse;
    }
  }
  return std::numeric_limits<double>::quiet_NaN();
}

std::string CsvDouble(double value) {
  if (!std::isfinite(value)) {
    return "nan";
  }
  std::ostringstream stream;
  stream << std::setprecision(12) << value;
  return stream.str();
}

std::string SanitizeFilename(std::string value) {
  for (char& ch : value) {
    if (!(std::isalnum(static_cast<unsigned char>(ch)) || ch == '_' ||
          ch == '-')) {
      ch = '_';
    }
  }
  return value;
}

double ReadCoObsSummaryScalar(const fs::path& summary_path,
                              const std::string& key) {
  std::ifstream input(summary_path.string().c_str());
  if (!input.is_open()) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  const std::string prefix = key + ":";
  std::string line;
  while (std::getline(input, line)) {
    if (line.compare(0, prefix.size(), prefix) != 0) {
      continue;
    }
    const std::string value = TrimCopy(line.substr(prefix.size()));
    if (value.empty() || value == "nan") {
      return std::numeric_limits<double>::quiet_NaN();
    }
    return std::stod(value);
  }
  return std::numeric_limits<double>::quiet_NaN();
}

struct CoObsFactorVariant {
  std::string name;
  double stereo_weight = 0.0;
  double layout_weight = 0.0;
  bool use_stereo = false;
  bool use_layout = false;
};

std::vector<CoObsFactorVariant> BuildCoObsFactorVariants(
    const StereoExtrinsicSolverOptions& options) {
  std::vector<CoObsFactorVariant> variants;
  variants.push_back(CoObsFactorVariant{"baseline_pixel", 0.0, 0.0,
                                        false, false});
  if (!options.coobs_factor_ba_run_experiment_matrix) {
    return variants;
  }
  for (double weight : options.coobs_factor_ba_stereo_weights) {
    variants.push_back(CoObsFactorVariant{
        "pixel_plus_stereo_factor_" + CsvDouble(weight), weight, 0.0,
        true, false});
  }
  for (double weight : options.coobs_factor_ba_layout_weights) {
    variants.push_back(CoObsFactorVariant{
        "pixel_plus_layout_factor_" + CsvDouble(weight), 0.0, weight,
        false, true});
  }
  for (double stereo_weight :
       options.coobs_factor_ba_combined_stereo_weights) {
    for (double layout_weight :
         options.coobs_factor_ba_combined_layout_weights) {
      variants.push_back(CoObsFactorVariant{
          "pixel_plus_stereo_layout_factor_s" + CsvDouble(stereo_weight) +
              "_l" + CsvDouble(layout_weight),
          stereo_weight, layout_weight, true, true});
    }
  }
  return variants;
}

}  // namespace

void WriteCoObsFactorBaExperiment(
    const std::string& directory,
    const StereoExtrinsicCalibrationResult& baseline_result) {
  const fs::path output_dir(directory);
  fs::create_directories(output_dir);
  const StereoExtrinsicSolverOptions& base_options =
      baseline_result.problem_input.solver_options;
  if (!base_options.coobs_factor_ba_enable) {
    return;
  }

  const StereoMeasurementDataset& dataset =
      baseline_result.problem_input.measurement_dataset;
  const StereoPairSelectionSummary& selection =
      baseline_result.pair_selection_summary;
  const std::set<int> selected_pairs(selection.selected_pair_indices.begin(),
                                     selection.selected_pair_indices.end());
  const std::set<int> holdout_pairs(dataset.holdout_pair_indices.begin(),
                                    dataset.holdout_pair_indices.end());
  const StereoMeasurementDataset selected_eval_dataset =
      MakePairBoardMaskedDataset(dataset, selection.selected_pair_board_keys);
  const StereoResidualEvaluator training_evaluator(
      StereoResidualEvaluationOptions{
          false, base_options.pair_pose_refit_mode,
          base_options.symmetric_refit_max_iterations,
          base_options.symmetric_refit_step, false});
  const StereoResidualEvaluator holdout_extrinsic_only_evaluator(
      StereoResidualEvaluationOptions{
          false, base_options.pair_pose_refit_mode,
          base_options.symmetric_refit_max_iterations,
          base_options.symmetric_refit_step, true});

  std::ofstream summary_csv(
      (output_dir / "coobs_factor_ba_summary.csv").string().c_str());
  summary_csv << std::setprecision(12);
  summary_csv
      << "variant_name,lambda_stereo,lambda_layout,"
      << "num_pixel_residuals,num_stereo_factors,num_layout_factors,"
      << "training_rmse,extrinsic_only_holdout_rmse,"
      << "extrinsic_only_holdout_cam0_rmse,"
      << "extrinsic_only_holdout_cam1_rmse,"
      << "extrinsic_only_holdout_cam_gap,"
      << "extrinsic_only_holdout_board4_rmse,"
      << "extrinsic_only_holdout_board5_rmse,"
      << "C_pose_median_rot,C_layout_median_rot,C_stereo_median_rot,"
      << "T_1_0_rot_drift_deg,T_1_0_trans_drift,"
      << "board4_layout_rot_drift_deg,board5_layout_rot_drift_deg,"
      << "solver_iterations,solver_final_cost,solver_success,notes\n";

  std::ofstream by_board_csv(
      (output_dir / "coobs_factor_ba_by_board.csv").string().c_str());
  by_board_csv << "variant_name,split,board_id,point_count,rmse\n";
  std::ofstream by_camera_csv(
      (output_dir / "coobs_factor_ba_by_camera.csv").string().c_str());
  by_camera_csv << "variant_name,split,camera_index,point_count,rmse\n";
  std::ofstream layout_pairs_csv(
      (output_dir / "coobs_factor_ba_layout_pairs.csv").string().c_str());
  layout_pairs_csv
      << "variant_name,selected_layout_pair_a,selected_layout_pair_b\n";
  std::ofstream polar_csv(
      (output_dir / "coobs_factor_ba_by_polar_bucket.csv").string().c_str());
  polar_csv << "variant_name,polar_csv_path\n";

  for (const CoObsFactorVariant& variant :
       BuildCoObsFactorVariants(base_options)) {
    StereoSceneState scene = baseline_result.optimized_scene;
    StereoGlobalSparseBaSummary ba_summary;
    StereoExtrinsicSolverOptions variant_options = base_options;
    variant_options.skip_final_global_ba = false;
    variant_options.final_ba_residual_mode = StereoFinalBaResidualMode::Pixel;
    variant_options.selection_ba_residual_mode =
        StereoFinalBaResidualMode::Pixel;
    variant_options.final_ba_optimize_intrinsics = false;
    variant_options.rig_param_mode = StereoRigParamMode::Cam0Reference;
    variant_options.coobs_factor_ba_apply_stereo_factor =
        variant.use_stereo;
    variant_options.coobs_factor_ba_apply_layout_factor =
        variant.use_layout;
    variant_options.coobs_factor_ba_current_stereo_weight =
        variant.stereo_weight;
    variant_options.coobs_factor_ba_current_layout_weight =
        variant.layout_weight;

    bool solver_success = true;
    std::string notes;
    if (variant.use_stereo || variant.use_layout) {
      solver_success =
          RunGlobalSparseBa(dataset, variant_options, selection, &ba_summary,
                            &scene);
      if (!solver_success) {
        notes = ba_summary.failure_reason;
      }
    } else {
      ba_summary = baseline_result.global_sparse_ba_summary;
      ba_summary.coobs_stereo_factor_count = 0;
      ba_summary.coobs_layout_factor_count = 0;
      ba_summary.coobs_stereo_factor_weight = 0.0;
      ba_summary.coobs_layout_factor_weight = 0.0;
    }

    const StereoResidualSummary training_summary =
        training_evaluator.Evaluate(selected_eval_dataset, scene,
                                    selected_pairs, "training_selected");
    const StereoResidualSummary holdout_summary =
        holdout_extrinsic_only_evaluator.Evaluate(
            dataset, scene, holdout_pairs, "holdout_extrinsic_only");

    StereoExtrinsicCalibrationResult variant_result = baseline_result;
    variant_result.optimized_scene = scene;
    variant_result.global_sparse_ba_summary = ba_summary;
    variant_result.training_residual_summary = training_summary;
    variant_result.holdout_extrinsic_only_residual_summary = holdout_summary;
    variant_result.problem_input.solver_options = variant_options;

    const fs::path variant_dir =
        output_dir / SanitizeFilename(variant.name);
    fs::create_directories(variant_dir);
    WriteStereoHoldoutBoardPolarRmseCsv(
        (variant_dir / "stereo_holdout_board_polar_rmse.csv").string(),
        variant_result);
    polar_csv << variant.name << ","
              << (variant_dir / "stereo_holdout_board_polar_rmse.csv").string()
              << "\n";

    MultiBoardCoObservationOptions coobs_options;
    coobs_options.enabled = true;
    coobs_options.output_dir = (variant_dir / "coobs_diagnostics").string();
    coobs_options.min_corners_per_group =
        base_options.coobs_min_corners_per_group;
    coobs_options.high_polar_threshold_deg =
        base_options.coobs_high_polar_threshold_deg;
    coobs_options.very_high_polar_threshold_deg =
        base_options.coobs_very_high_polar_threshold_deg;
    coobs_options.enable_rescue_suggestions = false;
    const MultiBoardCoObservationConsistency coobs(coobs_options);
    coobs.Evaluate(variant_result);
    const fs::path coobs_summary_path =
        variant_dir / "coobs_diagnostics" / "coobs_summary.txt";
    const double c_pose =
        ReadCoObsSummaryScalar(coobs_summary_path, "median_C_pose_rot_deg");
    const double c_layout =
        ReadCoObsSummaryScalar(coobs_summary_path, "median_C_layout_rot_deg");
    const double c_stereo =
        ReadCoObsSummaryScalar(coobs_summary_path, "median_C_stereo_rot_deg");

    const Eigen::Isometry3d T_before =
        ToIsometry3d(baseline_result.optimized_scene.T_cam1_cam0);
    const Eigen::Isometry3d T_after = ToIsometry3d(scene.T_cam1_cam0);
    const double stereo_rot_drift_deg =
        RotationDistanceRadians(T_before, T_after) * 180.0 / M_PI;
    const double stereo_trans_drift =
        (T_after.translation() - T_before.translation()).norm();
    const auto board_layout_drift_deg = [&](int board_id) {
      const auto before_it =
          baseline_result.optimized_scene.T_world_board_by_id.find(board_id);
      const auto after_it = scene.T_world_board_by_id.find(board_id);
      if (before_it ==
              baseline_result.optimized_scene.T_world_board_by_id.end() ||
          after_it == scene.T_world_board_by_id.end()) {
        return std::numeric_limits<double>::quiet_NaN();
      }
      return RotationDistanceRadians(ToIsometry3d(before_it->second),
                                     ToIsometry3d(after_it->second)) *
             180.0 / M_PI;
    };

    const double cam0 = CameraRmseOrNan(holdout_summary, 0);
    const double cam1 = CameraRmseOrNan(holdout_summary, 1);
    summary_csv
        << variant.name << "," << variant.stereo_weight << ","
        << variant.layout_weight << ","
        << ba_summary.reprojection_error_count << ","
        << ba_summary.coobs_stereo_factor_count << ","
        << ba_summary.coobs_layout_factor_count << ","
        << training_summary.total_stereo_rmse << ","
        << holdout_summary.total_stereo_rmse << ","
        << CsvDouble(cam0) << "," << CsvDouble(cam1) << ","
        << CsvDouble(std::abs(cam1 - cam0)) << ","
        << CsvDouble(BoardRmseOrNan(holdout_summary, 4)) << ","
        << CsvDouble(BoardRmseOrNan(holdout_summary, 5)) << ","
        << CsvDouble(c_pose) << "," << CsvDouble(c_layout) << ","
        << CsvDouble(c_stereo) << ","
        << CsvDouble(stereo_rot_drift_deg) << ","
        << CsvDouble(stereo_trans_drift) << ","
        << CsvDouble(board_layout_drift_deg(4)) << ","
        << CsvDouble(board_layout_drift_deg(5)) << ","
        << ba_summary.iterations << "," << ba_summary.objective_final << ","
        << (solver_success ? 1 : 0) << "," << notes << "\n";

    for (const StereoBoardResidualSummary& board :
         holdout_summary.board_summaries) {
      by_board_csv << variant.name << ",holdout_extrinsic_only,"
                   << board.board_id << "," << board.point_count << ","
                   << board.rmse << "\n";
    }
    for (const StereoCameraResidualSummary& camera :
         holdout_summary.camera_summaries) {
      by_camera_csv << variant.name << ",holdout_extrinsic_only,"
                    << camera.camera_index << "," << camera.point_count
                    << "," << camera.rmse << "\n";
    }
    for (const std::pair<int, int>& pair :
         base_options.coobs_factor_ba_layout_selected_pairs) {
      layout_pairs_csv << variant.name << "," << pair.first << ","
                       << pair.second << "\n";
    }
  }
  std::cout << "[Stage6][CoObsFactorBA] wrote experiment matrix to "
            << (output_dir / "coobs_factor_ba_summary.csv").string()
            << std::endl;
}

void WriteStereoReprojectionVisualizations(
    const std::string& directory,
    const StereoExtrinsicCalibrationResult& result,
    int top_k) {
  auto write_subset =
      [&](const std::string& subset_directory,
          const std::vector<StereoPairResidualSummary>& source_summaries,
          const std::set<int>& include_pairs,
          const std::map<int, std::string>& status_labels,
          const std::string& summary_label) {
        fs::create_directories(subset_directory);
        std::ofstream note(
            (fs::path(subset_directory) / "visualization_summary.txt").string().c_str());
        note << "label: " << summary_label << "\n";
        note << "top_k: " << top_k << "\n";
        note << "requested_pair_count: " << include_pairs.size() << "\n";
        const DoubleSphereCameraModel cam0 =
            DoubleSphereCameraModel::FromConfig(MakeCameraConfig(result.optimized_scene.cam0));
        const DoubleSphereCameraModel cam1 =
            DoubleSphereCameraModel::FromConfig(MakeCameraConfig(result.optimized_scene.cam1));
        const Eigen::Isometry3d T_cam1_cam0 =
            ToIsometry3d(result.optimized_scene.T_cam1_cam0);
        const bool use_training_mask = summary_label.find("training") != std::string::npos;
        const bool use_extrinsic_only_local_board_pose =
            summary_label.find("extrinsic_only") != std::string::npos;

        std::vector<StereoPairResidualSummary> pair_summaries;
        pair_summaries.reserve(source_summaries.size());
        for (const StereoPairResidualSummary& pair_summary : source_summaries) {
          if (include_pairs.empty() ||
              include_pairs.count(pair_summary.pair_index) > 0) {
            pair_summaries.push_back(pair_summary);
          }
        }
        std::sort(pair_summaries.begin(), pair_summaries.end(),
                  [](const StereoPairResidualSummary& lhs,
                     const StereoPairResidualSummary& rhs) {
                    return lhs.overall_rmse > rhs.overall_rmse;
                  });
        if (top_k > 0 && static_cast<int>(pair_summaries.size()) > top_k) {
          pair_summaries.resize(static_cast<std::size_t>(top_k));
        }

        const auto draw_pair =
            [&](const StereoPairResidualSummary& pair_summary,
                int camera_index,
                cv::Mat* image) {
              Eigen::Matrix4d T_cam0_world_matrix = Eigen::Matrix4d::Identity();
              bool have_pair_pose = use_extrinsic_only_local_board_pose;
              if (!use_extrinsic_only_local_board_pose) {
                have_pair_pose = false;
                const auto pair_pose_it =
                    result.optimized_scene.T_cam0_world_by_pair.find(pair_summary.pair_index);
                if (pair_pose_it != result.optimized_scene.T_cam0_world_by_pair.end()) {
                  T_cam0_world_matrix = pair_pose_it->second;
                  have_pair_pose = true;
                }
                if (!have_pair_pose &&
                    summary_label.find("holdout") != std::string::npos) {
                  StereoExtrinsicSolverOptions holdout_refit_options;
                  holdout_refit_options.pair_pose_refit_mode =
                      StereoPairPoseRefitMode::StereoSymmetric;
                  holdout_refit_options.symmetric_refit_max_iterations = 8;
                  holdout_refit_options.symmetric_refit_step = 1e-3;
                  RuntimeCounters runtime_counters;
                  StereoRefitDiagnostics refit_diagnostics;
                  have_pair_pose = RefitPairPoseFromStereoOuterObservations(
                      result.problem_input.measurement_dataset,
                      result.optimized_scene,
                      pair_summary.pair_index,
                      holdout_refit_options,
                      &runtime_counters,
                      &refit_diagnostics,
                      &T_cam0_world_matrix);
                }
                if (!have_pair_pose) {
                  return;
                }
              }
              const Eigen::Isometry3d T_cam0_world =
                  ToIsometry3d(T_cam0_world_matrix);
              std::map<int, Eigen::Matrix4d> local_T_cam0_board_by_board_id;
              std::set<int> failed_local_board_ids;
              for (const StereoObservation& observation :
                   result.problem_input.measurement_dataset.observations) {
                if (observation.pair_index != pair_summary.pair_index ||
                    observation.camera_index != camera_index ||
                    !observation.used_in_solver) {
                  continue;
                }
                if (use_training_mask &&
                    !PairBoardSelected(result.pair_selection_summary,
                                       observation.pair_index,
                                       observation.board_id)) {
                  continue;
                }
                const Eigen::Vector4d point_board(observation.target_point_board.x(),
                                                  observation.target_point_board.y(),
                                                  observation.target_point_board.z(),
                                                  1.0);
                Eigen::Vector3d point_cam0 = Eigen::Vector3d::Zero();
                if (use_extrinsic_only_local_board_pose) {
                  if (!ContainsBoard(result.problem_input.measurement_dataset
                                         .pair_shared_board_ids,
                                     pair_summary.pair_index,
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
                    if (!RefitStereoBoardPoseForVisualization(
                            result.problem_input.measurement_dataset,
                            result.optimized_scene,
                            pair_summary.pair_index,
                            observation.board_id,
                            result.problem_input.solver_options
                                .symmetric_refit_max_iterations,
                            result.problem_input.solver_options.symmetric_refit_step,
                            &T_cam0_board)) {
                      failed_local_board_ids.insert(observation.board_id);
                      continue;
                    }
                    local_pose_it =
                        local_T_cam0_board_by_board_id
                            .insert(std::make_pair(observation.board_id,
                                                   T_cam0_board))
                            .first;
                  }
                  point_cam0 = (local_pose_it->second * point_board).head<3>();
                } else {
                  const auto board_pose_it =
                      result.optimized_scene.T_world_board_by_id.find(
                          observation.board_id);
                  if (board_pose_it ==
                      result.optimized_scene.T_world_board_by_id.end()) {
                    continue;
                  }
                  const Eigen::Vector4d point_world =
                      board_pose_it->second * point_board;
                  point_cam0 = (T_cam0_world * point_world).head<3>();
                }
                Eigen::Vector2d predicted = Eigen::Vector2d::Zero();
                bool ok = false;
                if (camera_index == 0) {
                  ok = cam0.vsEuclideanToKeypoint(point_cam0, &predicted);
                } else {
                  ok = cam1.vsEuclideanToKeypoint(T_cam1_cam0 * point_cam0, &predicted);
                }
                if (!ok) {
                  continue;
                }
                const bool is_internal = observation.point_type == JointPointType::Internal;
                const cv::Point observed_pt(
                    static_cast<int>(std::lround(observation.observed_image_xy.x())),
                    static_cast<int>(std::lround(observation.observed_image_xy.y())));
                const cv::Point predicted_pt(
                    static_cast<int>(std::lround(predicted.x())),
                    static_cast<int>(std::lround(predicted.y())));
                DrawStereoResidualObservation(image, observed_pt, predicted_pt,
                                             is_internal);
              }
            };

        int index = 0;
        for (const StereoPairResidualSummary& pair_summary : pair_summaries) {
          const StereoFramePair* pair =
              FindPair(result.problem_input.measurement_dataset, pair_summary.pair_index);
          if (pair == nullptr) {
            continue;
          }
          cv::Mat cam0_img = cv::imread(pair->left_image_path, cv::IMREAD_COLOR);
          cv::Mat cam1_img = cv::imread(pair->right_image_path, cv::IMREAD_COLOR);
          if (cam0_img.empty() || cam1_img.empty()) {
            continue;
          }
          draw_pair(pair_summary, 0, &cam0_img);
          draw_pair(pair_summary, 1, &cam1_img);
          auto status_it = status_labels.find(pair_summary.pair_index);
          const std::string status =
              status_it == status_labels.end() ? "included" : status_it->second;
          const std::string title =
              "pair=" + std::to_string(pair_summary.pair_index) +
              " rmse=" + std::to_string(pair_summary.overall_rmse) +
              " " + status;
          const std::string extra =
              "pair_index=" + std::to_string(pair_summary.pair_index) +
              ", rmse=" + std::to_string(pair_summary.overall_rmse);
          DrawStereoVisualizationLegend(&cam0_img, status, extra);
          DrawStereoVisualizationLegend(&cam1_img, status, extra);
          cv::putText(cam0_img, "cam0 " + title, cv::Point(20, 30),
                      cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 255), 2);
          cv::putText(cam1_img, "cam1 " + title, cv::Point(20, 30),
                      cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 255), 2);
          cv::Mat side_by_side;
          cv::hconcat(cam0_img, cam1_img, side_by_side);
          const std::string prefix =
              MakeStereoVisualizationPrefix(index, pair_summary, *pair);
          cv::imwrite((fs::path(subset_directory) / (prefix + "_side_by_side.png")).string(),
                      side_by_side);
          note << prefix << ": pair_index=" << pair_summary.pair_index
               << ", overall_rmse=" << pair_summary.overall_rmse
               << ", status=" << status << "\n";
          ++index;
        }
      };

  std::map<int, std::string> default_status_labels;
  for (const StereoPairResidualSummary& pair_summary :
       result.training_residual_summary.pair_summaries) {
    default_status_labels[pair_summary.pair_index] =
        result.pair_selection_summary.selected_pair_indices.count(pair_summary.pair_index) > 0
            ? "selected"
            : "not_selected";
  }
  write_subset(directory,
               result.training_residual_summary.pair_summaries,
               std::set<int>(),
               default_status_labels,
               "training_reprojection_overview");
  write_subset((fs::path(directory) / "training_top_bad_side_by_side").string(),
               result.training_residual_summary.pair_summaries,
               std::set<int>(),
               default_status_labels,
               "training_top_bad_reprojection");
  write_subset((fs::path(directory) / "holdout_global_scene_top_bad_side_by_side").string(),
               result.holdout_residual_summary.pair_summaries,
               std::set<int>(),
               std::map<int, std::string>(),
               "holdout_top_bad_reprojection");
  write_subset((fs::path(directory) / "holdout_top_bad_side_by_side").string(),
               result.holdout_extrinsic_only_residual_summary.pair_summaries,
               std::set<int>(),
               std::map<int, std::string>(),
               "holdout_extrinsic_only_top_bad_reprojection");
}

void WriteStereoExtrinsicOnlyTopBadPairBoardVisualizations(
    const std::string& directory,
    const StereoExtrinsicCalibrationResult& result,
    const StereoSceneState& scene_state,
    const std::string& label,
    int top_k) {
  const fs::path output_dir = fs::path(directory);
  fs::create_directories(output_dir);
  std::ofstream csv(
      (output_dir / "extrinsic_only_top_bad_pair_boards.csv").string().c_str());
  csv << "label,rank,pair_index,left_frame_label,right_frame_label,board_id,"
      << "overall_rmse,outer_rmse,internal_rmse,cam0_rmse,cam1_rmse,"
      << "point_count,outer_point_count,internal_point_count,"
      << "cam0_point_count,cam1_point_count,local_stereo_outer_rmse\n";

  std::ofstream note(
      (output_dir / "visualization_summary.txt").string().c_str());
  note << "label: " << label << "\n";
  note << "top_k: " << top_k << "\n";
  note << "metric: holdout extrinsic-only pair-board RMSE with local stereo "
          "board pose refit\n";
  struct PairBoardResidual {
    int pair_index = -1;
    int board_id = -1;
    double overall_rmse = std::numeric_limits<double>::infinity();
    double outer_rmse = std::numeric_limits<double>::infinity();
    double internal_rmse = std::numeric_limits<double>::infinity();
    double cam0_rmse = std::numeric_limits<double>::infinity();
    double cam1_rmse = std::numeric_limits<double>::infinity();
    double local_stereo_outer_rmse = std::numeric_limits<double>::infinity();
    int point_count = 0;
    int outer_point_count = 0;
    int internal_point_count = 0;
    int cam0_point_count = 0;
    int cam1_point_count = 0;
    Eigen::Matrix4d T_cam0_board = Eigen::Matrix4d::Identity();
  };

  struct Accumulator {
    double total_sq = 0.0;
    int total_count = 0;
    double outer_sq = 0.0;
    int outer_count = 0;
    double internal_sq = 0.0;
    int internal_count = 0;
    double cam0_sq = 0.0;
    int cam0_count = 0;
    double cam1_sq = 0.0;
    int cam1_count = 0;
  };

  const auto rmse_or_inf = [](double squared_error_sum, int count) {
    if (count <= 0) {
      return std::numeric_limits<double>::infinity();
    }
    return std::sqrt(squared_error_sum / static_cast<double>(count));
  };

  const StereoMeasurementDataset& dataset =
      result.problem_input.measurement_dataset;
  const DoubleSphereCameraModel cam0 =
      DoubleSphereCameraModel::FromConfig(MakeCameraConfig(scene_state.cam0));
  const DoubleSphereCameraModel cam1 =
      DoubleSphereCameraModel::FromConfig(MakeCameraConfig(scene_state.cam1));
  const Eigen::Isometry3d T_cam1_cam0 =
      ToIsometry3d(scene_state.T_cam1_cam0);

  std::vector<PairBoardResidual> residuals;
  for (int pair_index : dataset.holdout_pair_indices) {
    const auto shared_it = dataset.pair_shared_board_ids.find(pair_index);
    if (shared_it == dataset.pair_shared_board_ids.end()) {
      continue;
    }
    for (int board_id : shared_it->second) {
      Eigen::Matrix4d T_cam0_board = Eigen::Matrix4d::Identity();
      if (!RefitStereoBoardPoseForVisualization(
              dataset,
              scene_state,
              pair_index,
              board_id,
              result.problem_input.solver_options.symmetric_refit_max_iterations,
              result.problem_input.solver_options.symmetric_refit_step,
              &T_cam0_board)) {
        continue;
      }

      Accumulator accum;
      for (const StereoObservation& observation : dataset.observations) {
        if (observation.pair_index != pair_index ||
            observation.board_id != board_id ||
            !observation.used_in_solver) {
          continue;
        }
        const Eigen::Vector4d point_board(observation.target_point_board.x(),
                                          observation.target_point_board.y(),
                                          observation.target_point_board.z(),
                                          1.0);
        const Eigen::Vector3d point_cam0 =
            (T_cam0_board * point_board).head<3>();
        Eigen::Vector2d predicted = Eigen::Vector2d::Zero();
        bool ok = false;
        if (observation.camera_index == 0) {
          ok = cam0.vsEuclideanToKeypoint(point_cam0, &predicted);
        } else {
          ok = cam1.vsEuclideanToKeypoint(T_cam1_cam0 * point_cam0,
                                          &predicted);
        }
        if (!ok) {
          continue;
        }
        const double squared_error =
            (predicted - observation.observed_image_xy).squaredNorm();
        accum.total_sq += squared_error;
        ++accum.total_count;
        if (observation.point_type == JointPointType::Outer) {
          accum.outer_sq += squared_error;
          ++accum.outer_count;
        } else {
          accum.internal_sq += squared_error;
          ++accum.internal_count;
        }
        if (observation.camera_index == 0) {
          accum.cam0_sq += squared_error;
          ++accum.cam0_count;
        } else {
          accum.cam1_sq += squared_error;
          ++accum.cam1_count;
        }
      }
      if (accum.total_count <= 0) {
        continue;
      }

      PairBoardResidual row;
      row.pair_index = pair_index;
      row.board_id = board_id;
      row.overall_rmse = rmse_or_inf(accum.total_sq, accum.total_count);
      row.outer_rmse = rmse_or_inf(accum.outer_sq, accum.outer_count);
      row.internal_rmse =
          rmse_or_inf(accum.internal_sq, accum.internal_count);
      row.cam0_rmse = rmse_or_inf(accum.cam0_sq, accum.cam0_count);
      row.cam1_rmse = rmse_or_inf(accum.cam1_sq, accum.cam1_count);
      row.point_count = accum.total_count;
      row.outer_point_count = accum.outer_count;
      row.internal_point_count = accum.internal_count;
      row.cam0_point_count = accum.cam0_count;
      row.cam1_point_count = accum.cam1_count;
      row.T_cam0_board = T_cam0_board;
      const std::vector<StereoBoardPoseObservation> outer_observations =
          CollectStereoBoardPoseObservations(dataset, pair_index, board_id);
      row.local_stereo_outer_rmse = EvaluateStereoBoardPoseRmse(
          outer_observations, scene_state, ToIsometry3d(T_cam0_board));
      residuals.push_back(row);
    }
  }

  std::sort(residuals.begin(), residuals.end(),
            [](const PairBoardResidual& lhs,
               const PairBoardResidual& rhs) {
              return lhs.overall_rmse > rhs.overall_rmse;
            });

  std::set<int> residual_pair_indices;
  for (const PairBoardResidual& row : residuals) {
    residual_pair_indices.insert(row.pair_index);
  }
  std::vector<int> visualized_pair_indices;
  for (int pair_index : dataset.holdout_pair_indices) {
    if (residual_pair_indices.count(pair_index) > 0) {
      visualized_pair_indices.push_back(pair_index);
    }
  }
  const int image_limit =
      top_k > 0 ? std::min(top_k, static_cast<int>(visualized_pair_indices.size()))
                : static_cast<int>(visualized_pair_indices.size());
  note << "candidate_pair_board_count: " << residuals.size() << "\n";
  note << "candidate_pair_count: " << visualized_pair_indices.size() << "\n";
  note << "visualized_pair_count: " << image_limit << "\n";
  note << "visual_order: holdout_pair_index_order\n";
  note << "png_layout: one whole stereo pair per image; all evaluable shared "
          "boards in that pair are drawn together\n";

  for (int index = 0; index < static_cast<int>(residuals.size()); ++index) {
    const PairBoardResidual& row = residuals[index];
    const StereoFramePair* pair = FindPair(dataset, row.pair_index);
    csv << label << "," << index << "," << row.pair_index << ","
        << (pair == nullptr ? "" : pair->left_frame_label) << ","
        << (pair == nullptr ? "" : pair->right_frame_label) << ","
        << row.board_id << "," << row.overall_rmse << ","
        << row.outer_rmse << "," << row.internal_rmse << ","
        << row.cam0_rmse << "," << row.cam1_rmse << ","
        << row.point_count << "," << row.outer_point_count << ","
        << row.internal_point_count << "," << row.cam0_point_count << ","
        << row.cam1_point_count << "," << row.local_stereo_outer_rmse
        << "\n";
  }

  std::map<int, std::vector<PairBoardResidual>> residuals_by_pair;
  for (const PairBoardResidual& row : residuals) {
    residuals_by_pair[row.pair_index].push_back(row);
  }

  for (int index = 0; index < image_limit; ++index) {
    const int pair_index = visualized_pair_indices[index];
    const auto rows_it = residuals_by_pair.find(pair_index);
    if (rows_it == residuals_by_pair.end() || rows_it->second.empty()) {
      continue;
    }
    const std::vector<PairBoardResidual>& pair_rows = rows_it->second;
    const PairBoardResidual& worst_row = pair_rows.front();
    const StereoFramePair* pair = FindPair(dataset, pair_index);
    if (pair == nullptr) {
      continue;
    }
    cv::Mat cam0_img = cv::imread(pair->left_image_path, cv::IMREAD_COLOR);
    cv::Mat cam1_img = cv::imread(pair->right_image_path, cv::IMREAD_COLOR);
    if (cam0_img.empty() || cam1_img.empty()) {
      continue;
    }
    std::map<int, const PairBoardResidual*> board_residuals;
    for (const PairBoardResidual& row : pair_rows) {
      board_residuals[row.board_id] = &row;
    }
    std::map<int, cv::Point2d> cam0_label_sum;
    std::map<int, cv::Point2d> cam1_label_sum;
    std::map<int, int> cam0_label_count;
    std::map<int, int> cam1_label_count;
    for (const StereoObservation& observation : dataset.observations) {
      if (observation.pair_index != pair_index || !observation.used_in_solver) {
        continue;
      }
      const auto board_it = board_residuals.find(observation.board_id);
      if (board_it == board_residuals.end() || board_it->second == nullptr) {
        continue;
      }
      const PairBoardResidual& row = *board_it->second;
      const Eigen::Vector4d point_board(observation.target_point_board.x(),
                                        observation.target_point_board.y(),
                                        observation.target_point_board.z(),
                                        1.0);
      const Eigen::Vector3d point_cam0 =
          (row.T_cam0_board * point_board).head<3>();
      Eigen::Vector2d predicted = Eigen::Vector2d::Zero();
      bool ok = false;
      if (observation.camera_index == 0) {
        ok = cam0.vsEuclideanToKeypoint(point_cam0, &predicted);
      } else {
        ok = cam1.vsEuclideanToKeypoint(T_cam1_cam0 * point_cam0,
                                        &predicted);
      }
      if (!ok) {
        continue;
      }
      const cv::Point observed_pt(
          static_cast<int>(std::lround(observation.observed_image_xy.x())),
          static_cast<int>(std::lround(observation.observed_image_xy.y())));
      const cv::Point predicted_pt(static_cast<int>(std::lround(predicted.x())),
                                   static_cast<int>(std::lround(predicted.y())));
      cv::Mat* image = observation.camera_index == 0 ? &cam0_img : &cam1_img;
      DrawStereoResidualObservation(
          image, observed_pt, predicted_pt,
          observation.point_type == JointPointType::Internal);
      if (observation.camera_index == 0) {
        cam0_label_sum[observation.board_id] +=
            cv::Point2d(observed_pt.x, observed_pt.y);
        ++cam0_label_count[observation.board_id];
      } else {
        cam1_label_sum[observation.board_id] +=
            cv::Point2d(observed_pt.x, observed_pt.y);
        ++cam1_label_count[observation.board_id];
      }
    }

    const auto draw_board_labels =
        [&](cv::Mat* image,
            const std::map<int, cv::Point2d>& label_sum,
            const std::map<int, int>& label_count) {
          if (image == nullptr || image->empty()) {
            return;
          }
          for (const auto& entry : label_sum) {
            const int board_id = entry.first;
            const auto count_it = label_count.find(board_id);
            const auto residual_it = board_residuals.find(board_id);
            if (count_it == label_count.end() || count_it->second <= 0 ||
                residual_it == board_residuals.end() ||
                residual_it->second == nullptr) {
              continue;
            }
            const cv::Point2d center =
                entry.second * (1.0 / static_cast<double>(count_it->second));
            const int x = std::max(12, std::min(image->cols - 260,
                                                static_cast<int>(center.x)));
            const int y = std::max(210, std::min(image->rows - 12,
                                                 static_cast<int>(center.y)));
            std::ostringstream board_label;
            board_label << "board=" << board_id << " rmse=" << std::fixed
                        << std::setprecision(2)
                        << residual_it->second->overall_rmse;
            cv::putText(*image,
                        board_label.str(),
                        cv::Point(x, y),
                        cv::FONT_HERSHEY_SIMPLEX,
                        0.55,
                        cv::Scalar(0, 255, 255),
                        2,
                        cv::LINE_AA);
          }
        };
    draw_board_labels(&cam0_img, cam0_label_sum, cam0_label_count);
    draw_board_labels(&cam1_img, cam1_label_sum, cam1_label_count);

    const std::string status = label;
    const std::string extra =
        "pair=" + std::to_string(pair_index) +
        ", boards=" + std::to_string(pair_rows.size()) +
        ", worst_board=" + std::to_string(worst_row.board_id) +
        ", worst_rmse=" + std::to_string(worst_row.overall_rmse);
    DrawStereoVisualizationLegend(&cam0_img, status, extra);
    DrawStereoVisualizationLegend(&cam1_img, status, extra);
    cv::putText(cam0_img, "cam0 all shared boards " + extra, cv::Point(20, 30),
                cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 255), 2);
    cv::putText(cam1_img, "cam1 all shared boards " + extra, cv::Point(20, 30),
                cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 255), 2);
    cv::Mat side_by_side;
    cv::hconcat(cam0_img, cam1_img, side_by_side);
    std::ostringstream prefix;
    prefix << "pair_" << std::setw(6) << std::setfill('0') << pair_index
           << "_" << SafeVisualizationToken(pair->left_frame_label)
           << "_" << SafeVisualizationToken(pair->right_frame_label)
           << "_all_boards";
    cv::imwrite((output_dir / (prefix.str() + "_side_by_side.png")).string(),
                side_by_side);
  }
}

void WriteStereoBackendInputVisualizations(
    const std::string& directory,
    const StereoExtrinsicCalibrationResult& result,
    int top_k) {
  std::vector<StereoPairResidualSummary> pair_summaries =
      result.training_residual_summary.pair_summaries;
  fs::create_directories(directory);

  std::ofstream summary(
      (fs::path(directory) / "backend_input_summary.txt").string().c_str());
  summary << "selected_pair_count: "
          << result.pair_selection_summary.selected_pair_count << "\n";
  summary << "eligible_pair_count: "
          << result.pair_selection_summary.eligible_pair_count << "\n";
  summary << "note: this directory is intended to show final training pairs that entered the backend.\n";

  auto write_selected_only =
      [&](const std::string& subdir) {
        fs::create_directories(subdir);
        std::ofstream note(
            (fs::path(subdir) / "visualization_summary.txt").string().c_str());
        note << "label: backend_input_selected_only\n";
        note << "top_k: " << top_k << "\n";
        note << "selected_pair_count: "
             << result.pair_selection_summary.selected_pair_count << "\n";
        note << "board_source_colors: seed=green, trial=orange, pair_cohesion=magenta, unknown=gray\n";
        note << "point_residual_colors: internal<=3px green, internal>3px yellow, residual_unavailable blue; outer markers use board source color\n";
        const DoubleSphereCameraModel cam0 =
            DoubleSphereCameraModel::FromConfig(MakeCameraConfig(result.optimized_scene.cam0));
        const DoubleSphereCameraModel cam1 =
            DoubleSphereCameraModel::FromConfig(MakeCameraConfig(result.optimized_scene.cam1));
        const Eigen::Isometry3d T_cam1_cam0 =
            ToIsometry3d(result.optimized_scene.T_cam1_cam0);
        const std::map<PairBoardKey, StereoBackendBoardSource>
            board_sources = BuildStereoBackendBoardSources(
                result.pair_board_trial_selection_summary);
        std::vector<StereoPairResidualSummary> selected_summaries;
        for (const StereoPairResidualSummary& pair_summary : pair_summaries) {
          if (result.pair_selection_summary.selected_pair_indices.count(
                  pair_summary.pair_index) > 0) {
            selected_summaries.push_back(pair_summary);
          }
        }
        std::sort(selected_summaries.begin(), selected_summaries.end(),
                  [](const StereoPairResidualSummary& lhs,
                     const StereoPairResidualSummary& rhs) {
                    return lhs.overall_rmse > rhs.overall_rmse;
                  });
        if (top_k > 0 && static_cast<int>(selected_summaries.size()) > top_k) {
          selected_summaries.resize(static_cast<std::size_t>(top_k));
        }
        const auto draw_pair =
            [&](const StereoPairResidualSummary& pair_summary,
                int camera_index,
                cv::Mat* image) {
              const auto pair_pose_it =
                  result.optimized_scene.T_cam0_world_by_pair.find(pair_summary.pair_index);
              if (pair_pose_it == result.optimized_scene.T_cam0_world_by_pair.end()) {
                return;
              }
              const Eigen::Isometry3d T_cam0_world = ToIsometry3d(pair_pose_it->second);
              struct BoardDrawInfo {
                std::vector<cv::Point> observed_points;
                int point_count = 0;
                int high_residual_count = 0;
                StereoBackendBoardSource source;
              };
              std::map<int, BoardDrawInfo> boards;
              for (const StereoObservation& observation :
                   result.problem_input.measurement_dataset.observations) {
                if (observation.pair_index != pair_summary.pair_index ||
                    observation.camera_index != camera_index ||
                    !observation.used_in_solver) {
                  continue;
                }
                if (!PairBoardSelected(result.pair_selection_summary,
                                       observation.pair_index,
                                       observation.board_id)) {
                  continue;
                }
                const auto board_pose_it =
                    result.optimized_scene.T_world_board_by_id.find(observation.board_id);
                if (board_pose_it == result.optimized_scene.T_world_board_by_id.end()) {
                  continue;
                }
                const Eigen::Vector4d point_board(observation.target_point_board.x(),
                                                  observation.target_point_board.y(),
                                                  observation.target_point_board.z(),
                                                  1.0);
                const Eigen::Vector4d point_world = board_pose_it->second * point_board;
                const Eigen::Vector3d point_cam0 =
                    (T_cam0_world * point_world).head<3>();
                Eigen::Vector2d predicted = Eigen::Vector2d::Zero();
                bool ok = false;
                if (camera_index == 0) {
                  ok = cam0.vsEuclideanToKeypoint(point_cam0, &predicted);
                } else {
                  ok = cam1.vsEuclideanToKeypoint(T_cam1_cam0 * point_cam0, &predicted);
                }
                if (!ok) {
                  continue;
                }
                const bool is_internal = observation.point_type == JointPointType::Internal;
                const cv::Point observed_pt(
                    static_cast<int>(std::lround(observation.observed_image_xy.x())),
                    static_cast<int>(std::lround(observation.observed_image_xy.y())));
                const cv::Point predicted_pt(
                    static_cast<int>(std::lround(predicted.x())),
                    static_cast<int>(std::lround(predicted.y())));
                const double residual_norm =
                    (predicted - observation.observed_image_xy).norm();
                const PairBoardKey key(observation.pair_index,
                                       observation.board_id);
                BoardDrawInfo& board_info = boards[observation.board_id];
                const auto source_it = board_sources.find(key);
                if (source_it != board_sources.end()) {
                  board_info.source = source_it->second;
                }
                board_info.observed_points.push_back(observed_pt);
                ++board_info.point_count;
                if (is_internal && residual_norm > 3.0) {
                  ++board_info.high_residual_count;
                }
                const cv::Scalar board_color =
                    StereoBackendBoardSourceColor(board_info.source);
                DrawStereoBackendObservation(image, observed_pt, predicted_pt,
                                             is_internal, board_color, true,
                                             residual_norm);
              }
              for (const auto& entry : boards) {
                const int board_id = entry.first;
                const BoardDrawInfo& board_info = entry.second;
                if (board_info.observed_points.empty()) {
                  continue;
                }
                std::vector<cv::Point> hull;
                cv::convexHull(board_info.observed_points, hull);
                const cv::Scalar board_color =
                    StereoBackendBoardSourceColor(board_info.source);
                if (hull.size() >= 3) {
                  cv::polylines(*image, hull, true, board_color, 2,
                                cv::LINE_AA);
                } else {
                  const cv::Rect bounds =
                      cv::boundingRect(board_info.observed_points);
                  cv::rectangle(*image, bounds, board_color, 2, cv::LINE_AA);
                }
                cv::Rect bounds = cv::boundingRect(board_info.observed_points);
                std::ostringstream label;
                label << "B" << board_id << ":"
                      << StereoBackendBoardSourceLabel(board_info.source)
                      << " n=" << board_info.point_count
                      << " hi=" << board_info.high_residual_count;
                const cv::Point text_origin(
                    std::max(8, bounds.x),
                    std::max(18, bounds.y - 6));
                cv::putText(*image, label.str(), text_origin,
                            cv::FONT_HERSHEY_SIMPLEX, 0.45, board_color, 2,
                            cv::LINE_AA);
              }
            };
        int index = 0;
        for (const StereoPairResidualSummary& pair_summary : selected_summaries) {
          const StereoFramePair* pair =
              FindPair(result.problem_input.measurement_dataset, pair_summary.pair_index);
          if (pair == nullptr) {
            continue;
          }
          cv::Mat cam0_img = cv::imread(pair->left_image_path, cv::IMREAD_COLOR);
          cv::Mat cam1_img = cv::imread(pair->right_image_path, cv::IMREAD_COLOR);
          if (cam0_img.empty() || cam1_img.empty()) {
            continue;
          }
          draw_pair(pair_summary, 0, &cam0_img);
          draw_pair(pair_summary, 1, &cam1_img);
          const std::string title =
              "pair=" + std::to_string(pair_summary.pair_index) +
              " rmse=" + std::to_string(pair_summary.overall_rmse) +
              " backend_input_selected";
          const std::string extra =
              "pair_index=" + std::to_string(pair_summary.pair_index) +
              ", rmse=" + std::to_string(pair_summary.overall_rmse);
          DrawStereoVisualizationLegend(&cam0_img, "backend_input_selected", extra);
          DrawStereoVisualizationLegend(&cam1_img, "backend_input_selected", extra);
          cv::putText(cam0_img,
                      "board source: seed green | trial orange | cohesion magenta | unknown gray",
                      cv::Point(20, 155), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                      cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
          cv::putText(cam1_img,
                      "board source: seed green | trial orange | cohesion magenta | unknown gray",
                      cv::Point(20, 155), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                      cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
          cv::putText(cam0_img,
                      "internal fill: green <=3px | yellow >3px | magenta x = backend projection",
                      cv::Point(20, 175), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                      cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
          cv::putText(cam1_img,
                      "internal fill: green <=3px | yellow >3px | magenta x = backend projection",
                      cv::Point(20, 175), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                      cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
          cv::putText(cam0_img, "cam0 " + title, cv::Point(20, 30),
                      cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 255), 2);
          cv::putText(cam1_img, "cam1 " + title, cv::Point(20, 30),
                      cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 255), 2);
          cv::Mat side_by_side;
          cv::hconcat(cam0_img, cam1_img, side_by_side);
          const std::string prefix =
              MakeStereoVisualizationPrefix(index, pair_summary, *pair);
          cv::imwrite((fs::path(subdir) / (prefix + "_side_by_side.png")).string(),
                      side_by_side);
          note << prefix << ": pair_index=" << pair_summary.pair_index
               << ", overall_rmse=" << pair_summary.overall_rmse << "\n";
          ++index;
        }
      };
  write_selected_only(directory);
}

void WriteStereoPairSelectionVisualizations(
    const std::string& directory,
    const StereoExtrinsicCalibrationResult& result,
    int top_k) {
  fs::create_directories(directory);
  std::ofstream summary(
      (fs::path(directory) / "selection_groups_summary.txt").string().c_str());
  std::set<int> seed_pairs;
  std::set<int> attempted_accepted_pairs;
  std::set<int> attempted_rejected_pairs;
  std::map<int, std::string> seed_labels;
  std::map<int, std::string> accepted_labels;
  std::map<int, std::string> rejected_labels;

  for (const StereoPairTrialSelectionDecision& decision :
       result.pair_trial_selection_summary.decisions) {
    if (decision.seed) {
      seed_pairs.insert(decision.pair_index);
      seed_labels[decision.pair_index] = "seed";
    }
    if (decision.attempted && decision.accepted) {
      attempted_accepted_pairs.insert(decision.pair_index);
      accepted_labels[decision.pair_index] = "attempted_accepted";
    }
    if (decision.attempted && !decision.accepted) {
      attempted_rejected_pairs.insert(decision.pair_index);
      rejected_labels[decision.pair_index] =
          decision.reject_reason.empty() ? "attempted_rejected"
                                         : "attempted_rejected_" + decision.reject_reason;
    }
  }

  summary << "seed_pair_count: " << seed_pairs.size() << "\n";
  summary << "attempted_accepted_pair_count: " << attempted_accepted_pairs.size()
          << "\n";
  summary << "attempted_rejected_pair_count: " << attempted_rejected_pairs.size()
          << "\n";

  WriteStereoReprojectionVisualizations((fs::path(directory) / "all_training_overview").string(),
                                        result, top_k);

  auto write_group =
      [&](const std::string& subdir,
          const std::set<int>& pairs,
          const std::map<int, std::string>& labels) {
        if (pairs.empty()) {
          return;
        }
        fs::create_directories(subdir);
        std::ofstream note(
            (fs::path(subdir) / "visualization_summary.txt").string().c_str());
        note << "top_k: " << top_k << "\n";
        note << "pair_count: " << pairs.size() << "\n";
        const DoubleSphereCameraModel cam0 =
            DoubleSphereCameraModel::FromConfig(MakeCameraConfig(result.optimized_scene.cam0));
        const DoubleSphereCameraModel cam1 =
            DoubleSphereCameraModel::FromConfig(MakeCameraConfig(result.optimized_scene.cam1));
        const Eigen::Isometry3d T_cam1_cam0 =
            ToIsometry3d(result.optimized_scene.T_cam1_cam0);
        std::vector<StereoPairResidualSummary> pair_summaries;
        for (const StereoPairResidualSummary& pair_summary :
             result.training_residual_summary.pair_summaries) {
          if (pairs.count(pair_summary.pair_index) > 0) {
            pair_summaries.push_back(pair_summary);
          }
        }
        std::sort(pair_summaries.begin(), pair_summaries.end(),
                  [](const StereoPairResidualSummary& lhs,
                     const StereoPairResidualSummary& rhs) {
                    return lhs.overall_rmse > rhs.overall_rmse;
                  });
        if (top_k > 0 && static_cast<int>(pair_summaries.size()) > top_k) {
          pair_summaries.resize(static_cast<std::size_t>(top_k));
        }

        const auto draw_pair =
            [&](const StereoPairResidualSummary& pair_summary,
                int camera_index,
                cv::Mat* image) {
              const auto pair_pose_it =
                  result.optimized_scene.T_cam0_world_by_pair.find(pair_summary.pair_index);
              if (pair_pose_it == result.optimized_scene.T_cam0_world_by_pair.end()) {
                return;
              }
              const Eigen::Isometry3d T_cam0_world = ToIsometry3d(pair_pose_it->second);
              for (const StereoObservation& observation :
                   result.problem_input.measurement_dataset.observations) {
                if (observation.pair_index != pair_summary.pair_index ||
                    observation.camera_index != camera_index ||
                    !observation.used_in_solver) {
                  continue;
                }
                if (!PairBoardSelected(result.pair_selection_summary,
                                       observation.pair_index,
                                       observation.board_id)) {
                  continue;
                }
                const auto board_pose_it =
                    result.optimized_scene.T_world_board_by_id.find(observation.board_id);
                if (board_pose_it == result.optimized_scene.T_world_board_by_id.end()) {
                  continue;
                }
                const Eigen::Vector4d point_board(observation.target_point_board.x(),
                                                  observation.target_point_board.y(),
                                                  observation.target_point_board.z(),
                                                  1.0);
                const Eigen::Vector4d point_world = board_pose_it->second * point_board;
                const Eigen::Vector3d point_cam0 =
                    (T_cam0_world * point_world).head<3>();
                Eigen::Vector2d predicted = Eigen::Vector2d::Zero();
                bool ok = false;
                if (camera_index == 0) {
                  ok = cam0.vsEuclideanToKeypoint(point_cam0, &predicted);
                } else {
                  ok = cam1.vsEuclideanToKeypoint(T_cam1_cam0 * point_cam0, &predicted);
                }
                if (!ok) {
                  continue;
                }
                const bool is_internal = observation.point_type == JointPointType::Internal;
                const cv::Point observed_pt(
                    static_cast<int>(std::lround(observation.observed_image_xy.x())),
                    static_cast<int>(std::lround(observation.observed_image_xy.y())));
                const cv::Point predicted_pt(
                    static_cast<int>(std::lround(predicted.x())),
                    static_cast<int>(std::lround(predicted.y())));
                DrawStereoResidualObservation(image, observed_pt, predicted_pt,
                                             is_internal);
              }
            };

        int index = 0;
        for (const StereoPairResidualSummary& pair_summary : pair_summaries) {
          const StereoFramePair* pair =
              FindPair(result.problem_input.measurement_dataset, pair_summary.pair_index);
          if (pair == nullptr) {
            continue;
          }
          cv::Mat cam0_img = cv::imread(pair->left_image_path, cv::IMREAD_COLOR);
          cv::Mat cam1_img = cv::imread(pair->right_image_path, cv::IMREAD_COLOR);
          if (cam0_img.empty() || cam1_img.empty()) {
            continue;
          }
          draw_pair(pair_summary, 0, &cam0_img);
          draw_pair(pair_summary, 1, &cam1_img);
          auto label_it = labels.find(pair_summary.pair_index);
          const std::string status =
              label_it == labels.end() ? "group_member" : label_it->second;
          const std::string title =
              "pair=" + std::to_string(pair_summary.pair_index) +
              " rmse=" + std::to_string(pair_summary.overall_rmse) +
              " " + status;
          const std::string extra =
              "pair_index=" + std::to_string(pair_summary.pair_index) +
              ", rmse=" + std::to_string(pair_summary.overall_rmse);
          DrawStereoVisualizationLegend(&cam0_img, status, extra);
          DrawStereoVisualizationLegend(&cam1_img, status, extra);
          cv::putText(cam0_img, "cam0 " + title, cv::Point(20, 30),
                      cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 255), 2);
          cv::putText(cam1_img, "cam1 " + title, cv::Point(20, 30),
                      cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 255), 2);
          cv::Mat side_by_side;
          cv::hconcat(cam0_img, cam1_img, side_by_side);
          const std::string prefix =
              MakeStereoVisualizationPrefix(index, pair_summary, *pair);
          cv::imwrite((fs::path(subdir) / (prefix + "_side_by_side.png")).string(),
                      side_by_side);
          note << prefix << ": pair_index=" << pair_summary.pair_index
               << ", overall_rmse=" << pair_summary.overall_rmse
               << ", status=" << status << "\n";
          ++index;
        }
      };

  write_group((fs::path(directory) / "seed").string(), seed_pairs, seed_labels);
  write_group((fs::path(directory) / "attempted_accepted").string(),
              attempted_accepted_pairs, accepted_labels);
  write_group((fs::path(directory) / "attempted_rejected").string(),
              attempted_rejected_pairs, rejected_labels);
}

void WriteStereoPairBoardSelectionVisualizations(
    const std::string& directory,
    const StereoExtrinsicCalibrationResult& result,
    int top_k) {
  fs::create_directories(directory);
  std::ofstream summary(
      (fs::path(directory) / "pair_board_visualization_summary.txt")
          .string()
          .c_str());
  summary << "enabled: "
          << (result.pair_board_trial_selection_summary.enabled ? 1 : 0)
          << "\n";
  summary << "success: "
          << (result.pair_board_trial_selection_summary.success ? 1 : 0)
          << "\n";
  summary << "top_k_per_group: " << top_k << "\n";
  summary << "note: each image highlights one pair-board decision; rejected "
             "boards may use a local stereo outer-pose refit for visualization "
             "when no final backend pose exists.\n";

  const DoubleSphereCameraModel cam0 =
      DoubleSphereCameraModel::FromConfig(MakeCameraConfig(result.optimized_scene.cam0));
  const DoubleSphereCameraModel cam1 =
      DoubleSphereCameraModel::FromConfig(MakeCameraConfig(result.optimized_scene.cam1));
  const Eigen::Isometry3d T_cam1_cam0 =
      ToIsometry3d(result.optimized_scene.T_cam1_cam0);

  auto group_name_for_decision =
      [](const StereoPairBoardTrialSelectionDecision& decision) {
        if (decision.seed) {
          return std::string("seed_pair_boards");
        }
        if (decision.attempted && decision.accepted) {
          return std::string("attempted_accepted_pair_boards");
        }
        std::string reason = decision.reject_reason.empty()
                                 ? std::string("unknown_reject")
                                 : decision.reject_reason;
        return std::string("rejected_") + SafeVisualizationToken(reason);
      };

  std::map<std::string, std::vector<StereoPairBoardTrialSelectionDecision> >
      decisions_by_group;
  for (const StereoPairBoardTrialSelectionDecision& decision :
       result.pair_board_trial_selection_summary.decisions) {
    decisions_by_group[group_name_for_decision(decision)].push_back(decision);
  }

  for (const auto& group : decisions_by_group) {
    summary << group.first << "_count: " << group.second.size() << "\n";
  }

  auto decision_sort_key =
      [](const StereoPairBoardTrialSelectionDecision& decision) {
        if (decision.accepted || decision.seed) {
          return decision.candidate_score;
        }
        if (std::isfinite(decision.total_rmse_delta)) {
          return decision.total_rmse_delta;
        }
        return decision.candidate_score;
      };

  auto have_pair_pose_or_refit =
      [&](int pair_index, Eigen::Matrix4d* T_cam0_world_matrix,
          std::string* pose_source) {
        const auto pair_pose_it =
            result.optimized_scene.T_cam0_world_by_pair.find(pair_index);
        if (pair_pose_it != result.optimized_scene.T_cam0_world_by_pair.end()) {
          *T_cam0_world_matrix = pair_pose_it->second;
          if (pose_source != nullptr) {
            *pose_source = "optimized_scene";
          }
          return true;
        }
        StereoExtrinsicSolverOptions refit_options;
        refit_options.pair_pose_refit_mode = StereoPairPoseRefitMode::StereoSymmetric;
        refit_options.symmetric_refit_max_iterations = 8;
        refit_options.symmetric_refit_step = 1e-3;
        RuntimeCounters runtime_counters;
        StereoRefitDiagnostics refit_diagnostics;
        if (RefitPairPoseFromStereoOuterObservations(
                result.problem_input.measurement_dataset,
                result.optimized_scene,
                pair_index,
                refit_options,
                &runtime_counters,
                &refit_diagnostics,
                T_cam0_world_matrix)) {
          if (pose_source != nullptr) {
            *pose_source = "local_stereo_refit";
          }
          return true;
        }
        return false;
      };

  auto draw_pair_board =
      [&](const StereoPairBoardTrialSelectionDecision& decision,
          int camera_index,
          const Eigen::Isometry3d& T_cam0_world,
          cv::Mat* image,
          int* drawn_point_count) {
        if (drawn_point_count != nullptr) {
          *drawn_point_count = 0;
        }
        const auto board_pose_it =
            result.optimized_scene.T_world_board_by_id.find(decision.board_id);
        if (board_pose_it == result.optimized_scene.T_world_board_by_id.end()) {
          return;
        }
        for (const StereoObservation& observation :
             result.problem_input.measurement_dataset.observations) {
          if (observation.pair_index != decision.pair_index ||
              observation.camera_index != camera_index ||
              observation.board_id != decision.board_id ||
              !observation.used_in_solver) {
            continue;
          }
          const Eigen::Vector4d point_board(observation.target_point_board.x(),
                                            observation.target_point_board.y(),
                                            observation.target_point_board.z(),
                                            1.0);
          const Eigen::Vector4d point_world =
              board_pose_it->second * point_board;
          const Eigen::Vector3d point_cam0 =
              (T_cam0_world * point_world).head<3>();
          Eigen::Vector2d predicted = Eigen::Vector2d::Zero();
          bool ok = false;
          if (camera_index == 0) {
            ok = cam0.vsEuclideanToKeypoint(point_cam0, &predicted);
          } else {
            ok = cam1.vsEuclideanToKeypoint(T_cam1_cam0 * point_cam0,
                                            &predicted);
          }
          if (!ok) {
            continue;
          }
          const bool is_internal =
              observation.point_type == JointPointType::Internal;
          const cv::Point observed_pt(
              static_cast<int>(std::lround(observation.observed_image_xy.x())),
              static_cast<int>(std::lround(observation.observed_image_xy.y())));
          const cv::Point predicted_pt(
              static_cast<int>(std::lround(predicted.x())),
              static_cast<int>(std::lround(predicted.y())));
          DrawStereoResidualObservation(image, observed_pt, predicted_pt,
                                       is_internal);
          if (drawn_point_count != nullptr) {
            ++(*drawn_point_count);
          }
        }
      };

  for (auto& group : decisions_by_group) {
    std::vector<StereoPairBoardTrialSelectionDecision>& decisions =
        group.second;
    std::sort(decisions.begin(), decisions.end(),
              [&](const StereoPairBoardTrialSelectionDecision& lhs,
                  const StereoPairBoardTrialSelectionDecision& rhs) {
                return decision_sort_key(lhs) > decision_sort_key(rhs);
              });
    if (top_k > 0 && static_cast<int>(decisions.size()) > top_k) {
      decisions.resize(static_cast<std::size_t>(top_k));
    }

    const fs::path group_dir = fs::path(directory) / group.first;
    fs::create_directories(group_dir);
    std::ofstream note((group_dir / "visualization_summary.txt").string().c_str());
    note << "group: " << group.first << "\n";
    note << "visualized_count: " << decisions.size() << "\n";

    int rank = 0;
    for (const StereoPairBoardTrialSelectionDecision& decision : decisions) {
      const StereoFramePair* pair =
          FindPair(result.problem_input.measurement_dataset, decision.pair_index);
      if (pair == nullptr) {
        note << "skip_pair_" << decision.pair_index << "_board_"
             << decision.board_id << ": missing_pair_metadata\n";
        continue;
      }
      cv::Mat cam0_img = cv::imread(pair->left_image_path, cv::IMREAD_COLOR);
      cv::Mat cam1_img = cv::imread(pair->right_image_path, cv::IMREAD_COLOR);
      if (cam0_img.empty() || cam1_img.empty()) {
        note << "skip_pair_" << decision.pair_index << "_board_"
             << decision.board_id << ": image_load_failed\n";
        continue;
      }
      Eigen::Matrix4d T_cam0_world_matrix = Eigen::Matrix4d::Identity();
      std::string pose_source = "none";
      if (!have_pair_pose_or_refit(decision.pair_index,
                                   &T_cam0_world_matrix,
                                   &pose_source)) {
        note << "skip_pair_" << decision.pair_index << "_board_"
             << decision.board_id << ": pose_unavailable\n";
        continue;
      }
      const Eigen::Isometry3d T_cam0_world =
          ToIsometry3d(T_cam0_world_matrix);
      int cam0_drawn = 0;
      int cam1_drawn = 0;
      draw_pair_board(decision, 0, T_cam0_world, &cam0_img, &cam0_drawn);
      draw_pair_board(decision, 1, T_cam0_world, &cam1_img, &cam1_drawn);

      std::string status = group.first;
      const std::string extra =
          "pair=" + std::to_string(decision.pair_index) +
          ", board=" + std::to_string(decision.board_id) +
          ", score=" + std::to_string(decision.candidate_score) +
          ", gain=" + std::to_string(decision.coverage_gain) +
          ", dRMSE=" + std::to_string(decision.total_rmse_delta) +
          ", pose=" + pose_source;
      DrawStereoVisualizationLegend(&cam0_img, status, extra);
      DrawStereoVisualizationLegend(&cam1_img, status, extra);
      cv::putText(cam0_img,
                  "cam0 pair=" + std::to_string(decision.pair_index) +
                      " board=" + std::to_string(decision.board_id),
                  cv::Point(20, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7,
                  cv::Scalar(0, 255, 255), 2);
      cv::putText(cam1_img,
                  "cam1 pair=" + std::to_string(decision.pair_index) +
                      " board=" + std::to_string(decision.board_id),
                  cv::Point(20, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7,
                  cv::Scalar(0, 255, 255), 2);
      cv::Mat side_by_side;
      cv::hconcat(cam0_img, cam1_img, side_by_side);
      std::ostringstream prefix;
      prefix << "rank_" << std::setw(3) << std::setfill('0') << rank
             << "_pair_" << decision.pair_index
             << "_board_" << decision.board_id
             << "_" << SafeVisualizationToken(pair->left_frame_label)
             << "_" << SafeVisualizationToken(pair->right_frame_label);
      cv::imwrite((group_dir / (prefix.str() + "_side_by_side.png")).string(),
                  side_by_side);
      note << prefix.str()
           << ": pair_index=" << decision.pair_index
           << ", board_id=" << decision.board_id
           << ", seed=" << (decision.seed ? 1 : 0)
           << ", attempted=" << (decision.attempted ? 1 : 0)
           << ", accepted=" << (decision.accepted ? 1 : 0)
           << ", reject_reason=" << decision.reject_reason
           << ", candidate_score=" << decision.candidate_score
           << ", coverage_gain=" << decision.coverage_gain
           << ", total_rmse_delta=" << decision.total_rmse_delta
           << ", pose_source=" << pose_source
           << ", cam0_drawn=" << cam0_drawn
           << ", cam1_drawn=" << cam1_drawn << "\n";
      ++rank;
    }
  }
}

void WriteStereoGlobalSparseBaSummary(const std::string& path,
                                      const StereoExtrinsicCalibrationResult& result) {
  std::ofstream output(path.c_str());
  output << "success: " << (result.global_sparse_ba_summary.success ? 1 : 0) << "\n";
  output << "failure_reason: " << result.global_sparse_ba_summary.failure_reason << "\n";
  output << "skip_final_global_ba: "
         << (result.problem_input.solver_options.skip_final_global_ba ? 1 : 0)
         << "\n";
  output << "incremental_batch_acceptance_enabled: "
         << (result.problem_input.solver_options
                     .enable_committing_pair_batch_selection
                 ? 1
                 : 0)
         << "\n";
  output << "persistent_incremental_batch_acceptance_enabled: "
         << (result.problem_input.solver_options
                     .enable_persistent_incremental_stereo_ba
                 ? 1
                 : 0)
         << "\n";
  output << "persistent_incremental_stereo_ba_enabled: "
         << (result.problem_input.solver_options
                     .enable_persistent_incremental_stereo_ba
                 ? 1
                 : 0)
         << "\n";
  output << "final_state_uses_persistent_committed_scene: "
         << (result.problem_input.solver_options.skip_final_global_ba &&
                     result.problem_input.solver_options
                         .enable_persistent_incremental_stereo_ba
                 ? 1
                 : 0)
         << "\n";
  output << "final_state_label: "
         << (result.problem_input.solver_options.skip_final_global_ba
                 ? (result.problem_input.solver_options
                            .enable_persistent_incremental_stereo_ba
                        ? "after_persistent_incremental_committed_scene"
                        : (result.problem_input.solver_options
                                   .enable_committing_pair_batch_selection
                               ? "after_incremental_batch_acceptance"
                               : "after_trial_selection"))
                 : "after_final_global_ba")
         << "\n";
  output << "solver_mode: " << ToString(result.global_sparse_ba_summary.solver_mode)
         << "\n";
  output << "stage6_ba_mode: "
         << result.problem_input.solver_options.ba_mode_label << "\n";
  output << "residual_mode: "
         << ToString(result.global_sparse_ba_summary.residual_mode) << "\n";
  output << "selection_ba_residual_mode: "
         << ToString(result.problem_input.solver_options
                         .selection_ba_residual_mode)
         << "\n";
  output << "optimize_intrinsics: "
         << (result.global_sparse_ba_summary.optimize_intrinsics ? 1 : 0)
         << "\n";
  output << "optimize_stereo_extrinsic: "
         << (result.global_sparse_ba_summary.optimize_stereo_extrinsic ? 1 : 0)
         << "\n";
  output << "optimize_pair_poses: "
         << (result.global_sparse_ba_summary.optimize_pair_poses ? 1 : 0)
         << "\n";
  output << "optimize_board_poses: "
         << (result.global_sparse_ba_summary.optimize_board_poses ? 1 : 0)
         << "\n";
  output << "board_masking_use_local_board_pose_ba: "
         << (result.global_sparse_ba_summary
                     .board_masking_use_local_board_pose_ba
                 ? 1
                 : 0)
         << "\n";
  output << "fixed_intrinsics_for_spherical: "
         << (result.problem_input.solver_options.fixed_intrinsics_for_spherical
                 ? 1
                 : 0)
         << "\n";
  output << "spherical_weight: "
         << result.global_sparse_ba_summary.spherical_weight << "\n";
  output << "spherical_polar_weighting: "
         << (result.global_sparse_ba_summary.spherical_polar_weighting ? 1 : 0)
         << "\n";
  output << "spherical_min_polar_deg: "
         << result.global_sparse_ba_summary.spherical_min_polar_deg << "\n";
  output << "spherical_max_weight: "
         << result.global_sparse_ba_summary.spherical_max_weight << "\n";
  output << "spherical_use_normalize_jacobian: "
         << (result.global_sparse_ba_summary.spherical_use_normalize_jacobian
                 ? 1
                 : 0)
         << "\n";
  output << "spherical_uncertainty_mode: "
         << ToString(result.global_sparse_ba_summary.spherical_uncertainty_mode)
         << "\n";
  output << "stage6_rig_param_mode: "
         << ToString(result.global_sparse_ba_summary.rig_param_mode) << "\n";
  output << "rig_camera_prior_translation_weight: "
         << result.global_sparse_ba_summary.rig_camera_prior_translation_weight
         << "\n";
  output << "rig_camera_prior_rotation_weight: "
         << result.global_sparse_ba_summary.rig_camera_prior_rotation_weight
         << "\n";
  output << "rig_stereo_relative_prior_weight: "
         << result.global_sparse_ba_summary.rig_stereo_relative_prior_weight
         << "\n";
  output << "rig_projection_equivalence_max_pixel_diff: "
         << result.global_sparse_ba_summary
                .rig_projection_equivalence_max_pixel_diff
         << "\n";
  output << "rig_projection_equivalence_max_angular_diff_rad: "
         << result.global_sparse_ba_summary
                .rig_projection_equivalence_max_angular_diff_rad
         << "\n";
  output << "rig_stereo_relative_rotation_drift_deg: "
         << result.global_sparse_ba_summary
                .rig_stereo_relative_rotation_drift_deg
         << "\n";
  output << "rig_stereo_relative_translation_drift_m: "
         << result.global_sparse_ba_summary
                .rig_stereo_relative_translation_drift_m
         << "\n";
  output << "coobs_stereo_factor_count: "
         << result.global_sparse_ba_summary.coobs_stereo_factor_count << "\n";
  output << "coobs_layout_factor_count: "
         << result.global_sparse_ba_summary.coobs_layout_factor_count << "\n";
  output << "coobs_stereo_factor_weight: "
         << result.global_sparse_ba_summary.coobs_stereo_factor_weight << "\n";
  output << "coobs_layout_factor_weight: "
         << result.global_sparse_ba_summary.coobs_layout_factor_weight << "\n";
  output << "coobs_stereo_initial_rot_mean_deg: "
         << result.global_sparse_ba_summary.coobs_stereo_initial_rot_mean_deg
         << "\n";
  output << "coobs_stereo_initial_rot_max_deg: "
         << result.global_sparse_ba_summary.coobs_stereo_initial_rot_max_deg
         << "\n";
  output << "coobs_stereo_initial_trans_mean_m: "
         << result.global_sparse_ba_summary.coobs_stereo_initial_trans_mean_m
         << "\n";
  output << "coobs_stereo_initial_trans_max_m: "
         << result.global_sparse_ba_summary.coobs_stereo_initial_trans_max_m
         << "\n";
  output << "coobs_layout_initial_rot_mean_deg: "
         << result.global_sparse_ba_summary.coobs_layout_initial_rot_mean_deg
         << "\n";
  output << "coobs_layout_initial_rot_max_deg: "
         << result.global_sparse_ba_summary.coobs_layout_initial_rot_max_deg
         << "\n";
  output << "coobs_layout_initial_trans_mean_m: "
         << result.global_sparse_ba_summary.coobs_layout_initial_trans_mean_m
         << "\n";
  output << "coobs_layout_initial_trans_max_m: "
         << result.global_sparse_ba_summary.coobs_layout_initial_trans_max_m
         << "\n";
  output << "spherical_pixel_sigma_px: "
         << result.global_sparse_ba_summary.spherical_pixel_sigma_px << "\n";
  output << "spherical_model_sigma: ";
  for (std::size_t index = 0;
       index < result.global_sparse_ba_summary.spherical_model_sigma.size();
       ++index) {
    if (index > 0) {
      output << ",";
    }
    output << result.global_sparse_ba_summary.spherical_model_sigma[index];
  }
  output << "\n";
  output << "spherical_covariance_damping: "
         << result.global_sparse_ba_summary.spherical_covariance_damping
         << "\n";
  output << "spherical_min_sigma_rad: "
         << result.global_sparse_ba_summary.spherical_min_sigma_rad << "\n";
  output << "spherical_max_whitening_weight: "
         << result.global_sparse_ba_summary.spherical_max_whitening_weight
         << "\n";
  output << "spherical_covariance_valid_count: "
         << result.global_sparse_ba_summary.spherical_covariance_valid_count
         << "\n";
  output << "spherical_covariance_invalid_count: "
         << result.global_sparse_ba_summary.spherical_covariance_invalid_count
         << "\n";
  output << "spherical_covariance_damped_count: "
         << result.global_sparse_ba_summary.spherical_covariance_damped_count
         << "\n";
  output << "spherical_whitening_clamped_count: "
         << result.global_sparse_ba_summary.spherical_whitening_clamped_count
         << "\n";
  output << "spherical_tangent_sigma_mean_rad: "
         << result.global_sparse_ba_summary.spherical_tangent_sigma_mean_rad
         << "\n";
  output << "spherical_tangent_sigma_min_rad: "
         << result.global_sparse_ba_summary.spherical_tangent_sigma_min_rad
         << "\n";
  output << "spherical_tangent_sigma_max_rad: "
         << result.global_sparse_ba_summary.spherical_tangent_sigma_max_rad
         << "\n";
  output << "spherical_whitening_weight_mean: "
         << result.global_sparse_ba_summary.spherical_whitening_weight_mean
         << "\n";
  output << "spherical_whitening_weight_min: "
         << result.global_sparse_ba_summary.spherical_whitening_weight_min
         << "\n";
  output << "spherical_whitening_weight_max: "
         << result.global_sparse_ba_summary.spherical_whitening_weight_max
         << "\n";
  output << "invalid_spherical_unprojection_count: "
         << result.global_sparse_ba_summary.invalid_spherical_unprojection_count
         << "\n";
  output << "eligible_pair_count: "
         << result.global_sparse_ba_summary.eligible_pair_count << "\n";
  output << "selected_pair_count: "
         << result.global_sparse_ba_summary.selected_pair_count << "\n";
  output << "active_board_count: "
         << result.global_sparse_ba_summary.active_board_count << "\n";
  output << "reprojection_error_count: "
         << result.global_sparse_ba_summary.reprojection_error_count << "\n";
  output << "shared_observation_count: "
         << result.global_sparse_ba_summary.shared_observation_count << "\n";
  output << "cam0_only_observation_count: "
         << result.global_sparse_ba_summary.cam0_only_observation_count << "\n";
  output << "cam1_only_observation_count: "
         << result.global_sparse_ba_summary.cam1_only_observation_count << "\n";
  output << "max_iterations: " << result.global_sparse_ba_summary.max_iterations << "\n";
  output << "convergence_threshold: "
         << result.global_sparse_ba_summary.convergence_threshold << "\n";
  output << "shared_observation_weight_scale: "
         << result.global_sparse_ba_summary.shared_observation_weight_scale << "\n";
  output << "single_camera_only_observation_weight_scale: "
         << result.global_sparse_ba_summary.single_camera_only_observation_weight_scale
         << "\n";
  output << "single_camera_only_weight_mode: "
         << ToString(result.global_sparse_ba_summary.single_camera_only_weight_mode)
         << "\n";
  output << "single_camera_only_base_scale: "
         << result.global_sparse_ba_summary.single_camera_only_base_scale << "\n";
  output << "single_camera_only_per_side_budget_ratio: "
         << result.global_sparse_ba_summary.single_camera_only_per_side_budget_ratio
         << "\n";
  output << "shared_total_base_weight: "
         << result.global_sparse_ba_summary.shared_total_base_weight << "\n";
  output << "cam0_only_total_base_weight: "
         << result.global_sparse_ba_summary.cam0_only_total_base_weight << "\n";
  output << "cam1_only_total_base_weight: "
         << result.global_sparse_ba_summary.cam1_only_total_base_weight << "\n";
  output << "per_side_budget_limit: "
         << result.global_sparse_ba_summary.per_side_budget_limit << "\n";
  output << "adaptive_single_camera_only_per_side_cap_ratio: "
         << result.global_sparse_ba_summary
                .adaptive_single_camera_only_per_side_cap_ratio
         << "\n";
  output << "cam0_only_cap: " << result.global_sparse_ba_summary.cam0_only_cap
         << "\n";
  output << "cam1_only_cap: " << result.global_sparse_ba_summary.cam1_only_cap
         << "\n";
  output << "cam0_only_effective_scale: "
         << result.global_sparse_ba_summary.cam0_only_effective_scale << "\n";
  output << "cam1_only_effective_scale: "
         << result.global_sparse_ba_summary.cam1_only_effective_scale << "\n";
  output << "cam0_only_budget_clamped: "
         << (result.global_sparse_ba_summary.cam0_only_budget_clamped ? 1 : 0)
         << "\n";
  output << "cam1_only_budget_clamped: "
         << (result.global_sparse_ba_summary.cam1_only_budget_clamped ? 1 : 0)
         << "\n";
  output << "shared_observation_weight_sum: "
         << result.global_sparse_ba_summary.shared_observation_weight_sum << "\n";
  output << "cam0_only_observation_weight_sum: "
         << result.global_sparse_ba_summary.cam0_only_observation_weight_sum << "\n";
  output << "cam1_only_observation_weight_sum: "
         << result.global_sparse_ba_summary.cam1_only_observation_weight_sum << "\n";
  output << "initial_selected_rmse: "
         << result.global_sparse_ba_summary.initial_selected_rmse << "\n";
  output << "final_selected_rmse: "
         << result.global_sparse_ba_summary.final_selected_rmse << "\n";
  output << "initial_selected_cam0_rmse: "
         << result.global_sparse_ba_summary.initial_selected_cam0_rmse << "\n";
  output << "initial_selected_cam1_rmse: "
         << result.global_sparse_ba_summary.initial_selected_cam1_rmse << "\n";
  output << "final_selected_cam0_rmse: "
         << result.global_sparse_ba_summary.final_selected_cam0_rmse << "\n";
  output << "final_selected_cam1_rmse: "
         << result.global_sparse_ba_summary.final_selected_cam1_rmse << "\n";
  output << "objective_start: " << result.global_sparse_ba_summary.objective_start << "\n";
  output << "objective_final: " << result.global_sparse_ba_summary.objective_final << "\n";
  output << "iterations: " << result.global_sparse_ba_summary.iterations << "\n";
  output << "failed_iterations: "
         << result.global_sparse_ba_summary.failed_iterations << "\n";
  output << "linear_solver_failure: "
         << (result.global_sparse_ba_summary.linear_solver_failure ? 1 : 0) << "\n";
}

void WriteStereoGlobalSparseBaInitialVsFinal(
    const std::string& path,
    const StereoExtrinsicCalibrationResult& result) {
  std::ofstream output(path.c_str());
  output << "metric,initial,final\n";
  output << "selected_total_rmse,"
         << result.global_sparse_ba_summary.initial_selected_rmse << ","
         << result.global_sparse_ba_summary.final_selected_rmse << "\n";
  output << "selected_cam0_rmse,"
         << result.global_sparse_ba_summary.initial_selected_cam0_rmse << ","
         << result.global_sparse_ba_summary.final_selected_cam0_rmse << "\n";
  output << "selected_cam1_rmse,"
         << result.global_sparse_ba_summary.initial_selected_cam1_rmse << ","
         << result.global_sparse_ba_summary.final_selected_cam1_rmse << "\n";
  output << "objective," << result.global_sparse_ba_summary.objective_start << ","
         << result.global_sparse_ba_summary.objective_final << "\n";
}

}  // namespace apriltag_internal
}  // namespace cameras
}  // namespace aslam
