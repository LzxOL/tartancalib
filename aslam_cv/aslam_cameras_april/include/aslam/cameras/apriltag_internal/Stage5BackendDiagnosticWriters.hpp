#ifndef ASLAM_CAMERAS_APRILTAG_INTERNAL_STAGE5_BACKEND_DIAGNOSTIC_WRITERS_HPP_
#define ASLAM_CAMERAS_APRILTAG_INTERNAL_STAGE5_BACKEND_DIAGNOSTIC_WRITERS_HPP_

#include <aslam/cameras/apriltag_internal/AslamBackendCalibrationRunner.hpp>
#include <aslam/cameras/apriltag_internal/FrozenRound2BaselinePipeline.hpp>
#include <aslam/cameras/apriltag_internal/Stage5Benchmark.hpp>

#include <boost/filesystem.hpp>

#include <vector>

namespace aslam {
namespace cameras {
namespace apriltag_internal {

void WriteInternalRegenerationDiagnostics(
    const boost::filesystem::path& output_dir,
    const Stage5BenchmarkReport& report);

void WriteInternalSeedStepOverlays(
    const boost::filesystem::path& output_dir,
    const Stage5BenchmarkReport& report,
    const std::vector<FrozenRound2BaselineFrameSource>& all_frames_for_lookup);

void WriteGeometryPriorOuterSeedDiagnostics(
    const boost::filesystem::path& output_dir,
    const Stage5BenchmarkReport& report);

void WriteIntermediateFrontendRegenerationSummary(
    const boost::filesystem::path& output_dir,
    const Stage5BenchmarkReport& report);

void WriteFrameBoardObservationFlowDiagnostics(
    const boost::filesystem::path& output_dir,
    const Stage5BenchmarkReport& report,
    const std::vector<FrozenRound2BaselineFrameSource>& all_frames_for_lookup,
    const CameraModelRefitEvaluationResult* backend_training_evaluation = nullptr,
    const JointResidualEvaluationResult* backend_optimized_residual = nullptr);

void WriteTrialBackendFrameBoardSelectionDiagnostics(
    const boost::filesystem::path& output_dir,
    const Stage5BenchmarkReport& report);

void WriteGlobalSceneStateConsistencyAudit(
    const boost::filesystem::path& output_dir,
    const Stage5BenchmarkReport& report);

}  // namespace apriltag_internal
}  // namespace cameras
}  // namespace aslam

#endif  // ASLAM_CAMERAS_APRILTAG_INTERNAL_STAGE5_BACKEND_DIAGNOSTIC_WRITERS_HPP_
