#ifndef ASLAM_CAMERAS_APRILTAG_INTERNAL_INTERNAL_REGENERATION_CACHE_HPP
#define ASLAM_CAMERAS_APRILTAG_INTERNAL_INTERNAL_REGENERATION_CACHE_HPP

#include <string>

#include <aslam/cameras/apriltag_internal/MultiBoardInternalMeasurementRegenerator.hpp>
#include <aslam/cameras/apriltag_internal/Stage5CacheManifest.hpp>

namespace aslam {
namespace cameras {
namespace apriltag_internal {

struct InternalRegenerationCacheOptions {
  bool enabled = false;
  std::string cache_dir;
};

struct InternalRegenerationCacheStats {
  int cache_hits = 0;
  int cache_misses = 0;
  int load_failures = 0;
  int store_failures = 0;
};

// Caches the final per-frame internal measurement result produced by the
// existing regeneration pipeline.  The key includes the internal algorithm
// configuration, the exact outer result, and the bootstrap/scene state used
// for pose recovery.  Consequently a changed internal algorithm invalidates
// only this stage, while an unchanged outer result remains reusable.
class InternalRegenerationCache {
 public:
  InternalRegenerationCache(
      ApriltagInternalConfig config,
      ApriltagInternalDetectionOptions detection_options,
      InternalRegenerationCacheOptions options =
          InternalRegenerationCacheOptions{});

  bool enabled() const;
  const std::string& cache_dir() const;
  const std::string& semantic_config_hash() const;

  bool Load(const std::string& image_path,
            const InternalRegenerationFrameInput& frame_input,
            const std::string& state_signature,
            InternalRegenerationFrameResult* frame_result,
            std::string* warning) const;
  bool Save(const std::string& image_path,
            const InternalRegenerationFrameInput& frame_input,
            const std::string& state_signature,
            const InternalRegenerationFrameResult& frame_result,
            std::string* warning) const;

  InternalRegenerationCacheStats stats() const { return stats_; }

  static std::string MakeOuterResultSignature(
      const OuterTagMultiDetectionResult& outer_detection);
  static std::string MakeBootstrapStateSignature(
      const OuterBootstrapResult& bootstrap_result);
  static std::string MakeSceneStateSignature(
      const JointReprojectionSceneState& scene_state);

 private:
  ApriltagInternalConfig config_;
  ApriltagInternalDetectionOptions detection_options_;
  InternalRegenerationCacheOptions options_;
  std::string semantic_config_hash_;
  mutable InternalRegenerationCacheStats stats_;
};

}  // namespace apriltag_internal
}  // namespace cameras
}  // namespace aslam

#endif  // ASLAM_CAMERAS_APRILTAG_INTERNAL_INTERNAL_REGENERATION_CACHE_HPP
