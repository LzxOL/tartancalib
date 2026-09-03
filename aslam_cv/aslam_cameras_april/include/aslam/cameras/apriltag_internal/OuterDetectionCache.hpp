#ifndef ASLAM_CAMERAS_APRILTAG_INTERNAL_OUTER_DETECTION_CACHE_HPP
#define ASLAM_CAMERAS_APRILTAG_INTERNAL_OUTER_DETECTION_CACHE_HPP

#include <cstdint>
#include <ctime>
#include <string>

#include <aslam/cameras/apriltag_internal/MultiScaleOuterTagDetector.hpp>
#include <aslam/cameras/apriltag_internal/Stage5CacheManifest.hpp>

namespace aslam {
namespace cameras {
namespace apriltag_internal {

struct OuterDetectionCacheOptions {
  bool enabled = false;
  std::string cache_dir;
  Stage5CacheStage stage = Stage5CacheStage::OuterDetectionFinal;
  // Optional stage-local key material. Rescue results include the exact
  // provisional-camera signature so they cannot overwrite raw detections.
  std::string cache_key_suffix;
};

struct CachedOuterDetectionRecord {
  std::string absolute_image_path;
  std::uintmax_t image_file_size = 0;
  std::time_t image_mtime = 0;
  std::string detector_config_hash;
  OuterTagMultiDetectionResult detection_result;
};

enum class OuterDetectionCacheLoadSource {
  None = 0,
  StageLayout,
  LegacyLayout,
};

struct OuterDetectionCacheStats {
  int cache_hits = 0;
  int cache_misses = 0;
  int stage_layout_cache_hits = 0;
  int legacy_layout_cache_hits = 0;
  int load_failures = 0;
  int store_failures = 0;
};

class OuterDetectionCache {
 public:
  explicit OuterDetectionCache(
      MultiScaleOuterTagDetectorConfig config,
      OuterDetectionCacheOptions options = OuterDetectionCacheOptions{});

  bool enabled() const;
  const std::string& cache_dir() const;
  const std::string& detector_config_hash() const;

  bool PrepareForDataset(const std::string& image_path,
                         std::string* warning) const;

  bool Load(const std::string& image_path,
            OuterTagMultiDetectionResult* detection_result,
            std::string* warning,
            OuterDetectionCacheLoadSource* load_source = nullptr) const;
  bool Save(const std::string& image_path,
            const OuterTagMultiDetectionResult& detection_result,
            std::string* warning) const;

 private:
  MultiScaleOuterTagDetectorConfig config_;
  OuterDetectionCacheOptions options_;
  std::string detector_config_hash_;
  mutable bool manifests_prepared_ = false;
};

}  // namespace apriltag_internal
}  // namespace cameras
}  // namespace aslam

#endif  // ASLAM_CAMERAS_APRILTAG_INTERNAL_OUTER_DETECTION_CACHE_HPP
