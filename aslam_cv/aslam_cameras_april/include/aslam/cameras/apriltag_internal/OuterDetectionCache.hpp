#ifndef ASLAM_CAMERAS_APRILTAG_INTERNAL_OUTER_DETECTION_CACHE_HPP
#define ASLAM_CAMERAS_APRILTAG_INTERNAL_OUTER_DETECTION_CACHE_HPP

#include <cstdint>
#include <ctime>
#include <string>

#include <aslam/cameras/apriltag_internal/MultiScaleOuterTagDetector.hpp>

namespace aslam {
namespace cameras {
namespace apriltag_internal {

struct OuterDetectionCacheOptions {
  bool enabled = false;
  std::string cache_dir;
};

struct CachedOuterDetectionRecord {
  std::string absolute_image_path;
  std::uintmax_t image_file_size = 0;
  std::time_t image_mtime = 0;
  std::string detector_config_hash;
  OuterTagMultiDetectionResult detection_result;
};

struct OuterDetectionCacheStats {
  int cache_hits = 0;
  int cache_misses = 0;
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

  bool Load(const std::string& image_path,
            OuterTagMultiDetectionResult* detection_result,
            std::string* warning) const;
  bool Save(const std::string& image_path,
            const OuterTagMultiDetectionResult& detection_result,
            std::string* warning) const;

 private:
  MultiScaleOuterTagDetectorConfig config_;
  OuterDetectionCacheOptions options_;
  std::string detector_config_hash_;
};

}  // namespace apriltag_internal
}  // namespace cameras
}  // namespace aslam

#endif  // ASLAM_CAMERAS_APRILTAG_INTERNAL_OUTER_DETECTION_CACHE_HPP
