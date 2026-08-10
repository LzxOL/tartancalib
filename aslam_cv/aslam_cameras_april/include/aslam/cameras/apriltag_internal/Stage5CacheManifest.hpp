#ifndef ASLAM_CAMERAS_APRILTAG_INTERNAL_STAGE5_CACHE_MANIFEST_HPP
#define ASLAM_CAMERAS_APRILTAG_INTERNAL_STAGE5_CACHE_MANIFEST_HPP

#include <string>
#include <vector>

namespace aslam {
namespace cameras {
namespace apriltag_internal {

// Cache artifacts are intentionally versioned per stage.  A change to an
// internal-point or backend stage must not invalidate an image-independent
// outer-detection artifact, while a true pipeline-wide compatibility break can
// still be represented by changing the layout version below.
const char* Stage5CacheLayoutVersion();

enum class Stage5CacheStage {
  ImageMetadata,
  OuterDetectionFinal,
  OuterDecode,
  OuterRefinement,
  OuterRescue,
  InternalSeed,
  InternalRefinement,
  PoseRecovery,
  FrontendMeasurements,
  BackendInput,
  BackendOptimization,
  Diagnostics,
};

const char* ToString(Stage5CacheStage stage);

struct Stage5CacheManifestEntry {
  Stage5CacheStage stage = Stage5CacheStage::Diagnostics;
  // Bump this only when this stage's output semantics change.  It deliberately
  // does not force unrelated cache stages to be rebuilt.
  std::string implementation_version;
  std::string artifact_schema_version;
  std::string semantic_config_hash;
  std::vector<std::string> parent_artifact_hashes;
  std::string semantic_config_description;
};

struct Stage5DatasetCacheIdentity {
  std::string dataset_label;
  // The cache is owned by this exact image directory.  A different directory
  // is a different dataset cache even if its file names happen to overlap.
  std::string absolute_image_root;
};

Stage5DatasetCacheIdentity MakeStage5DatasetCacheIdentity(
    const std::string& image_path);

class Stage5CacheManifest {
 public:
  explicit Stage5CacheManifest(std::string cache_root);

  bool enabled() const;
  const std::string& cache_root() const;

  // Directory reserved for one stage and one semantic configuration.  Existing
  // legacy cache files are never moved or overwritten.
  std::string StageDirectory(Stage5CacheStage stage,
                             const std::string& semantic_config_hash) const;

  // Writes an immutable, human-readable manifest next to artifacts from this
  // stage.  A compatible manifest is reused; a conflicting one is reported as
  // a warning instead of being overwritten.
  bool EnsureStageManifest(const Stage5CacheManifestEntry& entry,
                           std::string* warning) const;

  // Establishes one immutable dataset owner for a cache root.  When adopting
  // a legacy cache without a dataset manifest, all existing outer-cache image
  // records are checked before the ownership file is created.
  bool EnsureDatasetManifest(const Stage5DatasetCacheIdentity& identity,
                             std::string* warning) const;

 private:
  std::string cache_root_;
};

}  // namespace apriltag_internal
}  // namespace cameras
}  // namespace aslam

#endif  // ASLAM_CAMERAS_APRILTAG_INTERNAL_STAGE5_CACHE_MANIFEST_HPP
