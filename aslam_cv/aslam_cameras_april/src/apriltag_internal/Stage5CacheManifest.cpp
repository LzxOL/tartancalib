#include <aslam/cameras/apriltag_internal/Stage5CacheManifest.hpp>

#include <cctype>
#include <fstream>
#include <sstream>
#include <utility>
#include <vector>

#include <boost/filesystem.hpp>

namespace aslam {
namespace cameras {
namespace apriltag_internal {
namespace {

namespace fs = boost::filesystem;

constexpr const char kStage5CacheLayoutVersion[] = "stage5_cache_layout_v1";

std::string EscapeYamlScalar(const std::string& value) {
  std::string escaped;
  escaped.reserve(value.size() + 2);
  escaped.push_back('\'');
  for (char character : value) {
    if (character == '\'') {
      escaped += "''";
    } else {
      escaped.push_back(character);
    }
  }
  escaped.push_back('\'');
  return escaped;
}

std::string BuildManifestText(const Stage5CacheManifestEntry& entry) {
  std::ostringstream output;
  output << "cache_layout_version: " << kStage5CacheLayoutVersion << "\n";
  output << "stage: " << ToString(entry.stage) << "\n";
  output << "implementation_version: "
         << EscapeYamlScalar(entry.implementation_version) << "\n";
  output << "artifact_schema_version: "
         << EscapeYamlScalar(entry.artifact_schema_version) << "\n";
  output << "semantic_config_hash: "
         << EscapeYamlScalar(entry.semantic_config_hash) << "\n";
  output << "parent_artifact_hashes: [";
  for (std::size_t index = 0; index < entry.parent_artifact_hashes.size(); ++index) {
    if (index > 0) {
      output << ", ";
    }
    output << EscapeYamlScalar(entry.parent_artifact_hashes[index]);
  }
  output << "]\n";
  output << "semantic_config_description: "
         << EscapeYamlScalar(entry.semantic_config_description) << "\n";
  return output.str();
}

std::string Trim(const std::string& value) {
  const std::string whitespace = " \t\r\n";
  const std::size_t begin = value.find_first_not_of(whitespace);
  if (begin == std::string::npos) {
    return std::string();
  }
  const std::size_t end = value.find_last_not_of(whitespace);
  return value.substr(begin, end - begin + 1);
}

std::string UnquoteYamlScalar(const std::string& value) {
  std::string result = Trim(value);
  if (result.size() >= 2 &&
      ((result.front() == '\"' && result.back() == '\"') ||
       (result.front() == '\'' && result.back() == '\''))) {
    result = result.substr(1, result.size() - 2);
  }
  return result;
}

bool ReadYamlScalar(const fs::path& path,
                    const std::string& key,
                    std::string* value) {
  if (value == nullptr) {
    return false;
  }
  std::ifstream input(path.string().c_str());
  if (!input.is_open()) {
    return false;
  }
  const std::string prefix = key + ":";
  std::string line;
  while (std::getline(input, line)) {
    if (line.compare(0, prefix.size(), prefix) == 0) {
      *value = UnquoteYamlScalar(line.substr(prefix.size()));
      return true;
    }
  }
  return false;
}

std::string DatasetManifestText(const Stage5DatasetCacheIdentity& identity) {
  std::ostringstream output;
  output << "cache_layout_version: " << kStage5CacheLayoutVersion << "\n";
  output << "manifest_kind: dataset_owner\n";
  output << "dataset_label: " << EscapeYamlScalar(identity.dataset_label)
         << "\n";
  output << "absolute_image_root: "
         << EscapeYamlScalar(identity.absolute_image_root) << "\n";
  return output.str();
}

bool WriteImmutableTextFile(const fs::path& path,
                            const std::string& text,
                            std::string* warning) {
  if (fs::exists(path)) {
    return true;
  }
  fs::create_directories(path.parent_path());
  const fs::path temporary_path = path.string() + ".tmp";
  {
    std::ofstream output(temporary_path.string().c_str(), std::ios::trunc);
    if (!output.is_open()) {
      if (warning != nullptr) {
        *warning = "Failed to write cache manifest: " +
                   temporary_path.string();
      }
      return false;
    }
    output << text;
  }
  if (fs::exists(path)) {
    fs::remove(temporary_path);
  } else {
    fs::rename(temporary_path, path);
  }
  return true;
}

std::string MakeDatasetLabel(const fs::path& image_root) {
  std::vector<std::string> components;
  bool collect = false;
  for (fs::path::const_iterator it = image_root.begin(); it != image_root.end();
       ++it) {
    const std::string component = it->string();
    if (component.compare(0, 8, "dataset_") == 0) {
      collect = true;
    }
    if (collect && component != "images" && !component.empty() &&
        component != "/") {
      components.push_back(component);
    }
  }
  if (components.empty()) {
    return image_root.filename().string();
  }
  std::ostringstream label;
  for (std::size_t index = 0; index < components.size(); ++index) {
    if (index > 0) {
      label << "_";
    }
    for (char character : components[index]) {
      label << (std::isalnum(static_cast<unsigned char>(character)) ?
                    character :
                    '_');
    }
  }
  return label.str();
}

}  // namespace

const char* Stage5CacheLayoutVersion() {
  return kStage5CacheLayoutVersion;
}

const char* ToString(Stage5CacheStage stage) {
  switch (stage) {
    case Stage5CacheStage::ImageMetadata:
      return "image_metadata";
    case Stage5CacheStage::OuterDetectionFinal:
      return "outer_detection_final";
    case Stage5CacheStage::OuterDecode:
      return "outer_decode";
    case Stage5CacheStage::OuterRefinement:
      return "outer_refinement";
    case Stage5CacheStage::OuterRescue:
      return "outer_rescue";
    case Stage5CacheStage::InternalSeed:
      return "internal_seed";
    case Stage5CacheStage::InternalRefinement:
      return "internal_refinement";
    case Stage5CacheStage::PoseRecovery:
      return "pose_recovery";
    case Stage5CacheStage::FrontendMeasurements:
      return "frontend_measurements";
    case Stage5CacheStage::BackendInput:
      return "backend_input";
    case Stage5CacheStage::BackendOptimization:
      return "backend_optimization";
    case Stage5CacheStage::Diagnostics:
      return "diagnostics";
  }
  return "unknown";
}

Stage5DatasetCacheIdentity MakeStage5DatasetCacheIdentity(
    const std::string& image_path) {
  Stage5DatasetCacheIdentity identity;
  if (image_path.empty()) {
    return identity;
  }
  fs::path root = fs::absolute(fs::path(image_path)).lexically_normal();
  if (fs::is_regular_file(root)) {
    root = root.parent_path();
  }
  identity.absolute_image_root = root.string();
  identity.dataset_label = MakeDatasetLabel(root);
  return identity;
}

Stage5CacheManifest::Stage5CacheManifest(std::string cache_root)
    : cache_root_(std::move(cache_root)) {}

bool Stage5CacheManifest::enabled() const {
  return !cache_root_.empty();
}

const std::string& Stage5CacheManifest::cache_root() const {
  return cache_root_;
}

std::string Stage5CacheManifest::StageDirectory(
    Stage5CacheStage stage,
    const std::string& semantic_config_hash) const {
  if (!enabled() || semantic_config_hash.empty()) {
    return std::string();
  }
  return (fs::path(cache_root_) / kStage5CacheLayoutVersion /
          ToString(stage) / semantic_config_hash)
      .string();
}

bool Stage5CacheManifest::EnsureStageManifest(
    const Stage5CacheManifestEntry& entry,
    std::string* warning) const {
  if (warning != nullptr) {
    warning->clear();
  }
  if (!enabled() || entry.semantic_config_hash.empty()) {
    return true;
  }
  try {
    const fs::path stage_directory(
        StageDirectory(entry.stage, entry.semantic_config_hash));
    const fs::path manifest_path = stage_directory / "manifest.yaml";
    const std::string expected = BuildManifestText(entry);
    if (fs::exists(manifest_path)) {
      std::ifstream input(manifest_path.string().c_str());
      std::ostringstream existing;
      existing << input.rdbuf();
      if (existing.str() != expected) {
        if (warning != nullptr) {
          *warning = "Cache manifest conflict; preserving existing manifest: " +
                     manifest_path.string();
        }
        return false;
      }
      return true;
    }

    fs::create_directories(stage_directory);
    const fs::path temporary_path = manifest_path.string() + ".tmp";
    {
      std::ofstream output(temporary_path.string().c_str(), std::ios::trunc);
      if (!output.is_open()) {
        if (warning != nullptr) {
          *warning = "Failed to write cache manifest: " +
                     temporary_path.string();
        }
        return false;
      }
      output << expected;
    }
    if (!fs::exists(manifest_path)) {
      fs::rename(temporary_path, manifest_path);
    } else {
      fs::remove(temporary_path);
    }
    return true;
  } catch (const std::exception& error) {
    if (warning != nullptr) {
      *warning = error.what();
    }
    return false;
  }
}

bool Stage5CacheManifest::EnsureDatasetManifest(
    const Stage5DatasetCacheIdentity& identity,
    std::string* warning) const {
  if (warning != nullptr) {
    warning->clear();
  }
  if (!enabled() || identity.absolute_image_root.empty()) {
    return true;
  }
  try {
    const fs::path root(cache_root_);
    const fs::path dataset_manifest_path = root / "dataset_manifest.yaml";
    if (fs::exists(dataset_manifest_path)) {
      std::string existing_root;
      if (!ReadYamlScalar(dataset_manifest_path, "absolute_image_root",
                          &existing_root) ||
          existing_root != identity.absolute_image_root) {
        if (warning != nullptr) {
          *warning = "Cache root belongs to a different dataset: " +
                     dataset_manifest_path.string();
        }
        return false;
      }
      return true;
    }

    // Legacy caches predate dataset_manifest.yaml.  Adopt one only if every
    // cached image record agrees with the requested image directory.
    if (fs::exists(root)) {
      fs::recursive_directory_iterator end;
      for (fs::recursive_directory_iterator it(root); it != end; ++it) {
        if (!fs::is_regular_file(*it) || it->path().extension() != ".yml") {
          continue;
        }
        std::string cached_image_path;
        if (!ReadYamlScalar(it->path(), "absolute_image_path",
                            &cached_image_path) ||
            cached_image_path.empty()) {
          continue;
        }
        const fs::path cached_parent =
            fs::absolute(fs::path(cached_image_path)).lexically_normal()
                .parent_path();
        if (cached_parent.string() != identity.absolute_image_root) {
          if (warning != nullptr) {
            *warning = "Refusing to adopt mixed-dataset cache root; found " +
                       cached_parent.string() + " in " + it->path().string();
          }
          return false;
        }
      }
    }
    return WriteImmutableTextFile(dataset_manifest_path,
                                  DatasetManifestText(identity), warning);
  } catch (const std::exception& error) {
    if (warning != nullptr) {
      *warning = error.what();
    }
    return false;
  }
}

}  // namespace apriltag_internal
}  // namespace cameras
}  // namespace aslam
