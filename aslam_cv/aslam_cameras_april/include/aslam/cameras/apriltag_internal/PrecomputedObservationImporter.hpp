#ifndef ASLAM_CAMERAS_APRILTAG_INTERNAL_PRECOMPUTED_OBSERVATION_IMPORTER_HPP
#define ASLAM_CAMERAS_APRILTAG_INTERNAL_PRECOMPUTED_OBSERVATION_IMPORTER_HPP

#include <string>

#include <aslam/cameras/apriltag_internal/FrozenRound2BaselinePipeline.hpp>

namespace aslam {
namespace cameras {
namespace apriltag_internal {

class PrecomputedObservationImporter {
 public:
  FrozenPrecomputedMeasurementInput Load(const std::string& directory,
                                         int frame_index_offset = 0,
                                         const std::string& target_mode = "auto") const;
};

}  // namespace apriltag_internal
}  // namespace cameras
}  // namespace aslam

#endif  // ASLAM_CAMERAS_APRILTAG_INTERNAL_PRECOMPUTED_OBSERVATION_IMPORTER_HPP
