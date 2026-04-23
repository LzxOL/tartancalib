#ifndef ASLAM_CAMERAS_APRILTAG_INTERNAL_DEBUG_VISUALIZATION_HPP
#define ASLAM_CAMERAS_APRILTAG_INTERNAL_DEBUG_VISUALIZATION_HPP

#include <string>

#include <opencv2/core.hpp>

#include <aslam/cameras/apriltag_internal/ApriltagInternalDetector.hpp>

namespace aslam {
namespace cameras {
namespace apriltag_internal {

bool UsesSphereSeedPipeline(InternalProjectionMode mode);

cv::Mat BuildInternalSeedOverlay(const cv::Mat& image,
                                 const ApriltagInternalDetectionResult& result);

cv::Mat BuildInternalSphereDebugView(const ApriltagInternalDetectionResult& result);

cv::Mat BuildInternalMethodSpaceDebugView(const ApriltagInternalDetectionResult& result);

cv::Mat BuildSideBySideComparisonCanvas(const cv::Mat& left,
                                        const cv::Mat& right,
                                        const std::string& left_title,
                                        const std::string& right_title,
                                        const std::string& title,
                                        const std::string& subtitle);

}  // namespace apriltag_internal
}  // namespace cameras
}  // namespace aslam

#endif  // ASLAM_CAMERAS_APRILTAG_INTERNAL_DEBUG_VISUALIZATION_HPP
