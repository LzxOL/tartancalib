#include <aslam/cameras/apriltag_internal/ApriltagCanonicalModel.hpp>

#include <algorithm>
#include <stdexcept>
#include <vector>

#include "apriltags/TagFamily.h"
#include "apriltags/Tag36h11.h"

namespace aslam {
namespace cameras {
namespace apriltag_internal {
namespace {

std::vector<bool> BuildBlackMatrixTopRowMajor(int tag_id) {
  const auto& codes = AprilTags::tagCodes36h11.codes;
  if (tag_id < 0 || tag_id >= static_cast<int>(codes.size())) {
    throw std::runtime_error("tag_id is out of range for tag family t36h11.");
  }

  const unsigned long long code = codes[static_cast<std::size_t>(tag_id)];
  std::vector<bool> matrix(
      static_cast<std::size_t>(ApriltagCanonicalModel::kCodeDimension *
                               ApriltagCanonicalModel::kCodeDimension),
      false);

  for (int row = 0; row < ApriltagCanonicalModel::kCodeDimension; ++row) {
    for (int col = 0; col < ApriltagCanonicalModel::kCodeDimension; ++col) {
      const bool is_black =
          !static_cast<bool>(code & (1ULL << (ApriltagCanonicalModel::kCodeDimension * row + col)));
      const int flipped_row = ApriltagCanonicalModel::kCodeDimension - 1 - row;
      const int flipped_col = ApriltagCanonicalModel::kCodeDimension - 1 - col;
      matrix[static_cast<std::size_t>(
          flipped_row * ApriltagCanonicalModel::kCodeDimension + flipped_col)] = is_black;
    }
  }

  return matrix;
}

}  // namespace

const char* ToString(CornerType corner_type) {
  switch (corner_type) {
    case CornerType::Outer:
      return "outer";
    case CornerType::LCorner:
      return "l_corner";
    case CornerType::XCorner:
      return "x_corner";
  }
  return "unknown";
}

const char* ToString(InternalProjectionMode mode) {
  switch (mode) {
    case InternalProjectionMode::Homography:
      return "homography";
    case InternalProjectionMode::VirtualPinholePatch:
      return "virtual_pinhole_patch";
  }
  return "unknown";
}

ApriltagCanonicalModel::ApriltagCanonicalModel(ApriltagInternalConfig config)
    : config_(std::move(config)) {
  ValidateConfig();

  module_dimension_ = kCodeDimension + 2 * config_.black_border_bits;
  lattice_dimension_ = module_dimension_ + 1;
  pitch_ = config_.tag_size / static_cast<double>(module_dimension_);

  BuildModuleGrid();
  BuildCornerMetadata();
}

int ApriltagCanonicalModel::PointId(int lattice_u, int lattice_v) const {
  if (lattice_u < 0 || lattice_u >= lattice_dimension_ || lattice_v < 0 || lattice_v >= lattice_dimension_) {
    throw std::runtime_error("PointId requested with out-of-range lattice coordinates.");
  }
  return lattice_v * lattice_dimension_ + lattice_u;
}

bool ApriltagCanonicalModel::IsOuterCorner(int lattice_u, int lattice_v) const {
  return (lattice_u == 0 && lattice_v == 0) ||
         (lattice_u == module_dimension_ && lattice_v == 0) ||
         (lattice_u == module_dimension_ && lattice_v == module_dimension_) ||
         (lattice_u == 0 && lattice_v == module_dimension_);
}

bool ApriltagCanonicalModel::IsModuleBlack(int module_x, int module_y) const {
  if (module_x < 0 || module_x >= module_dimension_ || module_y < 0 || module_y >= module_dimension_) {
    throw std::runtime_error("IsModuleBlack requested with out-of-range module coordinates.");
  }
  return modules_[static_cast<std::size_t>(module_y * module_dimension_ + module_x)];
}

const CanonicalCorner& ApriltagCanonicalModel::corner(int point_id) const {
  if (point_id < 0 || point_id >= static_cast<int>(corners_.size())) {
    throw std::runtime_error("corner requested with out-of-range point_id.");
  }
  return corners_[static_cast<std::size_t>(point_id)];
}

int ApriltagCanonicalModel::ObservablePointCount() const {
  return static_cast<int>(std::count_if(
      corners_.begin(), corners_.end(), [](const CanonicalCorner& corner) { return corner.observable; }));
}

std::vector<int> ApriltagCanonicalModel::VisiblePointIds() const {
  std::vector<int> point_ids;
  point_ids.reserve(corners_.size());
  for (const auto& corner_info : corners_) {
    if (corner_info.observable) {
      point_ids.push_back(corner_info.point_id);
    }
  }
  return point_ids;
}

std::vector<CornerMeasurement> ApriltagCanonicalModel::MakeDefaultMeasurements() const {
  std::vector<CornerMeasurement> measurements(corners_.size());
  for (const auto& corner_info : corners_) {
    CornerMeasurement measurement;
    measurement.board_id = config_.tag_id;
    measurement.point_id = corner_info.point_id;
    measurement.target_xyz = corner_info.target_xyz;
    measurement.corner_type = corner_info.corner_type;
    measurements[static_cast<std::size_t>(corner_info.point_id)] = measurement;
  }
  return measurements;
}

cv::Mat ApriltagCanonicalModel::RenderBinaryPatch(int pixels_per_module) const {
  if (pixels_per_module <= 0) {
    throw std::runtime_error("pixels_per_module must be positive.");
  }

  const int patch_extent = module_dimension_ * pixels_per_module;
  cv::Mat patch(patch_extent + 1, patch_extent + 1, CV_8U, cv::Scalar(255));

  for (int module_y = 0; module_y < module_dimension_; ++module_y) {
    for (int module_x = 0; module_x < module_dimension_; ++module_x) {
      const int patch_x = module_x * pixels_per_module;
      const int patch_y = (module_dimension_ - 1 - module_y) * pixels_per_module;
      const cv::Rect roi(patch_x, patch_y, pixels_per_module, pixels_per_module);
      patch(roi).setTo(IsModuleBlack(module_x, module_y) ? 0 : 255);
    }
  }

  return patch;
}

void ApriltagCanonicalModel::ValidateConfig() const {
  if (config_.target_type != "apriltag_internal") {
    throw std::runtime_error("ApriltagCanonicalModel requires target_type=apriltag_internal.");
  }
  if (config_.tag_family != "t36h11") {
    throw std::runtime_error("ApriltagCanonicalModel currently supports only tagFamily=t36h11.");
  }
  if (config_.black_border_bits != 2) {
    throw std::runtime_error("ApriltagCanonicalModel currently supports only blackBorderBits=2.");
  }
  if (config_.tag_size <= 0.0) {
    throw std::runtime_error("tagSize must be positive.");
  }
  if (config_.min_visible_points <= 0) {
    throw std::runtime_error("minVisiblePoints must be positive.");
  }
}

void ApriltagCanonicalModel::BuildModuleGrid() {
  const std::vector<bool> black_matrix_top = BuildBlackMatrixTopRowMajor(config_.tag_id);
  modules_.assign(static_cast<std::size_t>(module_dimension_ * module_dimension_), false);

  for (int module_y = 0; module_y < module_dimension_; ++module_y) {
    for (int module_x = 0; module_x < module_dimension_; ++module_x) {
      bool is_black = true;
      if (module_x >= config_.black_border_bits &&
          module_x < module_dimension_ - config_.black_border_bits &&
          module_y >= config_.black_border_bits &&
          module_y < module_dimension_ - config_.black_border_bits) {
        const int inner_x = module_x - config_.black_border_bits;
        const int inner_y = module_y - config_.black_border_bits;
        const int row_from_top = kCodeDimension - 1 - inner_y;
        is_black =
            black_matrix_top[static_cast<std::size_t>(row_from_top * kCodeDimension + inner_x)];
      }
      modules_[static_cast<std::size_t>(module_y * module_dimension_ + module_x)] = is_black;
    }
  }
}

void ApriltagCanonicalModel::BuildCornerMetadata() {
  corners_.assign(static_cast<std::size_t>(lattice_dimension_ * lattice_dimension_), CanonicalCorner{});

  for (int lattice_v = 0; lattice_v < lattice_dimension_; ++lattice_v) {
    for (int lattice_u = 0; lattice_u < lattice_dimension_; ++lattice_u) {
      CanonicalCorner corner_info;
      corner_info.point_id = PointId(lattice_u, lattice_v);
      corner_info.lattice_u = lattice_u;
      corner_info.lattice_v = lattice_v;
      corner_info.target_xyz =
          Eigen::Vector3d(lattice_u * pitch_, lattice_v * pitch_, 0.0);

      if (IsOuterCorner(lattice_u, lattice_v)) {
        corner_info.observable = true;
        corner_info.corner_type = CornerType::Outer;
      } else if (lattice_u > 0 && lattice_u < module_dimension_ &&
                 lattice_v > 0 && lattice_v < module_dimension_) {
        const bool m00 = IsModuleBlack(lattice_u - 1, lattice_v - 1);
        const bool m10 = IsModuleBlack(lattice_u, lattice_v - 1);
        const bool m01 = IsModuleBlack(lattice_u - 1, lattice_v);
        const bool m11 = IsModuleBlack(lattice_u, lattice_v);

        corner_info.module_pattern = {m00, m10, m01, m11};

        const bool x_change = (m00 != m10) || (m01 != m11);
        const bool y_change = (m00 != m01) || (m10 != m11);
        if (x_change && y_change) {
          corner_info.observable = true;
          const bool x_corner = (m00 == m11) && (m10 == m01) && (m00 != m10);
          corner_info.corner_type = x_corner ? CornerType::XCorner : CornerType::LCorner;
        }
      }

      corners_[static_cast<std::size_t>(corner_info.point_id)] = corner_info;
    }
  }
}

}  // namespace apriltag_internal
}  // namespace cameras
}  // namespace aslam
