#ifndef ASLAM_CAMERAS_APRILTAG_INTERNAL_SYMMETRIC_STEREO_REPROJECTION_ERROR_HPP
#define ASLAM_CAMERAS_APRILTAG_INTERNAL_SYMMETRIC_STEREO_REPROJECTION_ERROR_HPP

#include <array>

// Homebrew glog requires consumers to opt into the generated export header
// before including Ceres headers. Keeping this local avoids changing global
// build flags for the current aslam backend.
#ifndef GLOG_USE_GLOG_EXPORT
#define GLOG_USE_GLOG_EXPORT
#endif

#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <Eigen/Core>

namespace aslam {
namespace cameras {
namespace apriltag_internal {

// Convention used by this prototype:
//   T_A_B maps a point from frame B to frame A: P_A = T_A_B * P_B.
//
// This matches the Stage6 cam0-reference code path inspected in
// StereoExtrinsicCalibrationRunner.cpp:
//   point_cam0 = T_cam0_board * point_board
//   point_cam1 = T_cam1_cam0 * point_cam0
//
// The symmetric prototype below replaces the asymmetric cam0-reference path
// with:
//   point_rig = T_rig_board * point_board
//   point_cam0 = T_cam0_rig * point_rig
//   point_cam1 = T_cam1_rig * point_rig
// where cam0/cam1 are both derived from one shared stereo spread variable.

constexpr int kSymmetricStereoPinholeIntrinsics = 4;

template <typename T>
inline void ApplyAngleAxisPose(const T* const pose6,
                               const T* const point,
                               T* const out) {
  ceres::AngleAxisRotatePoint(pose6, point, out);
  out[0] += pose6[3];
  out[1] += pose6[4];
  out[2] += pose6[5];
}

template <typename T>
inline bool ProjectPointPinholePlaceholder(const T* const point_camera,
                                           const T* const intrinsics,
                                           T* const u,
                                           T* const v) {
  // Placeholder only. Intrinsics are interpreted as [fx, fy, cx, cy].
  // Replace this with the project's DS/KB projection model before using this
  // functor for real calibration experiments.
  const T z = point_camera[2];
  const T inv_z = T(1) / z;
  *u = intrinsics[0] * point_camera[0] * inv_z + intrinsics[2];
  *v = intrinsics[1] * point_camera[1] * inv_z + intrinsics[3];
  return true;
}

struct SymmetricStereoReprojectionError {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  SymmetricStereoReprojectionError(int camera_index,
                                   double observed_u,
                                   double observed_v,
                                   const Eigen::Vector3d& point_3d)
      : camera_index_(camera_index),
        observed_u_(observed_u),
        observed_v_(observed_v),
        point_3d_{{point_3d.x(), point_3d.y(), point_3d.z()}} {}

  template <typename T>
  bool operator()(const T* const frame_pose,
                  const T* const ext_spread,
                  const T* const intrinsics,
                  T* const residuals) const {
    T point_board[3] = {
        T(point_3d_[0]),
        T(point_3d_[1]),
        T(point_3d_[2]),
    };
    return EvaluateWithPoint(frame_pose, ext_spread, intrinsics, point_board,
                             residuals);
  }

  template <typename T>
  bool operator()(const T* const frame_pose,
                  const T* const ext_spread,
                  const T* const point_3d,
                  const T* const intrinsics,
                  T* const residuals) const {
    return EvaluateWithPoint(frame_pose, ext_spread, intrinsics, point_3d,
                             residuals);
  }

 private:
  template <typename T>
  bool EvaluateWithPoint(const T* const frame_pose,
                         const T* const ext_spread,
                         const T* const intrinsics,
                         const T* const point_board,
                         T* const residuals) const {
    T point_rig[3];
    ApplyAngleAxisPose(frame_pose, point_board, point_rig);

    // The ideal formulation should use an AD-safe SE(3) exponential map:
    //   T_cam0_rig = Exp(-0.5 * xi_spread)
    //   T_cam1_rig = Exp(+0.5 * xi_spread)
    //
    // This prototype intentionally uses the approximate split requested in the
    // prompt because this codebase does not currently expose a Ceres-ready SE(3)
    // Exp for this standalone functor. Scaling angle-axis and translation by
    // +/-0.5 is not an exact SE(3) midpoint. It must be validated with
    // projection-equivalence tests before this functor is used for conclusions.
    const T sign =
        camera_index_ == 0 ? T(-0.5) : T(0.5);
    T half_extrinsic[6] = {
        sign * ext_spread[0],
        sign * ext_spread[1],
        sign * ext_spread[2],
        sign * ext_spread[3],
        sign * ext_spread[4],
        sign * ext_spread[5],
    };

    T point_camera[3];
    ApplyAngleAxisPose(half_extrinsic, point_rig, point_camera);

    T predicted_u = T(0);
    T predicted_v = T(0);
    ProjectPointPinholePlaceholder(point_camera, intrinsics, &predicted_u,
                                   &predicted_v);

    residuals[0] = predicted_u - T(observed_u_);
    residuals[1] = predicted_v - T(observed_v_);

    // TODO(angular): add a 2D tangent-plane angular residual variant here:
    //   observed pixel -> DS/KB unproject -> b_obs
    //   normalize(point_camera) -> b_pred
    //   residual = E(b_obs)^T * (b_pred - b_obs)
    return true;
  }

  int camera_index_ = 0;
  double observed_u_ = 0.0;
  double observed_v_ = 0.0;
  std::array<double, 3> point_3d_;
};

inline ceres::CostFunction* CreateSymmetricStereoReprojectionCost(
    int camera_index,
    double observed_u,
    double observed_v,
    const Eigen::Vector3d& point_3d) {
  // Parameter blocks:
  //   frame_pose:  6D virtual rig pose T_rig_board
  //   ext_spread:  6D symmetric stereo spread
  //   intrinsics:  4D placeholder pinhole intrinsics [fx, fy, cx, cy]
  //
  // Replace kSymmetricStereoPinholeIntrinsics when the placeholder projection is
  // replaced by the project camera model.
  return new ceres::AutoDiffCostFunction<
      SymmetricStereoReprojectionError,
      2,
      6,
      6,
      kSymmetricStereoPinholeIntrinsics>(
      new SymmetricStereoReprojectionError(camera_index, observed_u, observed_v,
                                           point_3d));
}

inline ceres::CostFunction* CreateSymmetricStereoReprojectionCostWithLandmark(
    int camera_index,
    double observed_u,
    double observed_v) {
  // Alternative optimized-landmark parameterization:
  //   frame_pose:  6D virtual rig pose T_rig_board
  //   ext_spread:  6D symmetric stereo spread
  //   point_3d:    3D board/landmark point
  //   intrinsics:  4D placeholder pinhole intrinsics [fx, fy, cx, cy]
  return new ceres::AutoDiffCostFunction<
      SymmetricStereoReprojectionError,
      2,
      6,
      6,
      3,
      kSymmetricStereoPinholeIntrinsics>(
      new SymmetricStereoReprojectionError(camera_index, observed_u, observed_v,
                                           Eigen::Vector3d::Zero()));
}

// Validation plan before using this in Stage6:
// 1. Convert the old cam0-reference calibration to a rig-centric initial state.
// 2. For several frames, boards, cameras, and corners, evaluate both paths:
//      old: P_cam0 = T_cam0_board * P_board
//           P_cam1 = T_cam1_cam0 * P_cam0
//      new: P_rig  = T_rig_board * P_board
//           P_camk = T_camk_rig * P_rig
// 3. Compare projected pixels before optimization. With an exact SE(3)
//    conversion the maximum difference should be near numerical precision.
//    With this approximate split, report the measured difference explicitly.
//
// Future replacement:
//   Replace the approximate half split with an AD-safe SE(3) Exp and use the
//   se(3) left Jacobian for the translational component instead of scaling raw
//   translation by +/-0.5.

}  // namespace apriltag_internal
}  // namespace cameras
}  // namespace aslam

#endif  // ASLAM_CAMERAS_APRILTAG_INTERNAL_SYMMETRIC_STEREO_REPROJECTION_ERROR_HPP
