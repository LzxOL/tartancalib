#!/usr/bin/env python3
"""Pose-only camera evaluation for fixed intrinsics and multi-board layout.

For each test frame this script fixes the camera model and T_reference_board
layout, optimizes only T_camera_reference, and reports reprojection metrics.
It also evaluates cross-board consistency by fitting the pose from one visible
board and projecting all other visible boards.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np


@dataclass
class Camera:
    label: str
    family: str
    intrinsics: List[float]
    distortion: List[float]


@dataclass
class PointObs:
    frame_index: int
    frame_label: str
    board_id: int
    point_id: int
    point_type: str
    observed: np.ndarray
    p_board: np.ndarray
    p_ref: np.ndarray


@dataclass
class PoseFit:
    success: bool
    params: Optional[np.ndarray]
    errors: np.ndarray
    projection_failures: int
    iterations: int
    failure_reason: str = ""


def parse_key_value_file(path: Path) -> Dict[str, str]:
    result: Dict[str, str] = {}
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            result[key.strip()] = value.strip()
    return result


def parse_float_list(text: str) -> List[float]:
    if not text:
        return []
    return [float(x) for x in text.split(",") if x.strip()]


def camera_from_summary(summary: Dict[str, str], label: str, prefix: str) -> Camera:
    family = summary.get(f"{prefix}_camera_model_family", "")
    intrinsics = parse_float_list(summary.get(f"{prefix}_camera_intrinsics_csv", ""))
    distortion = parse_float_list(summary.get(f"{prefix}_camera_distortion_csv", ""))
    if family != "pinhole-equi":
        raise ValueError(
            f"{label}: only pinhole-equi/KB is implemented for pose evaluation; "
            f"got family={family!r}"
        )
    if len(intrinsics) != 4 or len(distortion) != 4:
        raise ValueError(
            f"{label}: expected fu,fv,cu,cv and k1..k4, got "
            f"intrinsics={intrinsics}, distortion={distortion}"
        )
    return Camera(label=label, family=family, intrinsics=intrinsics, distortion=distortion)


def camera_from_yaml(path: Path, label: str) -> Camera:
    """Read the narrow, canonical cam0 YAML subset used by this evaluator."""
    text = path.read_text(encoding="utf-8")

    def scalar(name: str) -> str:
        match = re.search(rf"^\s*{re.escape(name)}:\s*(\S+)\s*$", text, re.MULTILINE)
        if not match:
            raise ValueError(f"{path}: missing {name!r}")
        return match.group(1)

    def values(name: str) -> List[float]:
        match = re.search(rf"^\s*{re.escape(name)}:\s*(\[[^\n]+\])\s*$", text, re.MULTILINE)
        if not match:
            raise ValueError(f"{path}: missing {name!r} list")
        parsed = ast.literal_eval(match.group(1))
        return [float(value) for value in parsed]

    if scalar("camera_model") != "pinhole" or scalar("distortion_model") != "equidistant":
        raise ValueError(f"{path}: only pinhole/equidistant KB YAMLs are supported")
    intrinsics = values("intrinsics")
    distortion = values("distortion_coeffs")
    if len(intrinsics) != 4 or len(distortion) != 4:
        raise ValueError(f"{path}: expected four intrinsics and four KB coefficients")
    return Camera(label=label, family="pinhole-equi", intrinsics=intrinsics, distortion=distortion)


def load_board_layout(path: Path) -> Dict[int, np.ndarray]:
    boards: Dict[int, np.ndarray] = {}
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("initialized", "1") not in ("1", "true", "True"):
                continue
            board_id = int(row["board_id"])
            matrix_text = None
            for key in ("T_reference_board_16", "T_world_board_16", "Rt_16", "T_16"):
                if key in row:
                    matrix_text = row[key]
                    break
            if matrix_text is None:
                raise ValueError(f"{path} has no T_reference_board_16-like column")
            values = parse_float_list(matrix_text)
            if len(values) == 1 and row.get(None):
                # Stage5 writes the 16 transform values as unquoted CSV tail
                # fields after the T_reference_board_16 header.
                values.extend(float(x) for x in row[None] if str(x).strip())
            if len(values) != 16:
                raise ValueError(f"board {board_id}: expected 16 transform values, got {len(values)}")
            boards[board_id] = np.array(values, dtype=np.float64).reshape(4, 4)
    if not boards:
        raise ValueError(f"No initialized board layouts found in {path}")
    return boards


def load_points(path: Path, method: str, boards: Dict[int, np.ndarray]) -> Dict[int, List[PointObs]]:
    frames: Dict[int, List[PointObs]] = {}
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("method", method) != method:
                continue
            if row.get("evaluation_included", "1") not in ("1", "true", "True"):
                continue
            board_id = int(row["board_id"])
            if board_id not in boards:
                continue
            p_board = np.array(
                [float(row["target_x"]), float(row["target_y"]), float(row["target_z"])],
                dtype=np.float64,
            )
            p_h = np.array([p_board[0], p_board[1], p_board[2], 1.0], dtype=np.float64)
            p_ref = (boards[board_id] @ p_h)[:3]
            obs = PointObs(
                frame_index=int(row["frame_index"]),
                frame_label=row.get("frame_label", ""),
                board_id=board_id,
                point_id=int(float(row.get("point_id", 0))),
                point_type=row.get("point_type", ""),
                observed=np.array([float(row["observed_x"]), float(row["observed_y"])], dtype=np.float64),
                p_board=p_board,
                p_ref=p_ref,
            )
            frames.setdefault(obs.frame_index, []).append(obs)
    return frames


def rodrigues(rvec: np.ndarray) -> np.ndarray:
    R, _ = cv2.Rodrigues(rvec.reshape(3, 1))
    return R.astype(np.float64)


def project_kb(points_ref: np.ndarray, params: np.ndarray, camera: Camera) -> Tuple[np.ndarray, np.ndarray]:
    rvec = params[:3]
    t = params[3:6]
    R = rodrigues(rvec)
    pc = (R @ points_ref.T).T + t.reshape(1, 3)
    z = pc[:, 2]
    valid = np.isfinite(pc).all(axis=1) & (z > 1e-9)
    proj = np.full((points_ref.shape[0], 2), np.nan, dtype=np.float64)
    if not valid.any():
        return proj, valid
    x = pc[valid, 0] / pc[valid, 2]
    y = pc[valid, 1] / pc[valid, 2]
    r = np.sqrt(x * x + y * y)
    theta = np.arctan(r)
    k1, k2, k3, k4 = camera.distortion
    theta2 = theta * theta
    theta_d = theta * (
        1.0
        + k1 * theta2
        + k2 * theta2 * theta2
        + k3 * theta2 * theta2 * theta2
        + k4 * theta2 * theta2 * theta2 * theta2
    )
    scale = np.ones_like(r)
    nonzero = r > 1e-12
    scale[nonzero] = theta_d[nonzero] / r[nonzero]
    fu, fv, cu, cv = camera.intrinsics
    proj_valid = np.column_stack((fu * x * scale + cu, fv * y * scale + cv))
    finite = np.isfinite(proj_valid).all(axis=1)
    valid_indices = np.flatnonzero(valid)
    proj[valid_indices[finite]] = proj_valid[finite]
    valid[valid_indices[~finite]] = False
    return proj, valid


def residual_vector(points_ref: np.ndarray, observed: np.ndarray, params: np.ndarray, camera: Camera) -> Tuple[np.ndarray, int]:
    proj, valid = project_kb(points_ref, params, camera)
    residual = proj - observed
    failures = int((~valid).sum())
    residual[~valid, :] = 1e6
    return residual.reshape(-1), failures


def initial_pose_candidates(points_ref: np.ndarray, observed: np.ndarray, camera: Camera) -> List[np.ndarray]:
    if points_ref.shape[0] < 4:
        return []
    fu, fv, cu, cv = camera.intrinsics
    K = np.array([[fu, 0.0, cu], [0.0, fv, cv], [0.0, 0.0, 1.0]], dtype=np.float64)
    points = points_ref.astype(np.float64)
    image_points = observed.astype(np.float64)
    distortion = np.zeros((4, 1), dtype=np.float64)
    candidates: List[np.ndarray] = []

    def append(rvec: np.ndarray, tvec: np.ndarray) -> None:
        pose = np.concatenate([rvec.reshape(3), tvec.reshape(3)]).astype(np.float64)
        if np.isfinite(pose).all() and not any(np.linalg.norm(pose - old) < 1e-8 for old in candidates):
            candidates.append(pose)

    # A planar square has two IPPE solutions. Keep both and let the real KB
    # projection residual choose, just as the C++ Outer4 pose path does.
    for flag in (cv2.SOLVEPNP_IPPE, cv2.SOLVEPNP_SQPNP):
        try:
            result = cv2.solvePnPGeneric(points, image_points, K, distortion, flags=flag)
        except cv2.error:
            continue
        if result and bool(result[0]):
            for rvec, tvec in zip(result[1], result[2]):
                append(rvec, tvec)
    try:
        ok, rvec, tvec = cv2.solvePnP(
            points, image_points, K, distortion, flags=cv2.SOLVEPNP_ITERATIVE
        )
        if ok:
            append(rvec, tvec)
    except cv2.error:
        pass
    return candidates


def optimize_pose_from_initial(
    points_ref: np.ndarray,
    observed: np.ndarray,
    camera: Camera,
    params: np.ndarray,
    max_iterations: int = 50,
) -> PoseFit:
    params = params.copy()
    damping = 1e-3
    last_r, last_failures = residual_vector(points_ref, observed, params, camera)
    last_cost = float(last_r @ last_r)
    iterations = 0
    eps = np.array([1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6], dtype=np.float64)
    for iterations in range(1, max_iterations + 1):
        J = np.zeros((last_r.size, 6), dtype=np.float64)
        for j in range(6):
            step = np.zeros(6, dtype=np.float64)
            step[j] = eps[j]
            rp, _ = residual_vector(points_ref, observed, params + step, camera)
            rm, _ = residual_vector(points_ref, observed, params - step, camera)
            J[:, j] = (rp - rm) / (2.0 * eps[j])
        H = J.T @ J
        g = J.T @ last_r
        diag = np.maximum(np.diag(H), 1.0)
        accepted = False
        for _ in range(12):
            try:
                delta = np.linalg.solve(H + damping * np.diag(diag), -g)
            except np.linalg.LinAlgError:
                damping *= 10.0
                continue
            if not np.isfinite(delta).all():
                damping *= 10.0
                continue
            candidate = params + delta
            r_new, failures_new = residual_vector(points_ref, observed, candidate, camera)
            cost_new = float(r_new @ r_new)
            if np.isfinite(cost_new) and cost_new < last_cost:
                params = candidate
                last_r = r_new
                last_failures = failures_new
                last_cost = cost_new
                damping = max(damping * 0.3, 1e-9)
                accepted = True
                break
            damping *= 10.0
        if not accepted or np.linalg.norm(delta) < 1e-9:
            break
    proj, valid = project_kb(points_ref, params, camera)
    err = np.linalg.norm(proj - observed, axis=1)
    err[~valid] = np.nan
    success = np.isfinite(err).any()
    return PoseFit(success, params, err, int((~valid).sum()), iterations)


def optimize_pose(
    points_ref: np.ndarray,
    observed: np.ndarray,
    camera: Camera,
    max_iterations: int = 50,
) -> PoseFit:
    candidates = initial_pose_candidates(points_ref, observed, camera)
    if not candidates:
        return PoseFit(False, None, np.array([], dtype=np.float64), 0, 0, "solvePnP_failed")
    fits = [
        optimize_pose_from_initial(points_ref, observed, camera, candidate, max_iterations)
        for candidate in candidates
    ]
    valid_fits = [fit for fit in fits if fit.success and fit.errors.size]
    if not valid_fits:
        return PoseFit(False, None, np.array([], dtype=np.float64), 0, 0, "all_pose_hypotheses_failed")
    return min(
        valid_fits,
        key=lambda fit: (
            fit.projection_failures,
            float(np.nansum(fit.errors * fit.errors)),
        ),
    )


def stats(errors: Sequence[float], thresholds: Sequence[float]) -> Dict[str, float]:
    arr = np.asarray([e for e in errors if np.isfinite(e)], dtype=np.float64)
    out: Dict[str, float] = {
        "point_count": int(arr.size),
        "rmse": float("nan"),
        "mean": float("nan"),
        "median": float("nan"),
        "p95": float("nan"),
    }
    for tau in thresholds:
        out[f"inlier_ratio_{tau:g}px"] = float("nan")
        out[f"inl_{tau:g}px_percent"] = float("nan")
    if arr.size == 0:
        return out
    out["rmse"] = float(math.sqrt(float(np.mean(arr * arr))))
    out["mean"] = float(np.mean(arr))
    out["median"] = float(np.median(arr))
    out["p95"] = float(np.percentile(arr, 95.0))
    for tau in thresholds:
        ratio = float(np.mean(arr < tau))
        out[f"inlier_ratio_{tau:g}px"] = ratio
        out[f"inl_{tau:g}px_percent"] = 100.0 * ratio
    return out


def split_arrays(points: Sequence[PointObs]) -> Tuple[np.ndarray, np.ndarray]:
    return (
        np.vstack([p.p_ref for p in points]).astype(np.float64),
        np.vstack([p.observed for p in points]).astype(np.float64),
    )


def evaluate_camera(
    camera: Camera,
    frames: Dict[int, List[PointObs]],
    thresholds: Sequence[float],
) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    all_errors: List[float] = []
    cross_errors: List[float] = []
    per_frame_rows: List[Dict[str, object]] = []
    pose_success = 0
    pose_attempt = 0
    cross_attempt = 0
    cross_success = 0
    projection_failures = 0
    cross_projection_failures = 0

    for frame_index in sorted(frames):
        pts = frames[frame_index]
        frame_label = pts[0].frame_label if pts else ""
        points_ref, observed = split_arrays(pts)
        pose_attempt += 1
        fit = optimize_pose(points_ref, observed, camera)
        if fit.success:
            pose_success += 1
            finite_errors = [float(e) for e in fit.errors if np.isfinite(e)]
            all_errors.extend(finite_errors)
            projection_failures += fit.projection_failures
            st = stats(finite_errors, thresholds)
        else:
            st = stats([], thresholds)
        row: Dict[str, object] = {
            "method": camera.label,
            "frame_index": frame_index,
            "frame_label": frame_label,
            "visible_board_count": len({p.board_id for p in pts}),
            "point_count": len(pts),
            "pose_success": int(fit.success),
            "pose_iterations": fit.iterations,
            "rms": st["rmse"],
            "mean": st["mean"],
            "median": st["median"],
            "projection_failure_count": fit.projection_failures,
            "failure_reason": fit.failure_reason,
        }
        for tau in thresholds:
            row[f"inl_{tau:g}px_percent"] = st[f"inl_{tau:g}px_percent"]

        frame_cross_errors: List[float] = []
        board_ids = sorted({p.board_id for p in pts})
        for source_board in board_ids:
            source_pts = [p for p in pts if p.board_id == source_board]
            other_pts = [p for p in pts if p.board_id != source_board]
            if len(source_pts) < 4 or not other_pts:
                continue
            cross_attempt += 1
            src_ref, src_obs = split_arrays(source_pts)
            source_fit = optimize_pose(src_ref, src_obs, camera)
            if not source_fit.success or source_fit.params is None:
                continue
            other_ref, other_obs = split_arrays(other_pts)
            proj, valid = project_kb(other_ref, source_fit.params, camera)
            err = np.linalg.norm(proj - other_obs, axis=1)
            err = err[np.isfinite(err) & valid]
            cross_projection_failures += int((~valid).sum())
            if err.size == 0:
                continue
            cross_success += 1
            err_list = [float(e) for e in err]
            frame_cross_errors.extend(err_list)
            cross_errors.extend(err_list)
        cross_st = stats(frame_cross_errors, thresholds)
        row["cross_board_rmse"] = cross_st["rmse"]
        row["cross_board_mean"] = cross_st["mean"]
        row["cross_board_median"] = cross_st["median"]
        for tau in thresholds:
            row[f"cross_board_inl_{tau:g}px_percent"] = cross_st[f"inl_{tau:g}px_percent"]
        per_frame_rows.append(row)

    global_stats = stats(all_errors, thresholds)
    cross_stats = stats(cross_errors, thresholds)
    summary: Dict[str, object] = {
        "method": camera.label,
        "camera_model_family": camera.family,
        "camera_intrinsics": camera.intrinsics,
        "camera_distortion": camera.distortion,
        "frame_count": len(frames),
        "pose_success_count": pose_success,
        "pose_attempt_count": pose_attempt,
        "pose_success_rate": pose_success / pose_attempt if pose_attempt else float("nan"),
        "rmse": global_stats["rmse"],
        "mean": global_stats["mean"],
        "median": global_stats["median"],
        "point_count": global_stats["point_count"],
        "projection_failure_count": projection_failures,
        "cross_board_attempt_count": cross_attempt,
        "cross_board_success_count": cross_success,
        "cross_board_success_rate": cross_success / cross_attempt if cross_attempt else float("nan"),
        "cross_board_rmse": cross_stats["rmse"],
        "cross_board_mean": cross_stats["mean"],
        "cross_board_median": cross_stats["median"],
        "cross_board_point_count": cross_stats["point_count"],
        "cross_board_projection_failure_count": cross_projection_failures,
    }
    for tau in thresholds:
        summary[f"inlier_ratio_{tau:g}px"] = global_stats[f"inlier_ratio_{tau:g}px"]
        summary[f"inl_{tau:g}px_percent"] = global_stats[f"inl_{tau:g}px_percent"]
        summary[f"cross_board_inlier_ratio_{tau:g}px"] = cross_stats[f"inlier_ratio_{tau:g}px"]
        summary[f"cross_board_inl_{tau:g}px_percent"] = cross_stats[f"inl_{tau:g}px_percent"]
    return summary, per_frame_rows


def evaluate_camera_per_frame_board_outer_pose(
    camera: Camera,
    frames: Dict[int, List[PointObs]],
    thresholds: Sequence[float],
) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    """Match the Stage5 holdout protocol: Outer4 pose fit, all-point scoring."""
    all_errors: List[float] = []
    outer_errors: List[float] = []
    internal_errors: List[float] = []
    per_observation_rows: List[Dict[str, object]] = []
    pose_success = 0
    pose_attempt = 0
    projection_failures = 0

    observations = [
        (frame_index, board_id, [point for point in points if point.board_id == board_id])
        for frame_index, points in sorted(frames.items())
        for board_id in sorted({point.board_id for point in points})
    ]
    for frame_index, board_id, points in observations:
        outer = [point for point in points if point.point_type == "outer"]
        pose_attempt += 1
        fit = PoseFit(False, None, np.array([], dtype=np.float64), 0, 0, "fewer_than_four_outer_points")
        if len(outer) >= 4:
            outer_xyz = np.vstack([point.p_board for point in outer]).astype(np.float64)
            outer_uv = np.vstack([point.observed for point in outer]).astype(np.float64)
            fit = optimize_pose(outer_xyz, outer_uv, camera)
        if fit.success and fit.params is not None:
            pose_success += 1
            xyz = np.vstack([point.p_board for point in points]).astype(np.float64)
            observed = np.vstack([point.observed for point in points]).astype(np.float64)
            projected, valid = project_kb(xyz, fit.params, camera)
            errors = np.linalg.norm(projected - observed, axis=1)
            errors[~valid] = np.nan
            projection_failures += int((~valid).sum())
            all_errors.extend(float(error) for error in errors if np.isfinite(error))
            outer_errors.extend(
                float(error)
                for point, error in zip(points, errors)
                if point.point_type == "outer" and np.isfinite(error)
            )
            internal_errors.extend(
                float(error)
                for point, error in zip(points, errors)
                if point.point_type != "outer" and np.isfinite(error)
            )
            row_stats = stats([float(error) for error in errors if np.isfinite(error)], thresholds)
        else:
            row_stats = stats([], thresholds)
        row: Dict[str, object] = {
            "method": camera.label,
            "frame_index": frame_index,
            "frame_label": points[0].frame_label if points else "",
            "board_id": board_id,
            "point_count": len(points),
            "outer_point_count": len(outer),
            "pose_success": int(fit.success),
            "pose_iterations": fit.iterations,
            "rmse": row_stats["rmse"],
            "p95": row_stats["p95"],
            "projection_failure_count": fit.projection_failures,
            "failure_reason": fit.failure_reason,
        }
        for tau in thresholds:
            row[f"inl_{tau:g}px_percent"] = row_stats[f"inl_{tau:g}px_percent"]
        per_observation_rows.append(row)

    all_stats = stats(all_errors, thresholds)
    outer_stats = stats(outer_errors, thresholds)
    internal_stats = stats(internal_errors, thresholds)
    summary: Dict[str, object] = {
        "method": camera.label,
        "camera_model_family": camera.family,
        "camera_intrinsics": camera.intrinsics,
        "camera_distortion": camera.distortion,
        "pose_scope": "per_frame_board_outer4_refit",
        "frame_board_observation_count": len(observations),
        "pose_success_count": pose_success,
        "pose_attempt_count": pose_attempt,
        "pose_success_rate": pose_success / pose_attempt if pose_attempt else float("nan"),
        "point_count": all_stats["point_count"],
        "rmse": all_stats["rmse"],
        "p95": all_stats["p95"],
        "outer_point_count": outer_stats["point_count"],
        "outer_rmse": outer_stats["rmse"],
        "internal_point_count": internal_stats["point_count"],
        "internal_rmse": internal_stats["rmse"],
        "projection_failure_count": projection_failures,
    }
    for tau in thresholds:
        summary[f"inlier_ratio_{tau:g}px"] = all_stats[f"inlier_ratio_{tau:g}px"]
        summary[f"inl_{tau:g}px_percent"] = all_stats[f"inl_{tau:g}px_percent"]
    return summary, per_observation_rows


def write_csv(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    rows = list(rows)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--points-csv", type=Path, default=None)
    parser.add_argument("--layout-csv", type=Path, required=True)
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--methods", default="ours,kalibr")
    parser.add_argument(
        "--camera-yaml",
        action="append",
        default=[],
        metavar="LABEL:PATH",
        help="Evaluate an explicit canonical KB YAML; may be repeated.",
    )
    parser.add_argument("--point-method", default="ours")
    parser.add_argument("--inlier-thresholds", default="1.5,3")
    parser.add_argument(
        "--pose-scope",
        choices=("per_frame_layout", "per_frame_board_outer4"),
        default="per_frame_layout",
        help="Use per-frame board layout fitting (legacy) or Stage5-style Outer4 board-pose refits.",
    )
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    points_csv = (args.points_csv or (run_dir / "benchmark_holdout_points.csv")).resolve()
    summary_path = (args.summary or (run_dir / "benchmark_holdout_summary.txt")).resolve()
    layout_csv = args.layout_csv.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_kv = parse_key_value_file(summary_path)
    boards = load_board_layout(layout_csv)
    frames = load_points(points_csv, args.point_method, boards)
    thresholds = [float(x) for x in args.inlier_thresholds.split(",") if x.strip()]
    method_prefixes = {
        "ours": "our",
        "kalibr": "kalibr",
        "tartancalib_kb": "reference_tartancalib_kb",
    }
    requested_methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    explicit_cameras: List[Camera] = []
    for spec in args.camera_yaml:
        if ":" not in spec:
            raise ValueError(f"--camera-yaml must be LABEL:PATH, got {spec!r}")
        label, raw_path = spec.split(":", 1)
        if not label or not raw_path:
            raise ValueError(f"--camera-yaml must be LABEL:PATH, got {spec!r}")
        explicit_cameras.append(camera_from_yaml(Path(raw_path).resolve(), label))
    summaries: List[Dict[str, object]] = []
    all_per_frame: List[Dict[str, object]] = []
    cameras: List[Camera] = explicit_cameras
    if not cameras:
        for method in requested_methods:
            if method not in method_prefixes:
                raise ValueError(f"Unknown method {method!r}; known={sorted(method_prefixes)}")
            cameras.append(camera_from_summary(summary_kv, method, method_prefixes[method]))
    for camera in cameras:
        if args.pose_scope == "per_frame_board_outer4":
            method_summary, per_frame = evaluate_camera_per_frame_board_outer_pose(camera, frames, thresholds)
        else:
            method_summary, per_frame = evaluate_camera(camera, frames, thresholds)
        summaries.append(method_summary)
        all_per_frame.extend(per_frame)

    payload: Dict[str, object] = {
        "evaluation": "camera_pose_estimation_fixed_intrinsics_fixed_layout",
        "projection_chain": "p_camera = T_camera_reference * T_reference_board(board) * p_board",
        "intrinsics_optimized": 0,
        "board_layout_optimized": 0,
        "frame_pose_optimized": 1,
        "camera_model": "pinhole-equi / KB",
        "pose_scope": args.pose_scope,
        "run_dir": str(run_dir),
        "points_csv": str(points_csv),
        "point_method_source": args.point_method,
        "layout_csv": str(layout_csv),
        "summary_source": str(summary_path),
        "frame_count": len(frames),
        "board_count": len(boards),
        "thresholds_px": thresholds,
        "methods": summaries,
    }
    (output_dir / "pose_eval_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    write_csv(output_dir / "pose_eval_per_frame.csv", all_per_frame)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
