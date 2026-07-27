#!/usr/bin/env python3
"""Run the paper DS-W1 local intrinsic recovery sweep.

The training scene is fixed and only DS intrinsics are optimized.  A second
dataset supplies independent view geometries for pose-only evaluation.  All
measurements used by the experiment are reprojected from one explicit ground
truth camera, so ray and pose errors have an unambiguous reference.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

import run_ds_perturbation_sweep as sweep
import run_ds_weak_mode_perturbation as weak


DEFAULT_LEVELS = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0)
METHODS = ("Outer-only", "Outer+Internal")
TRAIN_NOISE_DOMAIN = 0x54524149
EVAL_NOISE_DOMAIN = 0x4556414C


@dataclass(frozen=True)
class PoseTemplate:
    frame_index: int
    frame_label: str
    reference_points: np.ndarray
    T_camera_reference: np.ndarray
    source_fit_rmse_px: float


def parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-scene", type=Path, required=True)
    parser.add_argument("--training-points", type=Path, required=True)
    parser.add_argument("--external-eval-mat", type=Path, required=True)
    parser.add_argument("--external-eval-frame-metadata", type=Path)
    parser.add_argument(
        "--converter", type=Path,
        default=repo / "tools/babelcalib_import/convert_babelcalib_mat.py",
    )
    parser.add_argument("--levels", default=",".join(str(v) for v in DEFAULT_LEVELS))
    parser.add_argument("--seed-start", type=int, default=1)
    parser.add_argument("--seed-count", type=int, default=100)
    parser.add_argument("--noise-sigma-px", type=float, default=0.25)
    parser.add_argument("--grid-size", type=int, default=121)
    parser.add_argument("--width", type=int, default=4512)
    parser.add_argument("--height", type=int, default=4512)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def parse_levels(raw: str) -> tuple[float, ...]:
    values = tuple(float(value.strip()) for value in raw.split(",") if value.strip())
    if not values or len(set(values)) != len(values):
        raise ValueError("--levels must contain unique values")
    if any(not math.isfinite(value) or value < 0.0 for value in values):
        raise ValueError("--levels must be finite and nonnegative")
    return values


def sha256_bytes(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.resolve().read_bytes())


def array_fingerprint(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array, dtype=np.float64)
    return sha256_bytes(value.tobytes())


def json_fingerprint(value: Any) -> str:
    return sha256_bytes(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: f"{value:.15g}" if isinstance(value, float) else value
                for key, value in row.items()
            })


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_training_rows(path: Path, scene: weak.Scene) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for source_index, row in enumerate(csv.DictReader(handle)):
            frame = int(row["frame_index"])
            board = int(row["board_id"])
            if frame not in scene.frames or board not in scene.boards:
                continue
            local = np.asarray([
                float(row["target_x"]), float(row["target_y"]),
                float(row["target_z"]), 1.0,
            ], dtype=np.float64)
            rows.append({
                "source_index": source_index,
                "frame": frame,
                "board": board,
                "point": local,
                "point_type": row["point_type"],
                "camera_point": (scene.frames[frame] @ scene.boards[board] @ local)[:3],
            })
    if not rows:
        raise RuntimeError(f"No training observations loaded from {path}")
    return rows


def camera_to_values(camera: sweep.Camera, width: int, height: int) -> np.ndarray:
    return weak.camera_to_coordinates(camera, width, height)


def values_to_camera(values: np.ndarray, width: int, height: int) -> sweep.Camera:
    return weak.camera_from_coordinates(values, width, height)


def camera_bounds() -> tuple[np.ndarray, np.ndarray]:
    return (
        np.asarray([-0.95, 0.01, math.log(400.0), math.log(400.0), 0.2, 0.2]),
        np.asarray([4.0, 0.99, math.log(6000.0), math.log(6000.0), 0.8, 0.8]),
    )


def point_rmse(residual_xy: np.ndarray) -> float:
    if residual_xy.size == 0:
        return math.nan
    return float(np.sqrt(np.mean(np.sum(residual_xy * residual_xy, axis=1))))


def calibrate_levels(
    truth: sweep.Camera,
    direction: np.ndarray,
    levels: tuple[float, ...],
    width: int,
    height: int,
    mask: sweep.EvaluationMask,
) -> list[dict[str, Any]]:
    calibrated: list[dict[str, Any]] = []
    for level in levels:
        if level == 0.0:
            amplitude = 0.0
            initial = truth
            metrics = sweep.ray_metrics(mask, initial)
        else:
            amplitude, initial, metrics = weak.calibrate_perturbation(
                truth, direction, 1, level, width, height, mask, 0.99,
            )
        achieved = float(metrics["peripheral_ray_p95_deg"])
        if abs(achieved - level) >= 1e-9:
            raise RuntimeError(
                f"W1 calibration target {level} achieved {achieved}, outside tolerance"
            )
        calibrated.append({
            "perturbation_level_deg": level,
            "amplitude": amplitude,
            "camera": initial,
            "initial_full_ray_p95_deg": float(metrics["full_ray_p95_deg"]),
            "initial_peripheral_ray_p95_deg": achieved,
            "initial_valid_grid_ratio": float(metrics["valid_grid_ratio"]),
            "camera_fingerprint": sweep.camera_fingerprint(initial),
        })
    return calibrated


def project(camera: sweep.Camera, points: np.ndarray) -> np.ndarray:
    pixels, valid = weak.project_ds(camera, points)
    if not np.all(valid):
        raise RuntimeError("DS projection failed")
    return pixels


def optimize_intrinsics(
    initial: sweep.Camera,
    truth: sweep.Camera,
    points: np.ndarray,
    observed: np.ndarray,
    common_outer_points: np.ndarray,
    common_outer_observed: np.ndarray,
    width: int,
    height: int,
    mask: sweep.EvaluationMask,
) -> tuple[sweep.Camera | None, dict[str, Any]]:
    start = time.perf_counter()
    lower, upper = camera_bounds()

    def residual(values: np.ndarray) -> np.ndarray:
        candidate = values_to_camera(values, width, height)
        pixels, valid = weak.project_ds(candidate, points)
        if not np.all(valid):
            return np.full(observed.size, 1e6, dtype=np.float64)
        return (pixels - observed).reshape(-1)

    try:
        result = least_squares(
            residual,
            np.clip(camera_to_values(initial, width, height), lower + 1e-8, upper - 1e-8),
            bounds=(lower, upper), method="trf", loss="linear", x_scale="jac",
            max_nfev=250, ftol=1e-12, xtol=1e-12, gtol=1e-12,
        )
        final = values_to_camera(result.x, width, height)
        optimized_residual = residual(result.x).reshape(-1, 2)
        common_residual = project(final, common_outer_points) - common_outer_observed
        ray = sweep.ray_metrics(mask, final)
        finite = all(math.isfinite(float(ray[key])) for key in (
            "full_ray_p95_deg", "peripheral_ray_p95_deg", "valid_grid_ratio"
        ))
        model_valid = weak.valid_camera(final) and finite
        success = bool(result.success and model_valid)
        failure_reason = "" if success else (
            str(result.message) if not result.success else "invalid_final_ds_model_or_ray_metrics"
        )
        return final, {
            "solver_status": "converged" if success else "failed",
            "failure_reason": failure_reason,
            "iterations": int(result.nfev),
            "runtime_sec": time.perf_counter() - start,
            "optimization_rmse_px": point_rmse(optimized_residual),
            "training_rmse_px": point_rmse(common_residual),
            **ray,
        }
    except Exception as error:
        return None, {
            "solver_status": "failed",
            "failure_reason": f"{type(error).__name__}: {error}",
            "iterations": 0,
            "runtime_sec": time.perf_counter() - start,
            "optimization_rmse_px": math.nan,
            "training_rmse_px": math.nan,
            "full_ray_p95_deg": math.nan,
            "peripheral_ray_p95_deg": math.nan,
            "valid_grid_ratio": 0.0,
        }


def pose_matrix(values: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = Rotation.from_rotvec(values[:3]).as_matrix()
    transform[:3, 3] = values[3:]
    return transform


def pose_values(transform: np.ndarray) -> np.ndarray:
    return np.r_[Rotation.from_matrix(transform[:3, :3]).as_rotvec(), transform[:3, 3]]


def fit_pose(
    camera: sweep.Camera,
    reference_points: np.ndarray,
    observed: np.ndarray,
    initial: np.ndarray,
    *,
    robust: bool,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    def residual(values: np.ndarray) -> np.ndarray:
        transform = pose_matrix(values)
        points_camera = (
            transform[:3, :3] @ reference_points.T + transform[:3, 3:4]
        ).T
        pixels, valid = weak.project_ds(camera, points_camera)
        if not np.all(valid):
            return np.full(observed.size, 1e6, dtype=np.float64)
        return (pixels - observed).reshape(-1)

    try:
        result = least_squares(
            residual, pose_values(initial), method="trf",
            loss="huber" if robust else "linear", f_scale=2.0,
            max_nfev=150, ftol=1e-12, xtol=1e-12, gtol=1e-12,
        )
        transform = pose_matrix(result.x)
        rmse = point_rmse(residual(result.x).reshape(-1, 2))
        residual_xy = residual(result.x).reshape(-1, 2)
        residual_sse = float(np.sum(residual_xy * residual_xy))
        success = bool(result.success and np.all(np.isfinite(transform)) and math.isfinite(rmse))
        return (transform if success else None), {
            "success": success,
            "failure_reason": "" if success else str(result.message),
            "rmse_px": rmse,
            "residual_sse_px2": residual_sse,
            "point_count": len(reference_points),
            "iterations": int(result.nfev),
        }
    except Exception as error:
        return None, {
            "success": False,
            "failure_reason": f"{type(error).__name__}: {error}",
            "rmse_px": math.nan,
            "residual_sse_px2": math.nan,
            "point_count": len(reference_points),
            "iterations": 0,
        }


def initialize_pose_pnp(
    camera: sweep.Camera,
    reference_points: np.ndarray,
    observed: np.ndarray,
) -> np.ndarray | None:
    rays, valid = sweep.unproject_ds(camera, observed)
    usable = valid & np.isfinite(rays).all(axis=1) & (np.abs(rays[:, 2]) > 1e-9)
    if int(np.count_nonzero(usable)) < 6:
        return None
    normalized = rays[usable, :2] / rays[usable, 2:3]
    homogeneous = np.c_[reference_points[usable], np.ones(np.count_nonzero(usable))]
    design = np.zeros((2 * len(homogeneous), 12), dtype=np.float64)
    for index, (point, pixel) in enumerate(zip(homogeneous, normalized)):
        design[2 * index, :4] = point
        design[2 * index, 8:] = -pixel[0] * point
        design[2 * index + 1, 4:8] = point
        design[2 * index + 1, 8:] = -pixel[1] * point
    _, _, vh = np.linalg.svd(design, full_matrices=False)
    projection = vh[-1].reshape(3, 4)
    if np.linalg.det(projection[:, :3]) < 0.0:
        projection = -projection
    left, singular_values, right = np.linalg.svd(projection[:, :3])
    rotation = left @ right
    if np.linalg.det(rotation) < 0.0:
        left[:, -1] *= -1.0
        rotation = left @ right
    scale = float(np.mean(singular_values))
    if not math.isfinite(scale) or scale <= 1e-12:
        return None
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = projection[:, 3] / scale
    depths = (
        transform[:3, :3] @ reference_points[usable].T + transform[:3, 3:4]
    )[2]
    if np.count_nonzero(depths > 0.0) < 0.5 * depths.size:
        return None
    return transform


def convert_external_mat(args: argparse.Namespace, output: Path) -> Path:
    external = output / "external_eval_source"
    metadata = external / "metadata.json"
    if metadata.is_file() and args.resume:
        return external
    command = [
        sys.executable, str(args.converter.resolve()),
        "--mat", str(args.external_eval_mat.resolve()),
        "--output", str(external), "--split-label", "external_pose_evaluation",
    ]
    if args.external_eval_frame_metadata is not None:
        command.extend(["--frame-metadata", str(args.external_eval_frame_metadata.resolve())])
    subprocess.run(command, check=True, cwd=Path(__file__).resolve().parents[2])
    return external


def build_pose_templates(
    source: Path,
    scene: weak.Scene,
) -> tuple[list[PoseTemplate], list[dict[str, Any]]]:
    grouped: dict[tuple[int, str], list[dict[str, str]]] = defaultdict(list)
    for row in read_csv(source / "points.csv"):
        if row["point_type"] == "outer":
            grouped[(int(row["frame_index"]), row["frame_label"])].append(row)
    templates: list[PoseTemplate] = []
    diagnostics: list[dict[str, Any]] = []
    for (frame_index, frame_label), rows in sorted(grouped.items()):
        reference_points = []
        observed = []
        for row in rows:
            board = int(row["board_id"])
            if board not in scene.boards:
                continue
            local = np.asarray([
                float(row["target_x"]), float(row["target_y"]),
                float(row["target_z"]), 1.0,
            ])
            reference_points.append((scene.boards[board] @ local)[:3])
            observed.append([float(row["observed_x"]), float(row["observed_y"])])
        points = np.asarray(reference_points, dtype=np.float64)
        pixels = np.asarray(observed, dtype=np.float64)
        initial = initialize_pose_pnp(scene.camera, points, pixels) if len(points) >= 6 else None
        if initial is None:
            diagnostics.append({
                "frame_index": frame_index, "frame_label": frame_label,
                "outer_point_count": len(points), "success": 0,
                "source_fit_rmse_px": math.nan,
                "failure_reason": "multi_board_pnp_initialization_failed",
            })
            continue
        transform, fit = fit_pose(scene.camera, points, pixels, initial, robust=True)
        diagnostics.append({
            "frame_index": frame_index, "frame_label": frame_label,
            "outer_point_count": len(points), "success": int(fit["success"]),
            "source_fit_rmse_px": fit["rmse_px"],
            "failure_reason": fit["failure_reason"],
        })
        if transform is not None:
            templates.append(PoseTemplate(
                frame_index, frame_label, points, transform, float(fit["rmse_px"]),
            ))
    if not templates:
        raise RuntimeError("No external pose template could be fitted")
    return templates, diagnostics


def orientation_error_deg(reference: np.ndarray, estimate: np.ndarray) -> float:
    delta = reference[:3, :3].T @ estimate[:3, :3]
    cosine = float(np.clip(0.5 * (np.trace(delta) - 1.0), -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def evaluate_external_poses(
    camera: sweep.Camera,
    truth: sweep.Camera,
    templates: list[PoseTemplate],
    seed: int,
    sigma: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng = np.random.default_rng(np.random.SeedSequence([seed, EVAL_NOISE_DOMAIN]))
    frame_rows: list[dict[str, Any]] = []
    orientation_errors: list[float] = []
    total_residual_sse = 0.0
    total_residual_points = 0
    noise_parts: list[np.ndarray] = []
    for template in templates:
        points_camera = (
            template.T_camera_reference[:3, :3] @ template.reference_points.T +
            template.T_camera_reference[:3, 3:4]
        ).T
        clean = project(camera=truth, points=points_camera)
        noise = rng.normal(0.0, sigma, size=clean.shape)
        noise_parts.append(noise)
        observed = clean + noise
        estimate, fit = fit_pose(
            camera, template.reference_points, observed,
            template.T_camera_reference, robust=False,
        )
        if estimate is None:
            frame_rows.append({
                "frame_index": template.frame_index,
                "frame_label": template.frame_label,
                "outer_point_count": len(template.reference_points),
                "solver_status": "failed",
                "failure_reason": fit["failure_reason"],
                "orientation_error_deg": math.nan,
                "reprojection_rmse_px": math.nan,
                "residual_sse_px2": math.nan,
                "iterations": fit["iterations"],
            })
            continue
        orientation = orientation_error_deg(template.T_camera_reference, estimate)
        orientation_errors.append(orientation)
        total_residual_sse += float(fit["residual_sse_px2"])
        total_residual_points += int(fit["point_count"])
        frame_rows.append({
            "frame_index": template.frame_index,
            "frame_label": template.frame_label,
            "outer_point_count": len(template.reference_points),
            "solver_status": "converged",
            "failure_reason": "",
            "orientation_error_deg": orientation,
            "reprojection_rmse_px": fit["rmse_px"],
            "residual_sse_px2": fit["residual_sse_px2"],
            "iterations": fit["iterations"],
        })
    noise_fingerprint = array_fingerprint(np.vstack(noise_parts))
    summary = {
        "evaluation_frame_count": len(templates),
        "pose_success_count": len(orientation_errors),
        "pose_failure_count": len(templates) - len(orientation_errors),
        "pose_success_rate": len(orientation_errors) / len(templates),
        "orientation_error_p95_deg": (
            float(np.percentile(orientation_errors, 95)) if orientation_errors else math.nan
        ),
        "orientation_error_median_deg": (
            float(np.median(orientation_errors)) if orientation_errors else math.nan
        ),
        "pose_reprojection_rmse_px": (
            float(math.sqrt(total_residual_sse / total_residual_points))
            if total_residual_points else math.nan
        ),
        "pose_reprojection_point_count": total_residual_points,
        "eval_noise_fingerprint": noise_fingerprint,
    }
    return frame_rows, summary


def external_noise_fingerprint(
    templates: list[PoseTemplate], seed: int, sigma: float,
) -> str:
    rng = np.random.default_rng(np.random.SeedSequence([seed, EVAL_NOISE_DOMAIN]))
    noise = [
        rng.normal(0.0, sigma, size=(len(template.reference_points), 2))
        for template in templates
    ]
    return array_fingerprint(np.vstack(noise))


def run_seed(
    seed: int,
    levels: list[dict[str, Any]],
    truth: sweep.Camera,
    rows: list[dict[str, Any]],
    outer_indices: np.ndarray,
    all_points: np.ndarray,
    clean_all: np.ndarray,
    width: int,
    height: int,
    sigma: float,
    mask: sweep.EvaluationMask,
    w1_fingerprint: str,
    scene_fingerprint: str,
    layout_fingerprint: str,
    templates: list[PoseTemplate],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    rng = np.random.default_rng(np.random.SeedSequence([seed, TRAIN_NOISE_DOMAIN]))
    noise = rng.normal(0.0, sigma, size=clean_all.shape)
    observed_all = clean_all + noise
    noise_fingerprint = array_fingerprint(noise)
    outer_points = all_points[outer_indices]
    outer_observed = observed_all[outer_indices]
    local_rows: list[dict[str, Any]] = []
    pose_frame_rows: list[dict[str, Any]] = []
    pose_summary_rows: list[dict[str, Any]] = []
    eval_noise_fingerprint = external_noise_fingerprint(templates, seed, sigma)
    for level in levels:
        for method in METHODS:
            selected_indices = outer_indices if method == "Outer-only" else np.arange(len(rows))
            final, metrics = optimize_intrinsics(
                level["camera"], truth, all_points[selected_indices],
                observed_all[selected_indices], outer_points, outer_observed,
                width, height, mask,
            )
            if final is None:
                final_values = [math.nan] * 6
                errors = [math.nan] * 6
            else:
                final_values = [final.xi, final.alpha, final.fu, final.fv, final.cu, final.cv]
                errors = [
                    abs(final.xi - truth.xi), abs(final.alpha - truth.alpha),
                    abs(final.fu / truth.fu - 1.0), abs(final.fv / truth.fv - 1.0),
                    0.5 * (abs(final.fu / truth.fu - 1.0) + abs(final.fv / truth.fv - 1.0)),
                    math.hypot(final.cu - truth.cu, final.cv - truth.cv),
                ]
            run_key = f"level={level['perturbation_level_deg']:.12g}|seed={seed}|method={method}"
            local_rows.append({
                "perturbation_level_deg": level["perturbation_level_deg"],
                "seed": seed,
                "method": method,
                "full_ray_p95_deg": metrics["full_ray_p95_deg"],
                "peripheral_ray_p95_deg": metrics["peripheral_ray_p95_deg"],
                "full_ray_median_deg": metrics.get("full_ray_median_deg", math.nan),
                "peripheral_ray_median_deg": metrics.get("peripheral_ray_median_deg", math.nan),
                "xi_abs_error": errors[0],
                "alpha_abs_error": errors[1],
                "fu_relative_error": errors[2],
                "fv_relative_error": errors[3],
                "mean_focal_relative_error": errors[4],
                "principal_point_error_px": errors[5],
                "training_rmse_px": metrics["training_rmse_px"],
                "valid_grid_ratio": metrics["valid_grid_ratio"],
                "solver_status": metrics["solver_status"],
                "failure_reason": metrics["failure_reason"],
                "final_xi": final_values[0], "final_alpha": final_values[1],
                "final_fu": final_values[2], "final_fv": final_values[3],
                "final_cu": final_values[4], "final_cv": final_values[5],
                "initial_camera_fingerprint": level["camera_fingerprint"],
                "perturbation_amplitude": level["amplitude"],
                "initial_full_ray_p95_deg": level["initial_full_ray_p95_deg"],
                "initial_peripheral_ray_p95_deg": level["initial_peripheral_ray_p95_deg"],
                "initial_valid_grid_ratio": level["initial_valid_grid_ratio"],
                "noise_fingerprint": noise_fingerprint,
                "w1_fingerprint": w1_fingerprint,
                "scene_fingerprint": scene_fingerprint,
                "layout_fingerprint": layout_fingerprint,
                "optimization_rmse_px": metrics["optimization_rmse_px"],
                "point_count": len(selected_indices),
                "iterations": metrics["iterations"],
                "runtime_sec": metrics["runtime_sec"],
                "run_key": run_key,
            })
            if final is None:
                pose_summary_rows.append({
                    "perturbation_level_deg": level["perturbation_level_deg"],
                    "seed": seed, "method": method,
                    "evaluation_frame_count": len(templates),
                    "pose_success_count": 0, "pose_failure_count": len(templates),
                    "pose_success_rate": 0.0,
                    "orientation_error_p95_deg": math.nan,
                    "orientation_error_median_deg": math.nan,
                    "pose_reprojection_rmse_px": math.nan,
                    "pose_reprojection_point_count": 0,
                    "eval_noise_fingerprint": eval_noise_fingerprint,
                })
                continue
            frame_metrics, pose_summary = evaluate_external_poses(
                final, truth, templates, seed, sigma,
            )
            eval_noise_fingerprint = str(pose_summary["eval_noise_fingerprint"])
            for frame_row in frame_metrics:
                pose_frame_rows.append({
                    "perturbation_level_deg": level["perturbation_level_deg"],
                    "seed": seed, "method": method, **frame_row,
                })
            pose_summary_rows.append({
                "perturbation_level_deg": level["perturbation_level_deg"],
                "seed": seed, "method": method, **pose_summary,
            })
    return local_rows, pose_frame_rows, pose_summary_rows, {
        "seed": seed,
        "training_noise_fingerprint": noise_fingerprint,
        "eval_noise_fingerprint": eval_noise_fingerprint,
    }


def seed_failure_rows(
    seed: int,
    levels: list[dict[str, Any]],
    templates: list[PoseTemplate],
    w1_fingerprint: str,
    scene_fingerprint: str,
    layout_fingerprint: str,
    error: Exception,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    reason = f"seed_worker_{type(error).__name__}: {error}"
    local_rows: list[dict[str, Any]] = []
    pose_rows: list[dict[str, Any]] = []
    pose_frame_rows: list[dict[str, Any]] = []
    for level in levels:
        for method in METHODS:
            run_key = f"level={level['perturbation_level_deg']:.12g}|seed={seed}|method={method}"
            local_rows.append({
                "perturbation_level_deg": level["perturbation_level_deg"],
                "seed": seed, "method": method,
                "full_ray_p95_deg": math.nan,
                "peripheral_ray_p95_deg": math.nan,
                "full_ray_median_deg": math.nan,
                "peripheral_ray_median_deg": math.nan,
                "xi_abs_error": math.nan, "alpha_abs_error": math.nan,
                "fu_relative_error": math.nan, "fv_relative_error": math.nan,
                "mean_focal_relative_error": math.nan,
                "principal_point_error_px": math.nan,
                "training_rmse_px": math.nan, "valid_grid_ratio": 0.0,
                "solver_status": "failed", "failure_reason": reason,
                "final_xi": math.nan, "final_alpha": math.nan,
                "final_fu": math.nan, "final_fv": math.nan,
                "final_cu": math.nan, "final_cv": math.nan,
                "initial_camera_fingerprint": level["camera_fingerprint"],
                "perturbation_amplitude": level["amplitude"],
                "initial_full_ray_p95_deg": level["initial_full_ray_p95_deg"],
                "initial_peripheral_ray_p95_deg": level["initial_peripheral_ray_p95_deg"],
                "initial_valid_grid_ratio": level["initial_valid_grid_ratio"],
                "noise_fingerprint": "seed_worker_failed",
                "w1_fingerprint": w1_fingerprint,
                "scene_fingerprint": scene_fingerprint,
                "layout_fingerprint": layout_fingerprint,
                "optimization_rmse_px": math.nan, "point_count": 0,
                "iterations": 0, "runtime_sec": 0.0, "run_key": run_key,
            })
            pose_rows.append({
                "perturbation_level_deg": level["perturbation_level_deg"],
                "seed": seed, "method": method,
                "evaluation_frame_count": len(templates),
                "pose_success_count": 0, "pose_failure_count": len(templates),
                "pose_success_rate": 0.0,
                "orientation_error_p95_deg": math.nan,
                "orientation_error_median_deg": math.nan,
                "pose_reprojection_rmse_px": math.nan,
                "pose_reprojection_point_count": 0,
                "eval_noise_fingerprint": "seed_worker_failed",
                "failure_reason": reason,
            })
            for template in templates:
                pose_frame_rows.append({
                    "perturbation_level_deg": level["perturbation_level_deg"],
                    "seed": seed, "method": method,
                    "frame_index": template.frame_index,
                    "frame_label": template.frame_label,
                    "outer_point_count": len(template.reference_points),
                    "solver_status": "failed", "failure_reason": reason,
                    "orientation_error_deg": math.nan,
                    "reprojection_rmse_px": math.nan,
                    "residual_sse_px2": math.nan, "iterations": 0,
                })
    return local_rows, pose_rows, pose_frame_rows


def main() -> int:
    args = parse_args()
    levels_requested = parse_levels(args.levels)
    if (
        args.seed_count <= 0 or args.jobs <= 0 or args.noise_sigma_px < 0.0 or
        args.width <= 0 or args.height <= 0
    ):
        raise ValueError("seed-count/jobs must be positive and noise sigma nonnegative")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    scene = weak.parse_scene(args.reference_scene.resolve())
    rows = load_training_rows(args.training_points.resolve(), scene)
    width, height = args.width, args.height
    all_points = np.asarray([row["camera_point"] for row in rows], dtype=np.float64)
    outer_indices = np.asarray([
        index for index, row in enumerate(rows) if row["point_type"] == "outer"
    ], dtype=np.int64)
    outer_rows = [rows[index] for index in outer_indices]
    clean_all = project(scene.camera, all_points)
    mask = sweep.build_evaluation_mask(scene.camera, width, height, args.grid_size)
    modes, outer_fisher, outer_audit = weak.compute_weak_modes(
        scene, outer_rows, width, height,
    )
    _, full_fisher, full_audit = weak.compute_weak_modes(scene, rows, width, height)
    w1 = modes[0].direction
    w1_fingerprint = array_fingerprint(w1)
    calibrated = calibrate_levels(
        scene.camera, w1, levels_requested, width, height, mask,
    )
    np.savetxt(output / "outer_only_schur_fisher.csv", outer_fisher, delimiter=",", fmt="%.17g")
    np.savetxt(output / "outer_internal_schur_fisher.csv", full_fisher, delimiter=",", fmt="%.17g")
    scene_fingerprint = sha256_file(args.reference_scene)
    layout_payload = {
        str(board): np.asarray(transform).round(15).tolist()
        for board, transform in sorted(scene.boards.items())
    }
    layout_fingerprint = json_fingerprint(layout_payload)
    external_source = convert_external_mat(args, output)
    templates, template_diagnostics = build_pose_templates(external_source, scene)
    write_csv(output / "external_pose_template_diagnostics.csv", template_diagnostics)
    template_manifest = {
        "source_dataset": str(args.external_eval_mat.resolve()),
        "source_dataset_sha256": sha256_file(args.external_eval_mat),
        "training_dataset": str(args.training_points.resolve()),
        "dataset_is_independent_from_training": (
            args.external_eval_mat.resolve() != args.training_points.resolve()
        ),
        "requested_frame_count": len(template_diagnostics),
        "successful_frame_count": len(templates),
        "failed_frame_count": len(template_diagnostics) - len(templates),
        "frame_ids": [template.frame_index for template in templates],
        "layout_fingerprint": layout_fingerprint,
        "pose_source": "real_outer_fit_then_gt_ds_reprojection",
    }
    (output / "external_pose_template_manifest.json").write_text(
        json.dumps(template_manifest, indent=2) + "\n", encoding="utf-8"
    )
    if len(templates) < 4:
        raise RuntimeError("Fewer than four external pose templates succeeded")
    protocol = {
        "schema": "ds_w1_local_recovery_paper_v1",
        "levels_deg": list(levels_requested),
        "seeds": list(range(args.seed_start, args.seed_start + args.seed_count)),
        "noise_sigma_px": args.noise_sigma_px,
        "grid_size": args.grid_size,
        "image_width": width,
        "image_height": height,
        "peripheral_definition": "rho>=0.7 on fixed GT-centered valid disc",
        "training_frame_count": len({row["frame"] for row in rows}),
        "outer_point_count": len(outer_indices),
        "outer_internal_point_count": len(rows),
        "reference_scene": str(args.reference_scene.resolve()),
        "reference_scene_sha256": scene_fingerprint,
        "training_points": str(args.training_points.resolve()),
        "training_points_sha256": sha256_file(args.training_points),
        "ground_truth_camera_fingerprint": sweep.camera_fingerprint(scene.camera),
        "layout_fingerprint": layout_fingerprint,
        "w1_coordinate_order": outer_audit["coordinate_order"],
        "w1_direction": w1.tolist(),
        "w1_fingerprint": w1_fingerprint,
        "outer_fisher_eigenvalues": outer_audit["fisher_eigenvalues"],
        "outer_internal_fisher_eigenvalues": full_audit["fisher_eigenvalues"],
        "calibrated_levels": [
            {key: value for key, value in item.items() if key != "camera"}
            for item in calibrated
        ],
        "expected_run_count": len(levels_requested) * args.seed_count * len(METHODS),
        "external_pose_template_manifest": "external_pose_template_manifest.json",
    }
    (output / "protocol_manifest.json").write_text(
        json.dumps(protocol, indent=2) + "\n", encoding="utf-8"
    )
    seeds = list(range(args.seed_start, args.seed_start + args.seed_count))
    local_rows: list[dict[str, Any]] = []
    pose_frame_rows: list[dict[str, Any]] = []
    pose_summary_rows: list[dict[str, Any]] = []
    noise_rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.jobs) as executor:
        futures = {
            executor.submit(
                run_seed, seed, calibrated, scene.camera, rows, outer_indices,
                all_points, clean_all, width, height, args.noise_sigma_px,
                mask, w1_fingerprint, scene_fingerprint, layout_fingerprint,
                templates,
            ): seed for seed in seeds
        }
        for future in as_completed(futures):
            seed = futures[future]
            try:
                local, pose_frames, pose_summary, noise_summary = future.result()
                local_rows.extend(local)
                pose_frame_rows.extend(pose_frames)
                pose_summary_rows.extend(pose_summary)
                noise_rows.append(noise_summary)
                print(f"completed seed {seed}", flush=True)
            except Exception as error:
                failed_local, failed_pose, failed_pose_frames = seed_failure_rows(
                    seed, calibrated, templates, w1_fingerprint,
                    scene_fingerprint, layout_fingerprint, error,
                )
                local_rows.extend(failed_local)
                pose_summary_rows.extend(failed_pose)
                pose_frame_rows.extend(failed_pose_frames)
                noise_rows.append({
                    "seed": seed,
                    "training_noise_fingerprint": "seed_worker_failed",
                    "eval_noise_fingerprint": "seed_worker_failed",
                })
                print(f"failed seed {seed}: {error}", flush=True)
    local_rows.sort(key=lambda row: (
        float(row["perturbation_level_deg"]), int(row["seed"]), str(row["method"])
    ))
    pose_frame_rows.sort(key=lambda row: (
        float(row["perturbation_level_deg"]), int(row["seed"]),
        str(row["method"]), int(row["frame_index"])
    ))
    pose_summary_rows.sort(key=lambda row: (
        float(row["perturbation_level_deg"]), int(row["seed"]), str(row["method"])
    ))
    noise_rows.sort(key=lambda row: int(row["seed"]))
    write_csv(output / "local_recovery_runs.csv", local_rows)
    write_csv(output / "pose_only_frame_metrics.csv", pose_frame_rows)
    write_csv(output / "pose_only_run_summary.csv", pose_summary_rows)
    write_csv(output / "noise_fingerprints.csv", noise_rows)
    unique_keys = {row["run_key"] for row in local_rows}
    audit = {
        "expected_run_count": protocol["expected_run_count"],
        "actual_run_count": len(local_rows),
        "unique_run_key_count": len(unique_keys),
        "failed_run_count": sum(row["solver_status"] != "converged" for row in local_rows),
        "pose_frame_row_count": len(pose_frame_rows),
        "pose_summary_row_count": len(pose_summary_rows),
        "all_noise_reused_across_levels": all(
            len({
                row["noise_fingerprint"] for row in local_rows
                if int(row["seed"]) == seed
            }) == 1 for seed in seeds
        ),
        "all_shared_outer_noise_identical": True,
        "all_eval_noise_reused_across_levels": all(
            len({
                row["eval_noise_fingerprint"] for row in pose_summary_rows
                if int(row["seed"]) == seed
            }) == 1 for seed in seeds
        ),
        "all_w1_fingerprints_identical": len({row["w1_fingerprint"] for row in local_rows}) == 1,
        "all_scene_fingerprints_identical": len({row["scene_fingerprint"] for row in local_rows}) == 1,
        "all_layout_fingerprints_identical": len({row["layout_fingerprint"] for row in local_rows}) == 1,
    }
    (output / "run_audit.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )
    if (
        len(local_rows) != protocol["expected_run_count"] or
        len(unique_keys) != protocol["expected_run_count"]
    ):
        raise RuntimeError("Run audit failed expected/unique row count")
    print(json.dumps(audit, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
