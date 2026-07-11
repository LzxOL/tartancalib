#!/usr/bin/env python3
"""Evaluate whether DS intrinsics are constrained beyond reprojection RMSE.

This script compares several Double Sphere cameras on the same Stage5 point
observations.  For each frame-board observation it fits a local T_camera_board
using only outer corners, then evaluates internal points and multi-board layout
consistency.  It also compares camera ray curves against a reference camera.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class Camera:
    name: str
    xi: float
    alpha: float
    fu: float
    fv: float
    cu: float
    cv: float


@dataclass(frozen=True)
class PointObs:
    split: str
    frame_index: int
    frame_label: str
    board_id: int
    point_id: int
    point_type: str
    observed: np.ndarray
    target: np.ndarray


CAMERAS = [
    Camera(
        "ours",
        -0.190556954466,
        0.617124422502,
        1175.65483979,
        1175.28837601,
        2242.87579592,
        2275.61281069,
    ),
    Camera(
        "babel_outer_internal",
        -0.1914,
        0.6175,
        1176.5589,
        1175.9243,
        2242.8966,
        2275.6331,
    ),
    Camera(
        "babel_outer_only",
        0.3401,
        0.8048,
        1943.4942,
        1943.0797,
        2242.9147,
        2275.6706,
    ),
]


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    default_run = (
        root
        / "result_may"
        / "stage5_baseline_babelcalib_export_20260707_1444190clear_ds_strict"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-run", type=Path, default=default_run)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_run / "ds_intrinsics_disambiguation_eval",
    )
    parser.add_argument("--image-width", type=int, default=4512)
    parser.add_argument("--image-height", type=int, default=4512)
    parser.add_argument("--ray-reference", default="ours")
    return parser.parse_args()


def read_points(path: Path, split: str) -> List[PointObs]:
    rows: List[PointObs] = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            try:
                rows.append(
                    PointObs(
                        split=split,
                        frame_index=int(float(row["frame_index"])),
                        frame_label=row["frame_label"],
                        board_id=int(float(row["board_id"])),
                        point_id=int(float(row["point_id"])),
                        point_type=row["point_type"],
                        observed=np.array(
                            [float(row["observed_x"]), float(row["observed_y"])],
                            dtype=np.float64,
                        ),
                        target=np.array(
                            [
                                float(row["target_x"]),
                                float(row["target_y"]),
                                float(row.get("target_z", 0.0)),
                            ],
                            dtype=np.float64,
                        ),
                    )
                )
            except Exception:
                continue
    return rows


def read_layout(path: Path) -> Dict[int, np.ndarray]:
    layout: Dict[int, np.ndarray] = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            board_id = int(float(row["board_id"]))
            vals: List[float]
            if None in row:
                vals = [float(row["T_reference_board_16"])]
                vals.extend(float(v) for v in row[None])
            else:
                cols = [k for k in row if k and k.startswith("T_reference_board")]
                vals = [float(row[k]) for k in cols]
            layout[board_id] = np.array(vals, dtype=np.float64).reshape(4, 4)
    return layout


def project_ds(cam: Camera, pts_cam: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    x = pts_cam[:, 0]
    y = pts_cam[:, 1]
    z = pts_cam[:, 2]
    d1 = np.sqrt(x * x + y * y + z * z)
    k = cam.xi * d1 + z
    d2 = np.sqrt(x * x + y * y + k * k)
    denom = cam.alpha * d2 + (1.0 - cam.alpha) * k
    valid = np.isfinite(denom) & (denom > 1e-12) & np.isfinite(d1)
    u = cam.fu * x / denom + cam.cu
    v = cam.fv * y / denom + cam.cv
    return np.stack([u, v], axis=1), valid


def unproject_ds(cam: Camera, pixels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mx = (pixels[:, 0] - cam.cu) / cam.fu
    my = (pixels[:, 1] - cam.cv) / cam.fv
    r2 = mx * mx + my * my
    inner = 1.0 - (2.0 * cam.alpha - 1.0) * r2
    valid = np.isfinite(inner) & (inner > 1e-12)
    mz = np.full_like(mx, np.nan, dtype=np.float64)
    denom = cam.alpha * np.sqrt(np.maximum(inner, 0.0)) + (1.0 - cam.alpha)
    good = valid & (np.abs(denom) > 1e-12)
    mz[good] = (1.0 - cam.alpha * cam.alpha * r2[good]) / denom[good]
    sqrt1 = np.sqrt(np.maximum(mz * mz + (1.0 - cam.xi * cam.xi) * r2, 0.0))
    k = (mz * cam.xi + sqrt1) / (mz * mz + r2)
    rays = np.stack([k * mx, k * my, k * mz - cam.xi], axis=1)
    norms = np.linalg.norm(rays, axis=1)
    valid = good & np.isfinite(norms) & (norms > 1e-12)
    rays[valid] /= norms[valid, None]
    return rays, valid


def transform_points(params: np.ndarray, pts: np.ndarray) -> np.ndarray:
    R = Rotation.from_rotvec(params[:3]).as_matrix()
    t = params[3:6]
    return (R @ pts.T).T + t


def pose_matrix(params: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = Rotation.from_rotvec(params[:3]).as_matrix()
    T[:3, 3] = params[3:6]
    return T


def initial_pose_guesses(cam: Camera, outer: Sequence[PointObs]) -> Iterable[np.ndarray]:
    obs = np.array([o.observed for o in outer], dtype=np.float64)
    tgt = np.array([o.target for o in outer], dtype=np.float64)
    center_pix = np.mean(obs, axis=0)
    norm_center = np.array(
        [(center_pix[0] - cam.cu) / cam.fu, (center_pix[1] - cam.cv) / cam.fv]
    )
    board_span = max(np.ptp(tgt[:, 0]), np.ptp(tgt[:, 1]), 1e-3)
    pixel_span = max(np.ptp(obs[:, 0]), np.ptp(obs[:, 1]), 1.0)
    z0 = max(0.05, min(3.0, 0.5 * (cam.fu + cam.fv) * board_span / pixel_span))
    center_obj = np.mean(tgt, axis=0)
    rotations = [
        np.zeros(3),
        np.array([0, 0, math.radians(90)]),
        np.array([0, 0, math.radians(-90)]),
    ]
    scales = [0.7, 1.0, 1.5]
    for r in rotations:
        R = Rotation.from_rotvec(r).as_matrix()
        for s in scales:
            z = z0 * s
            t = np.array([norm_center[0] * z, norm_center[1] * z, z]) - R @ center_obj
            yield np.r_[r, t]


def fit_pose_from_outer(
    cam: Camera, obs: Sequence[PointObs], seed: np.ndarray | None = None
) -> Tuple[bool, np.ndarray, float]:
    outer = [o for o in obs if o.point_type == "outer"]
    if len(outer) < 4:
        return False, np.zeros(6), math.nan
    pts = np.array([o.target for o in outer], dtype=np.float64)
    pix = np.array([o.observed for o in outer], dtype=np.float64)

    def residual(params: np.ndarray) -> np.ndarray:
        projected, valid = project_ds(cam, transform_points(params, pts))
        r = (projected - pix).reshape(-1)
        if not np.all(valid) or not np.all(np.isfinite(r)):
            r = np.where(np.isfinite(r), r, 1e3)
            for idx, ok in enumerate(valid):
                if not ok:
                    r[2 * idx : 2 * idx + 2] = 1e3
        return r

    best = None
    guesses: List[np.ndarray] = []
    if seed is not None and seed.shape == (6,) and np.all(np.isfinite(seed)):
        guesses.append(seed)
    guesses.extend(initial_pose_guesses(cam, outer))
    # Keep the first camera robust, but make later seed-initialized cameras fast.
    if seed is not None:
        guesses = guesses[:4]
    for guess in guesses:
        result = least_squares(
            residual,
            guess,
            loss="linear",
            max_nfev=90,
            xtol=1e-8,
            ftol=1e-8,
            gtol=1e-8,
        )
        rmse = float(np.sqrt(np.mean(residual(result.x) ** 2)))
        if best is None or rmse < best[0]:
            best = (rmse, result.x)
    if best is None:
        return False, np.zeros(6), math.nan
    return bool(np.isfinite(best[0]) and best[0] < 10.0), best[1], best[0]


def rmse(values: Sequence[float]) -> float:
    if not values:
        return math.nan
    arr = np.asarray(values, dtype=np.float64)
    return float(np.sqrt(np.mean(arr * arr)))


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return math.nan
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def evaluate_pose_metrics(
    cam: Camera,
    points: Sequence[PointObs],
    layout: Dict[int, np.ndarray],
    split_name: str,
    pose_seed_map: Dict[Tuple[str, str, int], np.ndarray],
) -> Tuple[dict, List[dict], List[dict]]:
    groups: Dict[Tuple[str, int], List[PointObs]] = defaultdict(list)
    for p in points:
        groups[(p.frame_label, p.board_id)].append(p)

    board_rows: List[dict] = []
    frame_pose_candidates: Dict[str, List[Tuple[int, np.ndarray]]] = defaultdict(list)
    all_res: List[float] = []
    outer_res: List[float] = []
    internal_res: List[float] = []
    pose_fit_outer_rmses: List[float] = []
    success_count = 0

    for (frame_label, board_id), obs in sorted(groups.items()):
        key = (split_name, frame_label, board_id)
        ok, params, pose_rmse = fit_pose_from_outer(
            cam, obs, seed=pose_seed_map.get(key)
        )
        if not ok:
            board_rows.append(
                {
                    "camera": cam.name,
                    "split": split_name,
                    "frame_label": frame_label,
                    "board_id": board_id,
                    "pose_success": 0,
                    "pose_fit_outer_rmse": pose_rmse,
                    "overall_rmse": math.nan,
                    "outer_rmse": math.nan,
                    "internal_rmse": math.nan,
                    "point_count": len(obs),
                    "outer_count": sum(o.point_type == "outer" for o in obs),
                    "internal_count": sum(o.point_type == "internal" for o in obs),
                }
            )
            continue
        success_count += 1
        pose_seed_map[key] = params
        pose_fit_outer_rmses.append(pose_rmse)
        T_cam_board = pose_matrix(params)
        if board_id in layout:
            frame_pose_candidates[frame_label].append(
                (board_id, T_cam_board @ np.linalg.inv(layout[board_id]))
            )
        pts = np.array([o.target for o in obs], dtype=np.float64)
        pix = np.array([o.observed for o in obs], dtype=np.float64)
        projected, valid = project_ds(cam, transform_points(params, pts))
        residuals = np.linalg.norm(projected - pix, axis=1)
        residuals = [
            float(r) for r, v in zip(residuals, valid) if v and np.isfinite(r)
        ]
        local_outer = [
            r
            for r, o, v in zip(np.linalg.norm(projected - pix, axis=1), obs, valid)
            if o.point_type == "outer" and v and np.isfinite(r)
        ]
        local_internal = [
            r
            for r, o, v in zip(np.linalg.norm(projected - pix, axis=1), obs, valid)
            if o.point_type == "internal" and v and np.isfinite(r)
        ]
        all_res.extend(residuals)
        outer_res.extend(float(v) for v in local_outer)
        internal_res.extend(float(v) for v in local_internal)
        board_rows.append(
            {
                "camera": cam.name,
                "split": split_name,
                "frame_label": frame_label,
                "board_id": board_id,
                "pose_success": 1,
                "pose_fit_outer_rmse": pose_rmse,
                "overall_rmse": rmse(residuals),
                "outer_rmse": rmse(local_outer),
                "internal_rmse": rmse(local_internal),
                "point_count": len(obs),
                "outer_count": sum(o.point_type == "outer" for o in obs),
                "internal_count": sum(o.point_type == "internal" for o in obs),
            }
        )

    layout_rows: List[dict] = []
    rot_drifts: List[float] = []
    trans_drifts: List[float] = []
    for frame_label, candidates in sorted(frame_pose_candidates.items()):
        if len(candidates) < 2:
            continue
        ref = candidates[0][1]
        for board_id, T in candidates[1:]:
            delta = np.linalg.inv(ref) @ T
            angle = float(
                np.degrees(Rotation.from_matrix(delta[:3, :3]).magnitude())
            )
            trans = float(np.linalg.norm(delta[:3, 3]) * 1000.0)
            rot_drifts.append(angle)
            trans_drifts.append(trans)
            layout_rows.append(
                {
                    "camera": cam.name,
                    "split": split_name,
                    "frame_label": frame_label,
                    "reference_board_id": candidates[0][0],
                    "board_id": board_id,
                    "layout_rot_drift_deg": angle,
                    "layout_trans_drift_mm": trans,
                }
            )

    summary = {
        "camera": cam.name,
        "split": split_name,
        "pose_success_count": success_count,
        "pose_attempt_count": len(groups),
        "pose_success_rate": success_count / len(groups) if groups else math.nan,
        "pose_fit_outer_rmse": rmse(pose_fit_outer_rmses),
        "evaluation_overall_rmse": rmse(all_res),
        "evaluation_outer_rmse": rmse(outer_res),
        "evaluation_internal_rmse": rmse(internal_res),
        "internal_median_px": percentile(internal_res, 50),
        "internal_p95_px": percentile(internal_res, 95),
        "layout_pair_count": len(layout_rows),
        "layout_rot_drift_mean_deg": float(np.mean(rot_drifts)) if rot_drifts else math.nan,
        "layout_rot_drift_p95_deg": percentile(rot_drifts, 95),
        "layout_trans_drift_mean_mm": float(np.mean(trans_drifts)) if trans_drifts else math.nan,
        "layout_trans_drift_p95_mm": percentile(trans_drifts, 95),
        "point_count": len(points),
        "outer_point_count": sum(p.point_type == "outer" for p in points),
        "internal_point_count": sum(p.point_type == "internal" for p in points),
    }
    return summary, board_rows, layout_rows


def ray_curve_metrics(
    cameras: Sequence[Camera],
    reference_name: str,
    width: int,
    height: int,
) -> Tuple[List[dict], List[dict]]:
    ref = next(c for c in cameras if c.name == reference_name)
    xs = np.linspace(0, width - 1, 101)
    ys = np.linspace(0, height - 1, 101)
    pixels = np.array([(x, y) for y in ys for x in xs], dtype=np.float64)
    ref_rays, ref_valid = unproject_ds(ref, pixels)
    polar = np.degrees(np.arccos(np.clip(ref_rays[:, 2], -1.0, 1.0)))
    buckets = [(0, 30), (30, 50), (50, 70), (70, 90), (90, 120)]
    rows: List[dict] = []
    samples: List[dict] = []
    for cam in cameras:
        rays, valid = unproject_ds(cam, pixels)
        both = ref_valid & valid
        dot = np.sum(ref_rays * rays, axis=1)
        angles = np.degrees(np.arccos(np.clip(dot, -1.0, 1.0)))
        for lo, hi in buckets:
            mask = both & (polar >= lo) & (polar < hi)
            vals = angles[mask]
            rows.append(
                {
                    "camera": cam.name,
                    "reference": reference_name,
                    "bucket_min_deg": lo,
                    "bucket_max_deg": hi,
                    "sample_count": int(vals.size),
                    "ray_angle_rmse_deg": rmse(vals.tolist()),
                    "ray_angle_median_deg": percentile(vals.tolist(), 50),
                    "ray_angle_p95_deg": percentile(vals.tolist(), 95),
                }
            )
        for idx in np.where(both)[0][:: max(1, int(np.sum(both) / 200))]:
            samples.append(
                {
                    "camera": cam.name,
                    "reference": reference_name,
                    "u": float(pixels[idx, 0]),
                    "v": float(pixels[idx, 1]),
                    "reference_polar_deg": float(polar[idx]),
                    "ray_angle_deg": float(angles[idx]),
                }
            )
    return rows, samples


def write_csv(path: Path, rows: Sequence[dict]) -> None:
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    source_run = args.source_run.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    train = read_points(source_run / "backend_training_points.csv", "backend")
    holdout = read_points(source_run / "backend_holdout_points.csv", "holdout")
    layout = read_layout(source_run / "backend_board_poses.csv")

    summaries: List[dict] = []
    board_rows: List[dict] = []
    layout_rows: List[dict] = []
    pose_seed_map: Dict[Tuple[str, str, int], np.ndarray] = {}
    for cam in CAMERAS:
        for split_name, pts in [("backend", train), ("holdout", holdout)]:
            summary, boards, layouts = evaluate_pose_metrics(
                cam, pts, layout, split_name, pose_seed_map
            )
            summaries.append(summary)
            board_rows.extend(boards)
            layout_rows.extend(layouts)

    ray_rows, ray_samples = ray_curve_metrics(
        CAMERAS, args.ray_reference, args.image_width, args.image_height
    )
    write_csv(output_dir / "outer_pose_internal_eval_summary.csv", summaries)
    write_csv(output_dir / "outer_pose_internal_eval_by_board.csv", board_rows)
    write_csv(output_dir / "multiboard_layout_drift_by_frame.csv", layout_rows)
    write_csv(output_dir / "ray_curve_delta_summary.csv", ray_rows)
    write_csv(output_dir / "ray_curve_delta_samples.csv", ray_samples)

    payload = {
        "source_run": str(source_run),
        "output_dir": str(output_dir),
        "cameras": [cam.__dict__ for cam in CAMERAS],
        "metrics": {
            "outer_pose_internal_eval": (
                "Fit each frame-board pose from outer corners only, then evaluate "
                "all observed points and internal points using the same pose."
            ),
            "multiboard_layout_drift": (
                "For each independently fitted T_camera_board, compute "
                "T_camera_reference = T_camera_board * inv(T_reference_board), "
                "then compare boards within the same frame."
            ),
            "ray_curve_delta": (
                "Unproject a dense image grid and compare angular ray difference "
                f"against {args.ray_reference}."
            ),
        },
    }
    (output_dir / "experiment_manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({"output_dir": str(output_dir), "summary_rows": summaries}, indent=2))


if __name__ == "__main__":
    main()
