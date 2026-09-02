#!/usr/bin/env python3
"""Stratify a frozen Stage5 holdout by image position and planar geometry.

This is a post-hoc audit: it never changes calibration, frontend detections, or
pose-refit metrics.  It derives a board pose from each method's already frozen
Outer4 observations, then uses the median pose across methods as a neutral
geometry label for the shared observations.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


METHOD_SUMMARY_PREFIX = {
    "ours": "our",
    "kalibr": "kalibr",
    "basalt": "reference_basalt",
    "tartancalib": "reference_tartancalib",
}
METHOD_ORDER = tuple(METHOD_SUMMARY_PREFIX)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--points", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--width", type=float, default=4512.0)
    parser.add_argument("--height", type=float, default=4512.0)
    return parser.parse_args()


def read_cameras(summary_path: Path) -> dict[str, tuple[float, ...]]:
    text = summary_path.read_text(encoding="utf-8")
    cameras: dict[str, tuple[float, ...]] = {}
    for method, prefix in METHOD_SUMMARY_PREFIX.items():
        intrinsics = re.search(
            rf"^{prefix}_camera_intrinsics_csv: (.+)$", text, flags=re.MULTILINE
        )
        distortion = re.search(
            rf"^{prefix}_camera_distortion_csv: (.+)$", text, flags=re.MULTILINE
        )
        if intrinsics is None or distortion is None:
            raise RuntimeError(f"Could not read KB camera parameters for {method}")
        values = tuple(float(v) for v in intrinsics.group(1).split(",")) + tuple(
            float(v) for v in distortion.group(1).split(",")
        )
        if len(values) != 8:
            raise RuntimeError(f"Expected KB8 parameters for {method}, got {values}")
        cameras[method] = values
    return cameras


def unproject_kb(u: float, v: float, params: tuple[float, ...]) -> tuple[float, float]:
    fu, fv, cu, cv, k1, k2, k3, k4 = params
    xd, yd = (u - cu) / fu, (v - cv) / fv
    theta_d = math.hypot(xd, yd)
    if theta_d < 1e-12:
        return 0.0, 0.0
    theta = theta_d
    for _ in range(20):
        theta2 = theta * theta
        theta4 = theta2 * theta2
        theta6 = theta4 * theta2
        theta8 = theta4 * theta4
        value = theta * (1.0 + k1 * theta2 + k2 * theta4 + k3 * theta6 + k4 * theta8)
        derivative = 1.0 + 3.0 * k1 * theta2 + 5.0 * k2 * theta4 + 7.0 * k3 * theta6 + 9.0 * k4 * theta8
        if abs(derivative) < 1e-12:
            break
        step = (value - theta_d) / derivative
        theta = max(0.0, theta - step)
        if abs(step) < 1e-12:
            break
    scale = math.tan(theta) / theta_d
    return xd * scale, yd * scale


def homography(object_xy: np.ndarray, image_xy: np.ndarray) -> np.ndarray:
    rows = []
    for (x, y), (u, v) in zip(object_xy, image_xy):
        rows.append((-x, -y, -1.0, 0.0, 0.0, 0.0, u * x, u * y, u))
        rows.append((0.0, 0.0, 0.0, -x, -y, -1.0, v * x, v * y, v))
    _, _, vt = np.linalg.svd(np.asarray(rows, dtype=float))
    return vt[-1].reshape(3, 3)


def pose_from_outer(rows: list[dict[str, str]], camera: tuple[float, ...]) -> tuple[float, float, float]:
    object_xy = np.asarray(
        [(float(row["target_x"]), float(row["target_y"])) for row in rows], dtype=float
    )
    image_xy = np.asarray(
        [unproject_kb(float(row["observed_x"]), float(row["observed_y"]), camera) for row in rows],
        dtype=float,
    )
    h = homography(object_xy, image_xy)
    if np.linalg.det(h[:, :2].T @ h[:, :2]) < 0.0:
        h = -h
    scale = 2.0 / (np.linalg.norm(h[:, 0]) + np.linalg.norm(h[:, 1]))
    r1, r2, translation = scale * h[:, 0], scale * h[:, 1], scale * h[:, 2]
    rotation_approx = np.column_stack((r1, r2, np.cross(r1, r2)))
    u, _, vt = np.linalg.svd(rotation_approx)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0.0:
        rotation[:, 2] *= -1.0
    if translation[2] < 0.0:
        translation *= -1.0
        rotation[:, :2] *= -1.0
    normal = rotation[:, 2]
    tilt_deg = math.degrees(math.acos(min(1.0, max(-1.0, abs(normal[2])))))
    return tilt_deg, float(np.linalg.norm(translation)), float(translation[2])


def quantile_edges(values: list[float]) -> tuple[float, float]:
    return tuple(float(v) for v in np.quantile(np.asarray(values), (1.0 / 3.0, 2.0 / 3.0)))


def interval_label(value: float, low: float, high: float, labels: tuple[str, str, str]) -> str:
    return labels[0] if value < low else labels[1] if value < high else labels[2]


def radial_bin(radius: float) -> str:
    if radius < 0.35:
        return "center"
    if radius < 0.65:
        return "mid"
    if radius < 0.85:
        return "edge"
    return "extreme"


def metrics(errors: list[float]) -> tuple[float, float, float, int]:
    sorted_errors = sorted(errors)
    count = len(sorted_errors)
    return (
        math.sqrt(sum(value * value for value in sorted_errors) / count),
        sorted_errors[math.ceil(0.95 * count) - 1],
        100.0 * sum(value <= 1.0 for value in sorted_errors) / count,
        count,
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cameras = read_cameras(args.summary)
    with args.points.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["evaluation_included"] == "1"]

    by_observation: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_observation[(row["method"], row["frame_index"], row["board_id"])].append(row)

    poses: dict[tuple[str, str, str], tuple[float, float, float]] = {}
    for key, observation_rows in by_observation.items():
        method = key[0]
        outer = [row for row in observation_rows if row["point_type"] == "outer"]
        if method in cameras and len(outer) >= 4:
            poses[key] = pose_from_outer(outer, cameras[method])

    shared_geometry: dict[tuple[str, str], tuple[float, float, float]] = {}
    for frame_index, board_id in {(key[1], key[2]) for key in by_observation}:
        estimates = [poses[(method, frame_index, board_id)] for method in METHOD_ORDER if (method, frame_index, board_id) in poses]
        if estimates:
            shared_geometry[(frame_index, board_id)] = tuple(float(v) for v in np.median(np.asarray(estimates), axis=0))

    tilts = [value[0] for value in shared_geometry.values()]
    distances = [value[1] for value in shared_geometry.values()]
    tilt_q1, tilt_q2 = quantile_edges(tilts)
    distance_q1, distance_q2 = quantile_edges(distances)

    geometry_rows: list[dict[str, object]] = []
    for (frame_index, board_id), (tilt, distance, depth) in sorted(shared_geometry.items(), key=lambda x: (int(x[0][0]), int(x[0][1]))):
        observed = by_observation[("ours", frame_index, board_id)]
        radii = [math.hypot((float(row["observed_x"]) - args.width / 2.0) / (args.width / 2.0), (float(row["observed_y"]) - args.height / 2.0) / (args.height / 2.0)) for row in observed]
        geometry_rows.append({
            "frame_index": frame_index,
            "frame_label": observed[0]["frame_label"],
            "board_id": board_id,
            "tilt_deg_median": tilt,
            "range_median": distance,
            "depth_median": depth,
            "tilt_bin": interval_label(tilt, tilt_q1, tilt_q2, ("low_tilt", "mid_tilt", "high_tilt")),
            "distance_bin": interval_label(distance, distance_q1, distance_q2, ("near", "mid_range", "far")),
            "mean_radius": float(np.mean(radii)),
            "max_radius": float(np.max(radii)),
            "board_position_bin": radial_bin(float(np.median(radii))),
            "point_count": len(observed),
            "outer_point_count": sum(row["point_type"] == "outer" for row in observed),
            "internal_point_count": sum(row["point_type"] == "internal" for row in observed),
        })

    geometry_lookup = {(row["frame_index"], row["board_id"]): row for row in geometry_rows}
    with (args.output_dir / "board_geometry.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(geometry_rows[0]))
        writer.writeheader()
        writer.writerows(geometry_rows)

    grouped_errors: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    paired: dict[tuple[str, str, str, str], dict[str, float]] = defaultdict(dict)
    for row in rows:
        geometry = geometry_lookup[(row["frame_index"], row["board_id"])]
        radius = math.hypot((float(row["observed_x"]) - args.width / 2.0) / (args.width / 2.0), (float(row["observed_y"]) - args.height / 2.0) / (args.height / 2.0))
        factors = {
            "point_radius": radial_bin(radius),
            "board_tilt": str(geometry["tilt_bin"]),
            "board_distance": str(geometry["distance_bin"]),
            "radius_x_tilt": f"{radial_bin(radius)}__{geometry['tilt_bin']}",
            "radius_x_distance": f"{radial_bin(radius)}__{geometry['distance_bin']}",
            "visibility": "full" if int(geometry["internal_point_count"]) >= 30 else "partial",
        }
        for factor, bin_name in factors.items():
            grouped_errors[(factor, bin_name, row["method"])].append(float(row["residual_norm"]))
        paired[(row["frame_index"], row["board_id"], row["point_id"], "point_radius:" + radial_bin(radius))][row["method"]] = float(row["residual_norm"])
        paired[(row["frame_index"], row["board_id"], row["point_id"], "radius_x_tilt:" + factors["radius_x_tilt"])][row["method"]] = float(row["residual_norm"])
        paired[(row["frame_index"], row["board_id"], row["point_id"], "radius_x_distance:" + factors["radius_x_distance"])][row["method"]] = float(row["residual_norm"])

    metric_rows: list[dict[str, object]] = []
    for (factor, bin_name, method), errors in sorted(grouped_errors.items()):
        rmse, p95, inlier, count = metrics(errors)
        metric_rows.append({"factor": factor, "bin": bin_name, "method": method, "rmse_px": rmse, "p95_px": p95, "inlier_at_1px_pct": inlier, "point_count": count})
    with (args.output_dir / "geometry_bin_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metric_rows[0]))
        writer.writeheader()
        writer.writerows(metric_rows)

    paired_bins: dict[tuple[str, str], list[float]] = defaultdict(list)
    for (*_, factor_bin), values in paired.items():
        if "ours" in values and "kalibr" in values:
            factor, bin_name = factor_bin.split(":", 1)
            paired_bins[(factor, bin_name)].append(values["ours"] - values["kalibr"])
    paired_rows = []
    for (factor, bin_name), deltas in sorted(paired_bins.items()):
        paired_rows.append({
            "factor": factor,
            "bin": bin_name,
            "mean_ours_minus_kalibr_px": float(np.mean(deltas)),
            "median_ours_minus_kalibr_px": float(np.median(deltas)),
            "ours_better_point_pct": 100.0 * sum(delta < 0.0 for delta in deltas) / len(deltas),
            "point_count": len(deltas),
        })
    with (args.output_dir / "paired_ours_vs_kalibr.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(paired_rows[0]))
        writer.writeheader()
        writer.writerows(paired_rows)

    summary_lines = [
        "# Frozen Holdout Geometry Stratification",
        "",
        "This is an offline audit of the existing frozen holdout; calibration and benchmark metrics were not rerun.",
        f"Board geometry: median of {', '.join(METHOD_ORDER)} Outer4 KB pose estimates.",
        f"Tilt terciles: {tilt_q1:.3f} deg, {tilt_q2:.3f} deg.",
        f"Range terciles: {distance_q1:.6f}, {distance_q2:.6f} (target-coordinate units).",
        "Point radius uses the fixed sensor center and half image width/height.",
        "",
        "See `geometry_bin_metrics.csv` for per-method RMSE/P95/Inlier@1px and `paired_ours_vs_kalibr.csv` for paired deltas.",
    ]
    (args.output_dir / "README.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print("\n".join(summary_lines))


if __name__ == "__main__":
    main()
