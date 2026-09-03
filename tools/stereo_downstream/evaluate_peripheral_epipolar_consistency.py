#!/usr/bin/env python3
"""Evaluate frozen stereo feature matches in ray space across polar regions.

The script never uses a test correspondence to calibrate or refit either
system.  It freezes mutual SIFT matches from the raw synchronized images,
then evaluates each complete stereo calibration independently with the same
ray-space epipolar-angle formula and the same sensor-coordinate polar bands.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DOWNSTREAM = Path(__file__).with_name("run_rectification_disparity_visualization.py")
DEFAULT_BUNDLE = ROOT / (
    "intrintic/catalog/current_baseline/stereo_bundles/"
    "2026_08_02_stage6_selfcontained_ds_1444190_stride3"
)
DEFAULT_KALIBR = ROOT / "config/stereo_4_2-3-camchain.yaml"
DEFAULT_TEST_LEFT = ROOT / "image/mid_far_dataset/stereo_dataset_20260430_144928/left"
DEFAULT_TEST_RIGHT = ROOT / "image/mid_far_dataset/stereo_dataset_20260430_144928/right"
DEFAULT_CALIB_LEFT = ROOT / "image/datatset_5_1/stereo_dataset_20260430_1444190-clear/left"
DEFAULT_CALIB_RIGHT = ROOT / "image/datatset_5_1/stereo_dataset_20260430_1444190-clear/right"


def load_downstream() -> Any:
    spec = importlib.util.spec_from_file_location("stereo_downstream", DOWNSTREAM)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {DOWNSTREAM}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


D = load_downstream()

REGIONS = (
    ("central_0_30", 0.0, 30.0, 15.0),
    ("middle_30_60", 30.0, 60.0, 45.0),
    ("peripheral_60_80", 60.0, 80.0, 70.0),
)


def fail(message: str) -> None:
    raise RuntimeError(message)


def require_selection_residual(actual: str, required: str) -> None:
    """Keep a pixel-refined bundle out of the angular-refinement protocol."""
    if required and actual != required:
        fail(
            "Ours bundle does not satisfy the requested final residual mode: "
            f"got {actual!r}, expected {required!r}"
        )


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    materialized = list(rows)
    if not materialized:
        fail(f"refusing to write empty CSV: {path.name}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(materialized[0]))
        writer.writeheader()
        writer.writerows(materialized)


def percentile(values: np.ndarray, q: float) -> float:
    return float(np.percentile(values, q)) if len(values) else math.nan


def sensor_polar_proxy_deg(u: np.ndarray, v: np.ndarray, width: int, height: int) -> np.ndarray:
    """A method-independent equidistant polar proxy used only for fixed bins."""
    rho = np.sqrt(
        ((u - (width - 1.0) / 2.0) / (width / 2.0)) ** 2
        + ((v - (height - 1.0) / 2.0) / (height / 2.0)) ** 2
    )
    return 90.0 * rho


def region_name(theta_proxy_deg: float) -> str:
    for name, lower, upper, _ in REGIONS:
        if lower <= theta_proxy_deg < upper:
            return name
    return "outside_0_80"


def ensure_independent_test(args: argparse.Namespace, pairs: Sequence[Any]) -> None:
    if args.left_dir.resolve() == args.calibration_left_dir.resolve():
        fail("test left directory equals calibration left directory")
    if args.right_dir.resolve() == args.calibration_right_dir.resolve():
        fail("test right directory equals calibration right directory")
    calibration_paths = set(args.calibration_left_dir.glob("*")) | set(args.calibration_right_dir.glob("*"))
    if any(pair.left_path in calibration_paths or pair.right_path in calibration_paths for pair in pairs):
        fail("a selected test frame also appears in the calibration manifest")


def write_frame_manifest(pairs: Sequence[Any], output: Path) -> None:
    rows = [
        {
            "frame_id": pair.frame_id,
            "left_image": str(pair.left_path),
            "right_image": str(pair.right_path),
            "timestamp_delta_ns": pair.timestamp_delta_ns,
        }
        for pair in pairs
    ]
    write_csv(output / "frame_manifest.csv", rows)
    lines = [
        "selection_protocol: all_strict_timestamp_synchronized_test_pairs",
        "frame_count: " + str(len(pairs)),
        "frames:",
    ]
    for row in rows:
        lines.extend(
            [
                f"  - frame_id: {row['frame_id']}",
                f"    left_image: {row['left_image']}",
                f"    right_image: {row['right_image']}",
                f"    timestamp_delta_ns: {row['timestamp_delta_ns']}",
            ]
        )
    (output / "frame_manifest.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_manifest(path: Path) -> list[Any]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        fail(f"empty frame manifest: {path}")
    pairs = []
    for row in rows:
        pairs.append(
            D.ImagePair(
                int(row["frame_id"]),
                Path(row["left_image"]),
                Path(row["right_image"]),
                0.0,
                0.0,
                0.0,
                0.0,
                int(row["timestamp_delta_ns"]),
            )
        )
    return pairs


def frozen_high_quality_matches(
    pairs: Sequence[Any], output_path: Path, refresh: bool, ratio: float, max_per_cell: int
) -> list[dict[str, Any]]:
    if output_path.is_file() and not refresh:
        with output_path.open(newline="", encoding="utf-8") as handle:
            return [
                {
                    key: int(value) if key in {"frame_id", "match_rank"} else float(value)
                    for key, value in row.items()
                }
                for row in csv.DictReader(handle)
            ]
    sift = cv2.SIFT_create(nfeatures=8000, contrastThreshold=0.015)
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    records: list[dict[str, Any]] = []
    for pair in pairs:
        left = cv2.imread(str(pair.left_path), cv2.IMREAD_GRAYSCALE)
        right = cv2.imread(str(pair.right_path), cv2.IMREAD_GRAYSCALE)
        if left is None or right is None:
            fail(f"cannot read synchronized pair {pair.frame_id}")
        scale = min(1.0, 2048.0 / max(*left.shape, *right.shape))
        left_small = cv2.resize(left, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1.0 else left
        right_small = cv2.resize(right, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1.0 else right
        keypoints_left, descriptors_left = sift.detectAndCompute(left_small, None)
        keypoints_right, descriptors_right = sift.detectAndCompute(right_small, None)
        if descriptors_left is None or descriptors_right is None:
            continue
        forward = matcher.knnMatch(descriptors_left, descriptors_right, k=2)
        backward = matcher.knnMatch(descriptors_right, descriptors_left, k=2)
        forward_ok = {
            first.queryIdx: first
            for candidates in forward
            if len(candidates) == 2
            for first, second in [candidates]
            if first.distance < ratio * second.distance
        }
        backward_ok = {
            first.queryIdx: first
            for candidates in backward
            if len(candidates) == 2
            for first, second in [candidates]
            if first.distance < ratio * second.distance
        }
        candidates = []
        for left_index, match in forward_ok.items():
            reverse = backward_ok.get(match.trainIdx)
            if reverse is None or reverse.trainIdx != left_index:
                continue
            u_left, v_left = np.asarray(keypoints_left[left_index].pt) / scale
            u_right, v_right = np.asarray(keypoints_right[match.trainIdx].pt) / scale
            theta_proxy = sensor_polar_proxy_deg(
                np.asarray([u_left]), np.asarray([v_left]), left.shape[1], left.shape[0]
            )[0]
            candidates.append((float(match.distance), float(u_left), float(v_left), float(u_right), float(v_right), float(theta_proxy)))
        # Descriptor-ranked spatial thinning avoids central texture regions
        # dominating the frozen correspondence population.
        cell_counts: dict[tuple[int, int], int] = defaultdict(int)
        rank = 0
        for distance, u_left, v_left, u_right, v_right, theta_proxy in sorted(candidates):
            cell = (min(7, int(8.0 * u_left / left.shape[1])), min(7, int(8.0 * v_left / left.shape[0])))
            if cell_counts[cell] >= max_per_cell:
                continue
            cell_counts[cell] += 1
            rank += 1
            records.append(
                {
                    "frame_id": pair.frame_id,
                    "match_rank": rank,
                    "u_left": u_left,
                    "v_left": v_left,
                    "u_right": u_right,
                    "v_right": v_right,
                    "descriptor_distance": distance,
                    "polar_proxy_deg": theta_proxy,
                }
            )
    if not records:
        fail("no high-quality mutual SIFT matches were found")
    write_csv(output_path, records)
    return records


def symmetric_epipolar_angle_deg(system: Any, rays_left: np.ndarray, rays_right: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Point-to-epipolar-plane angular error, symmetrized over both cameras."""
    rotation, translation = system.rotation_cam1_cam0, system.translation_cam1_cam0
    right_prediction = rays_left @ rotation.T
    normal_right = np.cross(translation.reshape(1, 3), right_prediction)
    center_cam1_in_cam0 = -rotation.T @ translation
    left_prediction = rays_right @ rotation
    normal_left = np.cross(center_cam1_in_cam0.reshape(1, 3), left_prediction)
    right_norm = np.linalg.norm(normal_right, axis=1)
    left_norm = np.linalg.norm(normal_left, axis=1)
    valid = (right_norm > 1e-12) & (left_norm > 1e-12)
    normal_right[valid] /= right_norm[valid, None]
    normal_left[valid] /= left_norm[valid, None]
    sin_right = np.full(len(rays_left), np.nan)
    sin_left = np.full(len(rays_left), np.nan)
    sin_right[valid] = np.clip(np.abs(np.sum(normal_right[valid] * rays_right[valid], axis=1)), 0.0, 1.0)
    sin_left[valid] = np.clip(np.abs(np.sum(normal_left[valid] * rays_left[valid], axis=1)), 0.0, 1.0)
    error_deg = np.degrees(0.5 * (np.arcsin(sin_left) + np.arcsin(sin_right)))
    return error_deg, valid & np.isfinite(error_deg)


def method_records(matches: Sequence[dict[str, Any]], system: Any, spec: Any) -> list[dict[str, Any]]:
    by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in matches:
        by_frame[int(row["frame_id"])].append(row)
    rect_left, rect_right = D.rectification_rotations(system)
    records = []
    for frame_id, rows in sorted(by_frame.items()):
        left_pixels = np.asarray([[row["u_left"], row["v_left"]] for row in rows], dtype=np.float64)
        right_pixels = np.asarray([[row["u_right"], row["v_right"]] for row in rows], dtype=np.float64)
        rays_left, valid_left = D.ds_unproject(system.left, left_pixels)
        rays_right, valid_right = D.ds_unproject(system.right, right_pixels)
        angular_error, valid_epipolar = symmetric_epipolar_angle_deg(system, rays_left, rays_right)
        rectified_left, valid_rect_left = D.rectified_points(system.left, rect_left, left_pixels, spec)
        rectified_right, valid_rect_right = D.rectified_points(system.right, rect_right, right_pixels, spec)
        valid = valid_left & valid_right & valid_epipolar
        valid_vertical = valid_rect_left & valid_rect_right
        polar_left = np.degrees(np.arccos(np.clip(rays_left[:, 2], -1.0, 1.0)))
        polar_right = np.degrees(np.arccos(np.clip(rays_right[:, 2], -1.0, 1.0)))
        for index, raw in enumerate(rows):
            records.append(
                {
                    "method": system.name,
                    "frame_id": frame_id,
                    "match_rank": int(raw["match_rank"]),
                    "u_left": raw["u_left"],
                    "v_left": raw["v_left"],
                    "u_right": raw["u_right"],
                    "v_right": raw["v_right"],
                    "descriptor_distance": raw["descriptor_distance"],
                    "polar_proxy_deg": raw["polar_proxy_deg"],
                    "region": region_name(float(raw["polar_proxy_deg"])),
                    "left_polar_deg": float(polar_left[index]) if valid_left[index] else math.nan,
                    "right_polar_deg": float(polar_right[index]) if valid_right[index] else math.nan,
                    "angular_valid": int(valid[index]),
                    "epipolar_angular_error_deg": float(angular_error[index]) if valid[index] else math.nan,
                    "vertical_valid": int(valid_vertical[index]),
                    "vertical_disparity_px": float(abs(rectified_left[index, 1] - rectified_right[index, 1])) if valid_vertical[index] else math.nan,
                }
            )
    return records


def summarize(records: Sequence[dict[str, Any]], threshold_deg: float) -> list[dict[str, Any]]:
    rows = []
    for method in ("Kalibr", "Ours"):
        for region in ("all", *(name for name, _, _, _ in REGIONS)):
            selected = [
                row
                for row in records
                if row["method"] == method and (region == "all" or row["region"] == region)
            ]
            angular = np.asarray(
                [row["epipolar_angular_error_deg"] for row in selected if row["angular_valid"] == 1], dtype=np.float64
            )
            vertical = np.asarray(
                [row["vertical_disparity_px"] for row in selected if row["vertical_valid"] == 1], dtype=np.float64
            )
            rows.append(
                {
                    "method": method,
                    "region": region,
                    "frozen_match_count": len(selected),
                    "angular_valid_count": len(angular),
                    "median_epipolar_angular_error_deg": percentile(angular, 50),
                    "p95_epipolar_angular_error_deg": percentile(angular, 95),
                    "epipolar_inlier_ratio": float(np.mean(angular <= threshold_deg)) if len(angular) else math.nan,
                    "vertical_valid_count": len(vertical),
                    "vertical_disparity_p95_px": percentile(vertical, 95),
                }
            )
    return rows


def validate_metrics(metrics: Sequence[dict[str, Any]]) -> None:
    """Do not silently emit a paper table with undefined regional statistics."""
    required_regions = {"all", *(name for name, _, _, _ in REGIONS)}
    for method in ("Kalibr", "Ours"):
        available = {str(row["region"]) for row in metrics if row["method"] == method}
        if available != required_regions:
            fail(f"metric regions for {method} are incomplete: {sorted(available)}")
        for row in (item for item in metrics if item["method"] == method):
            region = str(row["region"])
            if int(row["frozen_match_count"]) == 0:
                fail(f"{method} has no frozen matches in {region}")
            if int(row["angular_valid_count"]) == 0:
                fail(f"{method} has no valid angular epipolar samples in {region}")
            if not math.isfinite(float(row["p95_epipolar_angular_error_deg"])):
                fail(f"{method} has undefined angular P95 in {region}")
            if int(row["vertical_valid_count"]) == 0:
                fail(f"{method} has no valid rectified samples in {region}")
            if not math.isfinite(float(row["vertical_disparity_p95_px"])):
                fail(f"{method} has undefined vertical-disparity P95 in {region}")


def make_curve(metrics: Sequence[dict[str, Any]], output: Path) -> Path:
    canvas = np.full((1000, 1600, 3), 255, dtype=np.uint8)
    cv2.putText(canvas, "", (0, 0), cv2.FONT_HERSHEY_SIMPLEX, 0.1, (0, 0, 0), 1)
    panels = (("median_epipolar_angular_error_deg", "Median epipolar angular error (deg)"), ("p95_epipolar_angular_error_deg", "P95 epipolar angular error (deg)"))
    colors = {"Kalibr": (45, 125, 215), "Ours": (198, 76, 48)}
    for panel_index, (column, ylabel) in enumerate(panels):
        x0, y0, width, height = 90 + panel_index * 775, 100, 650, 730
        cv2.line(canvas, (x0, y0), (x0, y0 + height), (30, 30, 30), 2)
        cv2.line(canvas, (x0, y0 + height), (x0 + width, y0 + height), (30, 30, 30), 2)
        values = [float(row[column]) for row in metrics if row["region"] != "all" and math.isfinite(float(row[column]))]
        y_max = max(0.02, max(values) * 1.15) if values else 1.0
        for tick in np.linspace(0.0, y_max, 5):
            y = int(y0 + height - height * tick / y_max)
            cv2.line(canvas, (x0, y), (x0 + width, y), (225, 225, 225), 1)
            cv2.putText(canvas, f"{tick:.2f}", (x0 - 66, y + 7), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (60, 60, 60), 1, cv2.LINE_AA)
        cv2.putText(canvas, ylabel, (x0, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (20, 20, 20), 2, cv2.LINE_AA)
        for label, _, _, center in REGIONS:
            x = int(x0 + width * (center - 10.0) / 70.0)
            cv2.putText(canvas, label.replace("_", " ").replace("central", "0-30").replace("middle", "30-60").replace("peripheral", "60-80"), (x - 50, y0 + height + 42), cv2.FONT_HERSHEY_SIMPLEX, 0.47, (50, 50, 50), 1, cv2.LINE_AA)
        for method in ("Kalibr", "Ours"):
            points = []
            for _, _, _, center in REGIONS:
                region = region_name(center)
                row = next(item for item in metrics if item["method"] == method and item["region"] == region)
                value = float(row[column])
                x = int(x0 + width * (center - 10.0) / 70.0)
                y = int(y0 + height - height * min(value, y_max) / y_max)
                points.append((x, y))
            cv2.polylines(canvas, [np.asarray(points, dtype=np.int32)], False, colors[method], 3, cv2.LINE_AA)
            for point in points:
                cv2.circle(canvas, point, 7, colors[method], -1, cv2.LINE_AA)
        for legend_index, method in enumerate(("Kalibr", "Ours")):
            lx, ly = x0 + 12, y0 + 30 + legend_index * 32
            cv2.line(canvas, (lx, ly), (lx + 30, ly), colors[method], 3, cv2.LINE_AA)
            cv2.putText(canvas, method, (lx + 42, ly + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (40, 40, 40), 1, cv2.LINE_AA)
    cv2.putText(canvas, "Fixed sensor polar proxy (deg)", (620, 930), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (30, 30, 30), 2, cv2.LINE_AA)
    path = output / "peripheral_epipolar_angular_curve.png"
    cv2.imwrite(str(path), canvas)
    return path


def error_color(error_deg: float) -> tuple[int, int, int]:
    normalized = int(round(255.0 * min(max(error_deg, 0.0), 2.0) / 2.0))
    value = cv2.applyColorMap(np.asarray([[normalized]], dtype=np.uint8), cv2.COLORMAP_TURBO)[0, 0]
    return int(value[0]), int(value[1]), int(value[2])


def make_overlay(records: Sequence[dict[str, Any]], pairs: Sequence[Any], output: Path) -> Path:
    counts: dict[int, int] = defaultdict(int)
    for row in records:
        if row["region"] == "peripheral_60_80":
            counts[int(row["frame_id"])] += 1
    frame_id = max(counts, key=lambda item: (counts[item], -item))
    pair = next(item for item in pairs if item.frame_id == frame_id)
    raw = cv2.imread(str(pair.left_path), cv2.IMREAD_COLOR)
    if raw is None:
        fail(f"cannot read display frame {frame_id}")
    peripheral = [row for row in records if int(row["frame_id"]) == frame_id and row["region"] == "peripheral_60_80"]
    anchor = min(peripheral, key=lambda row: (row["descriptor_distance"], row["match_rank"]))
    crop_half = max(180, min(raw.shape[:2]) // 7)
    center_x, center_y = int(anchor["u_left"]), int(anchor["v_left"])
    x0, x1 = max(0, center_x - crop_half), min(raw.shape[1], center_x + crop_half)
    y0, y1 = max(0, center_y - crop_half), min(raw.shape[0], center_y + crop_half)
    panels = []
    for method in ("Kalibr", "Ours"):
        image = raw.copy()
        for row in records:
            if row["method"] != method or int(row["frame_id"]) != frame_id or row["angular_valid"] != 1:
                continue
            point = (int(round(row["u_left"])), int(round(row["v_left"])))
            cv2.circle(image, point, 5, error_color(float(row["epipolar_angular_error_deg"])), -1, cv2.LINE_AA)
        cv2.rectangle(image, (x0, y0), (x1, y1), (255, 255, 255), 5, cv2.LINE_AA)
        inset = cv2.resize(image[y0:y1, x0:x1], (360, 360), interpolation=cv2.INTER_CUBIC)
        image[20:380, image.shape[1] - 380:image.shape[1] - 20] = inset
        cv2.rectangle(image, (image.shape[1] - 380, 20), (image.shape[1] - 20, 380), (255, 255, 255), 3, cv2.LINE_AA)
        header = np.full((64, image.shape[1], 3), 255, dtype=np.uint8)
        cv2.putText(header, f"{method}: epipolar angular error overlay", (18, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (20, 20, 20), 2, cv2.LINE_AA)
        panels.append(np.vstack([header, image]))
    result = np.hstack([panels[0], np.full((panels[0].shape[0], 8, 3), 235, dtype=np.uint8), panels[1]])
    path = output / "peripheral_epipolar_spatial_overlay.png"
    cv2.imwrite(str(path), result)
    return path


def write_pdf_from_png(png: Path) -> None:
    if shutil.which("pdflatex") is None:
        return
    tex = png.with_suffix(".tex")
    tex.write_text(
        "\\documentclass{standalone}\n\\usepackage{graphicx}\n\\begin{document}\n"
        f"\\includegraphics[width=0.99\\linewidth]{{{png.name}}}\n"
        "\\end{document}\n",
        encoding="utf-8",
    )
    subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", tex.name],
        cwd=png.parent,
        check=True,
        stdout=subprocess.DEVNULL,
    )


def write_table(path: Path, metrics: Sequence[dict[str, Any]]) -> None:
    rows = []
    for method in ("Kalibr", "Ours"):
        overall = next(row for row in metrics if row["method"] == method and row["region"] == "all")
        peripheral = next(row for row in metrics if row["method"] == method and row["region"] == "peripheral_60_80")
        rows.append(
            f"{method} & {overall['median_epipolar_angular_error_deg']:.3f} & "
            f"{overall['p95_epipolar_angular_error_deg']:.3f} & "
            f"{peripheral['median_epipolar_angular_error_deg']:.3f} & "
            f"{peripheral['p95_epipolar_angular_error_deg']:.3f} & "
            f"{100.0 * peripheral['epipolar_inlier_ratio']:.1f}\\% & "
            f"{peripheral['vertical_disparity_p95_px']:.3f} \\\\"
        )
    path.write_text(
        "\n".join(
            [
                "\\begin{tabular}{lrrrrrr}",
                "\\toprule",
                "Method & All Med. $\\downarrow$ & All P95 $\\downarrow$ & Periph. Med. $\\downarrow$ & Periph. P95 $\\downarrow$ & Periph. Inlier $\\uparrow$ & Rect. $|\\Delta v|$ P95 $\\downarrow$ \\\\",
                "\\midrule",
                *rows,
                "\\bottomrule",
                "\\end{tabular}",
                "",
            ]
        ),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ours-bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--kalibr-camchain", type=Path, default=DEFAULT_KALIBR)
    parser.add_argument("--left-dir", type=Path, default=DEFAULT_TEST_LEFT)
    parser.add_argument("--right-dir", type=Path, default=DEFAULT_TEST_RIGHT)
    parser.add_argument("--calibration-left-dir", type=Path, default=DEFAULT_CALIB_LEFT)
    parser.add_argument("--calibration-right-dir", type=Path, default=DEFAULT_CALIB_RIGHT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--test-scene-label", default="independent stereo test")
    parser.add_argument(
        "--timestamp-tolerance-ms",
        type=float,
        default=0.0,
        help="strict synchronization tolerance; default is exact filename timestamp matching",
    )
    parser.add_argument("--ratio-test", type=float, default=0.70)
    parser.add_argument("--max-matches-per-cell", type=int, default=8)
    parser.add_argument("--inlier-threshold-deg", type=float, default=0.5)
    parser.add_argument("--max-frames", type=int, default=0, help="0 evaluates every synchronized test pair")
    parser.add_argument("--refresh-freeze", action="store_true")
    parser.add_argument("--skip-pdf", action="store_true")
    parser.add_argument("--require-ours-selection-residual", default="spherical_tangent")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not (0.0 < args.ratio_test < 1.0):
        fail("ratio test must be in (0, 1)")
    if args.max_matches_per_cell < 1 or args.inlier_threshold_deg <= 0.0:
        fail("matching and inlier thresholds must be positive")
    bundle = args.ours_bundle.resolve()
    ours_left, ours_right, ours_extrinsic = (bundle / "left_intrinsics.yaml", bundle / "right_intrinsics.yaml", bundle / "stereo_extrinsic.yaml")
    manifest = bundle / "stereo_bundle_manifest.json"
    bundle_audit = D.verify_ours_bundle_manifest(manifest, ours_left, ours_right, ours_extrinsic)
    source_output = Path(str(bundle_audit["stage6_source_output"]))
    selection_summary = source_output / "stage6_persistent_incremental_selection_summary.txt"
    selection_mode = D.yaml_scalar(D.read_text(selection_summary), "selection_ba_residual_mode")
    require_selection_residual(selection_mode, args.require_ours_selection_residual)
    ours = D.load_ours_system(ours_left, ours_right, ours_extrinsic)
    kalibr = D.load_kalibr_system(args.kalibr_camchain)
    systems = [kalibr, ours]
    raw_pairs = D.list_pairs(args.left_dir, args.right_dir, int(round(args.timestamp_tolerance_ms * 1e6)))
    if args.max_frames > 0:
        raw_pairs = raw_pairs[:args.max_frames]
    ensure_independent_test(args, raw_pairs)
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / "frame_manifest.csv"
    if args.refresh_freeze or not manifest_path.is_file():
        write_frame_manifest(raw_pairs, args.output)
    pairs = load_manifest(manifest_path)
    matches = frozen_high_quality_matches(
        pairs,
        args.output / "frozen_matches.csv",
        args.refresh_freeze,
        args.ratio_test,
        args.max_matches_per_cell,
    )
    spec = D.RectificationSpec(2048, 1536, 120.0, 0, 192, 7)
    records = [row for system in systems for row in method_records(matches, system, spec)]
    write_csv(args.output / "per_match_epipolar_errors.csv", records)
    metrics = summarize(records, args.inlier_threshold_deg)
    validate_metrics(metrics)
    write_csv(args.output / "metrics_by_region.csv", metrics)
    write_table(args.output / "peripheral_epipolar_consistency_table.tex", metrics)
    curve = make_curve(metrics, args.output)
    overlay = make_overlay(records, pairs, args.output)
    if not args.skip_pdf:
        write_pdf_from_png(curve)
        write_pdf_from_png(overlay)
    protocol = {
        "protocol": "Peripheral Epipolar Consistency",
        "test_scene_label": args.test_scene_label,
        "test_left_dir": str(args.left_dir.resolve()),
        "test_right_dir": str(args.right_dir.resolve()),
        "calibration_left_dir": str(args.calibration_left_dir.resolve()),
        "calibration_right_dir": str(args.calibration_right_dir.resolve()),
        "test_frame_count": len(pairs),
        "strict_timestamp_pairing_tolerance_ms": args.timestamp_tolerance_ms,
        "selected_timestamp_deltas_ns": [pair.timestamp_delta_ns for pair in pairs],
        "matching": {
            "detector": "SIFT",
            "mutual_nearest_neighbor": True,
            "lowe_ratio": args.ratio_test,
            "spatial_thinning_grid": "8x8",
            "max_matches_per_cell": args.max_matches_per_cell,
            "calibration_independent": True,
        },
        "polar_bands": {
            "definition": "method-independent equidistant sensor polar proxy: theta_proxy_deg = 90 * rho",
            "bands": [{"name": name, "lower_deg": lower, "upper_deg": upper} for name, lower, upper, _ in REGIONS],
        },
        "epipolar_error": {
            "definition": "symmetric mean point-to-epipolar-plane bearing angle",
            "unit": "deg",
            "inlier_threshold_deg": args.inlier_threshold_deg,
            "no_test_time_extrinsic_refit": True,
        },
        "vertical_disparity": {"virtual_pinhole": {"width": spec.width, "height": spec.height, "hfov_deg": spec.hfov_deg}},
        "ours_bundle": bundle_audit,
        "ours_selection_ba_residual_mode": selection_mode,
        "kalibr_camchain": str(args.kalibr_camchain.resolve()),
    }
    (args.output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    print(f"output={args.output}")
    for row in metrics:
        if row["region"] == "peripheral_60_80":
            print(
                f"{row['method']}: peripheral median={row['median_epipolar_angular_error_deg']:.4f}deg "
                f"p95={row['p95_epipolar_angular_error_deg']:.4f}deg "
                f"inlier={100.0 * row['epipolar_inlier_ratio']:.2f}%"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
