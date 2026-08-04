#!/usr/bin/env python3
"""Audit a frozen DS stereo-rectification experiment without changing calibration.

This diagnostic deliberately consumes the existing frozen frame manifest and
raw SIFT matches.  It separately checks the geometry implementation with
synthetic points and checks real-image correspondences with identity-preserved
AprilTag outer corners.  No result from this tool is used to select a frame,
calibrate a camera, or update an extrinsic transform.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DOWNSTREAM = Path(__file__).with_name("run_rectification_disparity_visualization.py")
DEFAULT_INPUT = ROOT / "paper_experiments/2026_07_20_stereo_rectification_disparity"
DEFAULT_OUTPUT = ROOT / "paper_experiments/2026_07_25_stereo_rectification_epipolar_audit"
DEFAULT_CONFIG = ROOT / "aslam_cv/aslam_cameras_april/config/example_apriltag_internal.yaml"
DEFAULT_DETECTOR = ROOT / "build/detect_apriltag_internal"


def load_downstream() -> Any:
    spec = importlib.util.spec_from_file_location("stereo_downstream", DOWNSTREAM)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {DOWNSTREAM}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


D = load_downstream()


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def percentile_summary(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return {"count": 0, "p50": math.nan, "p75": math.nan, "p90": math.nan,
                "p95": math.nan, "p99": math.nan, "mean": math.nan, "rmse": math.nan,
                "maximum": math.nan, "inlier_0_5": math.nan, "inlier_1": math.nan,
                "inlier_2": math.nan, "inlier_5": math.nan, "inlier_10": math.nan}
    return {
        "count": int(len(values)),
        "p50": float(np.percentile(values, 50)), "p75": float(np.percentile(values, 75)),
        "p90": float(np.percentile(values, 90)), "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)), "mean": float(np.mean(values)),
        "rmse": float(np.sqrt(np.mean(values * values))), "maximum": float(np.max(values)),
        "inlier_0_5": float(np.mean(values <= 0.5)), "inlier_1": float(np.mean(values <= 1.0)),
        "inlier_2": float(np.mean(values <= 2.0)), "inlier_5": float(np.mean(values <= 5.0)),
        "inlier_10": float(np.mean(values <= 10.0)),
    }


def parse_timestamp(path: Path) -> int:
    match = re.match(r"\d+_(?:left|right)_(\d+)_mono8\.(?:png|jpg|jpeg)$", path.name)
    if not match:
        raise RuntimeError(f"cannot parse timestamp from {path.name}")
    return int(match.group(1))


def system_transform_rows(system: Any) -> list[dict[str, Any]]:
    rotation = system.rotation_cam1_cam0
    translation = system.translation_cam1_cam0
    transform = np.eye(4)
    transform[:3, :3], transform[:3, 3] = rotation, translation
    inverse = np.linalg.inv(transform)
    return [{
        "method": system.name,
        "definition": "p_cam1 = R_cam1_from_cam0 p_cam0 + t_cam1_from_cam0",
        "T_cam1_cam0": json.dumps(transform.tolist()),
        "T_cam0_cam1": json.dumps(inverse.tolist()),
        "R_cam1_from_cam0": json.dumps(rotation.tolist()),
        "t_cam1_from_cam0": json.dumps(translation.tolist()),
        "baseline_m": float(np.linalg.norm(translation)),
        "det_R": float(np.linalg.det(rotation)),
        "orthogonality_fro": float(np.linalg.norm(rotation.T @ rotation - np.eye(3))),
    }]


def roundtrip_rows(system: Any, rng: np.random.Generator) -> list[dict[str, Any]]:
    rows = []
    for side, camera in (("left", system.left), ("right", system.right)):
        pixels = np.column_stack([
            rng.uniform(0.0, camera.width - 1.0, 30000),
            rng.uniform(0.0, camera.height - 1.0, 30000),
        ])
        rays, valid0 = D.ds_unproject(camera, pixels)
        reprojection, valid1 = D.ds_project(camera, rays)
        valid = valid0 & valid1
        errors = np.linalg.norm(reprojection[valid] - pixels[valid], axis=1)
        row = percentile_summary(errors)
        row.update({"method": system.name, "camera": side, "valid_count": int(np.count_nonzero(valid))})
        rows.append(row)
    return rows


def synthetic_rows(system: Any, spec: Any, rng: np.random.Generator) -> list[dict[str, Any]]:
    # Points are generated in cam0.  The loaded transform is explicitly cam0->cam1.
    rays = D.normalize(rng.normal(size=(200000, 3)))
    rays = rays[rays[:, 2] > 0.08]
    depths = rng.uniform(0.8, 6.0, len(rays))
    points0 = rays * depths[:, None]
    points1 = points0 @ system.rotation_cam1_cam0.T + system.translation_cam1_cam0
    pixels0, visible0 = D.ds_project(system.left, points0)
    pixels1, visible1 = D.ds_project(system.right, points1)
    rect0, rect1 = D.rectification_rotations(system)
    rect_pixels0, rect_visible0 = D.rectified_points(system.left, rect0, pixels0, spec)
    rect_pixels1, rect_visible1 = D.rectified_points(system.right, rect1, pixels1, spec)
    valid = visible0 & visible1 & rect_visible0 & rect_visible1
    errors = np.abs(rect_pixels0[valid, 1] - rect_pixels1[valid, 1])
    width, height = system.left.width, system.left.height
    rho = np.sqrt(((pixels0[valid, 0] - (width - 1.0) / 2.0) / (width / 2.0)) ** 2 +
                  ((pixels0[valid, 1] - (height - 1.0) / 2.0) / (height / 2.0)) ** 2)
    bins = [("central_rho_lt_0_35", rho < 0.35),
            ("middle_rho_0_35_to_0_65", (rho >= 0.35) & (rho < 0.65)),
            ("peripheral_rho_0_65_to_0_90", (rho >= 0.65) & (rho <= 0.90))]
    rows = []
    for label, mask in [("all", np.ones(len(errors), dtype=bool)), *bins]:
        row = percentile_summary(errors[mask])
        row.update({"method": system.name, "region": label, "generated_points": int(len(points0)),
                    "visible_rectified_points": int(np.count_nonzero(valid))})
        rows.append(row)
    return rows


def run_outer_detector(detector: Path, config: Path, image: Path, output: Path, support_camera_yaml: Path) -> dict[tuple[int, int], tuple[float, float]]:
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "outer_corner_results.csv"
    if not csv_path.is_file():
        command = [str(detector), "--image", str(image), "--config", str(config), "--output", str(output),
                   "--outer-distortion-experiment", "--outer-experiment-camera-yaml", str(support_camera_yaml),
                   "--no-debug-output"]
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    points: dict[tuple[int, int], tuple[float, float]] = {}
    for row in load_csv(csv_path):
        # Use only ordinary decoded, pre-camera-aware corners.  This keeps the
        # diagnostic correspondence set independent of the evaluated method.
        if row["source"] == "baseline_multiscale":
            points[(int(row["board_id"]), int(row["corner_index"]))] = (float(row["final_x"]), float(row["final_y"]))
    return points


def known_correspondences(pairs: list[Any], detector: Path, config: Path, support_camera_yaml: Path, work: Path) -> list[dict[str, Any]]:
    rows = []
    for pair in pairs:
        left = run_outer_detector(detector, config, pair.left_path, work / f"frame_{pair.frame_id:06d}" / "left", support_camera_yaml)
        right = run_outer_detector(detector, config, pair.right_path, work / f"frame_{pair.frame_id:06d}" / "right", support_camera_yaml)
        for board_id, corner_index in sorted(left.keys() & right.keys()):
            lx, ly = left[(board_id, corner_index)]
            rx, ry = right[(board_id, corner_index)]
            rows.append({"frame_id": pair.frame_id, "board_id": board_id, "corner_index": corner_index,
                         "u_left": lx, "v_left": ly, "u_right": rx, "v_right": ry,
                         "timestamp_delta_ns": parse_timestamp(pair.right_path) - parse_timestamp(pair.left_path)})
    return rows


def rectify_records(records: list[dict[str, Any]], system: Any, spec: Any, prefix: str) -> tuple[list[dict[str, Any]], dict[str, int]]:
    left = np.asarray([[float(row["u_left"]), float(row["v_left"])] for row in records], dtype=np.float64)
    right = np.asarray([[float(row["u_right"]), float(row["v_right"])] for row in records], dtype=np.float64)
    raw_rho = np.sqrt(((left[:, 0] - (system.left.width - 1.0) / 2.0) / (system.left.width / 2.0)) ** 2 +
                      ((left[:, 1] - (system.left.height - 1.0) / 2.0) / (system.left.height / 2.0)) ** 2)
    _, unproject_left = D.ds_unproject(system.left, left)
    _, unproject_right = D.ds_unproject(system.right, right)
    rect0, rect1 = D.rectification_rotations(system)
    rect_left, valid_left = D.rectified_points(system.left, rect0, left, spec)
    rect_right, valid_right = D.rectified_points(system.right, rect1, right, spec)
    valid = valid_left & valid_right
    _, _, map_left_valid = D.build_remap(system.left, rect0, spec)
    _, _, map_right_valid = D.build_remap(system.right, rect1, spec)
    common_map = np.zeros(len(records), dtype=bool)
    for index in np.flatnonzero(valid):
        xl, yl = np.rint(rect_left[index]).astype(int)
        xr, yr = np.rint(rect_right[index]).astype(int)
        if 0 <= xl < spec.width and 0 <= yl < spec.height and 0 <= xr < spec.width and 0 <= yr < spec.height:
            common_map[index] = bool(map_left_valid[yl, xl] and map_right_valid[yr, xr])
    output = []
    for index, source in enumerate(records):
        row = dict(source)
        row["record_index"] = int(source.get("match_index", index))
        row.update({"method": system.name, "source": prefix, "raw_left_rho": float(raw_rho[index]),
                    "u_left_rect": float(rect_left[index, 0]), "v_left_rect": float(rect_left[index, 1]),
                    "u_right_rect": float(rect_right[index, 0]), "v_right_rect": float(rect_right[index, 1]),
                    "horizontal_disparity_px": float(rect_left[index, 0] - rect_right[index, 0]),
                    "vertical_error_px": float(abs(rect_left[index, 1] - rect_right[index, 1])) if valid[index] else math.nan,
                    "valid": int(valid[index])})
        output.append(row)
    stages = {"initial_records": len(records), "valid_left_unprojection": int(np.count_nonzero(unproject_left)),
              "valid_right_unprojection": int(np.count_nonzero(unproject_right)),
              "inside_left_rectified": int(np.count_nonzero(valid_left)),
              "inside_right_rectified": int(np.count_nonzero(valid_right)),
              "inside_common_rectified": int(np.count_nonzero(valid)),
              "inside_common_valid_map": int(np.count_nonzero(common_map))}
    return output, stages


def essential_rotation_rows(records: list[dict[str, Any]], systems: list[Any]) -> list[dict[str, Any]]:
    rows = []
    for system in systems:
        for frame_id in sorted({int(row["frame_id"]) for row in records}):
            group = [row for row in records if int(row["frame_id"]) == frame_id]
            left = np.asarray([[float(row["u_left"]), float(row["v_left"])] for row in group])
            right = np.asarray([[float(row["u_right"]), float(row["v_right"])] for row in group])
            rays_left, valid_left = D.ds_unproject(system.left, left)
            rays_right, valid_right = D.ds_unproject(system.right, right)
            valid = valid_left & valid_right & (rays_left[:, 2] > 1e-8) & (rays_right[:, 2] > 1e-8)
            if np.count_nonzero(valid) < 5:
                continue
            normalized_left = rays_left[valid, :2] / rays_left[valid, 2:3]
            normalized_right = rays_right[valid, :2] / rays_right[valid, 2:3]
            essential, mask = cv2.findEssentialMat(normalized_left, normalized_right, 1.0, (0.0, 0.0), cv2.RANSAC, 0.999, 0.003)
            if essential is None:
                continue
            _, estimated_rotation, estimated_translation, pose_mask = cv2.recoverPose(essential, normalized_left, normalized_right)
            delta = estimated_rotation @ system.rotation_cam1_cam0.T
            angle_deg = math.degrees(math.acos(float(np.clip((np.trace(delta) - 1.0) / 2.0, -1.0, 1.0))))
            rows.append({"method": system.name, "frame_id": frame_id, "input_correspondence_count": int(np.count_nonzero(valid)),
                         "essential_ransac_inliers": int(np.count_nonzero(mask)), "recover_pose_inliers": int(np.count_nonzero(pose_mask)),
                         "rotation_delta_deg": angle_deg, "estimated_R_cam1_from_cam0": json.dumps(estimated_rotation.tolist()),
                         "estimated_t_direction_cam1": json.dumps(estimated_translation.reshape(3).tolist())})
    return rows


def transform_direction_rows(records: list[dict[str, Any]], systems: list[Any], spec: Any) -> list[dict[str, Any]]:
    rows = []
    for system in systems:
        inverse_rotation = system.rotation_cam1_cam0.T
        inverse_translation = -inverse_rotation @ system.translation_cam1_cam0
        hypotheses = [("loaded_cam1_from_cam0", system),
                      ("inverse_misread_as_cam1_from_cam0", D.StereoSystem(system.name, system.left, system.right, inverse_rotation, inverse_translation))]
        for name, candidate in hypotheses:
            rect0, rect1 = D.rectification_rotations(candidate)
            left = np.asarray([[float(row["u_left"]), float(row["v_left"])] for row in records])
            right = np.asarray([[float(row["u_right"]), float(row["v_right"])] for row in records])
            projected_left, valid_left = D.rectified_points(candidate.left, rect0, left, spec)
            projected_right, valid_right = D.rectified_points(candidate.right, rect1, right, spec)
            valid = valid_left & valid_right
            summary = percentile_summary(np.abs(projected_left[valid, 1] - projected_right[valid, 1]))
            summary.update({"method": system.name, "transform_hypothesis": name})
            rows.append(summary)
    return rows


def grouped_statistics(records: list[dict[str, Any]], label: str) -> list[dict[str, Any]]:
    rows = []
    for method in sorted({str(row["method"]) for row in records}):
        selected = [row for row in records if row["method"] == method and int(row["valid"]) == 1]
        regions = [("all", selected), ("central_rho_lt_0_35", [r for r in selected if float(r["raw_left_rho"]) < 0.35]),
                   ("middle_rho_0_35_to_0_65", [r for r in selected if 0.35 <= float(r["raw_left_rho"]) < 0.65]),
                   ("peripheral_rho_0_65_to_0_90", [r for r in selected if 0.65 <= float(r["raw_left_rho"]) <= 0.90])]
        for region, group in regions:
            row = percentile_summary(np.asarray([float(item["vertical_error_px"]) for item in group]))
            row.update({"source": label, "method": method, "group": region})
            rows.append(row)
        for frame_id in sorted({int(row["frame_id"]) for row in selected}):
            group = [row for row in selected if int(row["frame_id"]) == frame_id]
            row = percentile_summary(np.asarray([float(item["vertical_error_px"]) for item in group]))
            row.update({"source": label, "method": method, "group": f"frame_{frame_id:06d}"})
            rows.append(row)
        if label == "known_apriltag_outer_corners":
            for board_id in sorted({int(row["board_id"]) for row in selected}):
                group = [row for row in selected if int(row["board_id"]) == board_id]
                row = percentile_summary(np.asarray([float(item["vertical_error_px"]) for item in group]))
                row.update({"source": label, "method": method, "group": f"board_{board_id}"})
                rows.append(row)
    return rows


def make_plots(records: list[dict[str, Any]], output: Path) -> None:
    width, height, left, top, right, bottom = 1180, 700, 100, 72, 42, 100
    colors = {"Kalibr": (177, 124, 62), "Ours": (95, 95, 214)}  # BGR, blue/red print-safe.
    groups = {method: [row for row in records if row["method"] == method and int(row["valid"]) == 1]
              for method in ("Kalibr", "Ours")}
    def canvas(title: str, xlabel: str, ylabel: str) -> np.ndarray:
        image = np.full((height, width, 3), 255, dtype=np.uint8)
        cv2.putText(image, title, (left, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.86, (30, 30, 30), 2, cv2.LINE_AA)
        cv2.line(image, (left, height - bottom), (width - right, height - bottom), (40, 40, 40), 1, cv2.LINE_AA)
        cv2.line(image, (left, top), (left, height - bottom), (40, 40, 40), 1, cv2.LINE_AA)
        cv2.putText(image, xlabel, (width // 2 - 115, height - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (30, 30, 30), 1, cv2.LINE_AA)
        cv2.putText(image, ylabel, (12, top + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (30, 30, 30), 1, cv2.LINE_AA)
        return image
    plot_width, plot_height = width - left - right, height - top - bottom
    hist = canvas("Frozen SIFT match vertical-error distribution", "Rectified vertical error (px)", "Match count")
    bins = np.linspace(0.0, 100.0, 51)
    counts = {method: np.histogram([float(row["vertical_error_px"]) for row in group], bins=bins)[0] for method, group in groups.items()}
    max_count = max(int(np.max(value)) for value in counts.values())
    for tick in range(0, 101, 20):
        x = left + int(tick / 100.0 * plot_width)
        cv2.line(hist, (x, height - bottom), (x, height - bottom + 6), (40, 40, 40), 1)
        cv2.putText(hist, str(tick), (x - 10, height - bottom + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (50, 50, 50), 1, cv2.LINE_AA)
    for method, values in counts.items():
        points = []
        for index, value in enumerate(values):
            x = left + int((index + 0.5) / len(values) * plot_width)
            y = height - bottom - int(value / max(max_count, 1) * plot_height)
            points.append((x, y))
        cv2.polylines(hist, [np.asarray(points, dtype=np.int32)], False, colors[method], 2, cv2.LINE_AA)
        y = 52 if method == "Kalibr" else 77
        cv2.line(hist, (width - 215, y), (width - 185, y), colors[method], 3, cv2.LINE_AA)
        cv2.putText(hist, method, (width - 177, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 30, 30), 1, cv2.LINE_AA)
    cv2.imwrite(str(output / "epipolar_error_histogram.png"), hist)
    scatter = canvas("Frozen SIFT match error versus raw image radius", "Raw-left normalized radius rho", "Vertical error (px)")
    for tick in range(0, 101, 20):
        y = height - bottom - int(tick / 100.0 * plot_height)
        cv2.line(scatter, (left - 5, y), (left, y), (40, 40, 40), 1)
        cv2.putText(scatter, str(tick), (left - 42, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (50, 50, 50), 1, cv2.LINE_AA)
    for tick in np.arange(0.0, 1.01, 0.2):
        x = left + int(tick * plot_width)
        cv2.line(scatter, (x, height - bottom), (x, height - bottom + 6), (40, 40, 40), 1)
        cv2.putText(scatter, f"{tick:.1f}", (x - 11, height - bottom + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (50, 50, 50), 1, cv2.LINE_AA)
    for method, group in groups.items():
        for row in group:
            x = left + int(min(max(float(row["raw_left_rho"]), 0.0), 1.0) * plot_width)
            y = height - bottom - int(min(float(row["vertical_error_px"]), 100.0) / 100.0 * plot_height)
            cv2.circle(scatter, (x, y), 2, colors[method], -1, cv2.LINE_AA)
    cv2.imwrite(str(output / "epipolar_error_vs_radius.png"), scatter)


def make_match_contact_sheet(records: list[dict[str, Any]], pairs: list[Any], output: Path, largest: bool) -> None:
    method = "Ours"
    records = [row for row in records if row["method"] == method and int(row["valid"]) == 1]
    records.sort(key=lambda row: float(row["vertical_error_px"]), reverse=largest)
    selected = records[:50]
    pair_by_id = {pair.frame_id: pair for pair in pairs}
    tiles = []
    for row in selected:
        pair = pair_by_id[int(row["frame_id"])]
        left, right = cv2.imread(str(pair.left_path)), cv2.imread(str(pair.right_path))
        def crop(image: np.ndarray, x: float, y: float) -> np.ndarray:
            size = 180; x0, y0 = int(round(x)) - size // 2, int(round(y)) - size // 2
            patch = cv2.copyMakeBorder(image[max(0, y0):min(image.shape[0], y0 + size), max(0, x0):min(image.shape[1], x0 + size)], max(0, -y0), max(0, y0 + size - image.shape[0]), max(0, -x0), max(0, x0 + size - image.shape[1]), cv2.BORDER_CONSTANT)
            cv2.drawMarker(patch, (size // 2, size // 2), (0, 255, 255), cv2.MARKER_CROSS, 18, 1, cv2.LINE_AA)
            return patch
        tile = cv2.hconcat([crop(left, float(row["u_left"]), float(row["v_left"])), crop(right, float(row["u_right"]), float(row["v_right"]))])
        bar = np.full((24, tile.shape[1], 3), 245, dtype=np.uint8)
        cv2.putText(bar, f"f{int(row['frame_id'])} id={int(row['record_index'])} e={float(row['vertical_error_px']):.1f}px rho={float(row['raw_left_rho']):.2f} d={float(row['descriptor_distance']):.0f} vR={float(row['v_left_rect']):.0f}/{float(row['v_right_rect']):.0f}", (5, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (20, 20, 20), 1, cv2.LINE_AA)
        tiles.append(np.vstack([bar, tile]))
    rows = [cv2.hconcat(tiles[index:index + 5]) for index in range(0, len(tiles), 5)]
    cv2.imwrite(str(output / ("top_50_largest_epipolar_errors.png" if largest else "top_50_smallest_epipolar_errors.png")), cv2.vconcat(rows))


def make_vector_overlay(records: list[dict[str, Any]], pairs: list[Any], output: Path) -> None:
    pair_by_id = {pair.frame_id: pair for pair in pairs}
    panels = []
    for pair in pairs:
        image = cv2.imread(str(pair.left_path))
        image = cv2.resize(image, (752, 752), interpolation=cv2.INTER_AREA)
        group = [row for row in records if row["method"] == "Ours" and int(row["frame_id"]) == pair.frame_id and int(row["valid"]) == 1]
        for row in sorted(group, key=lambda item: float(item["vertical_error_px"]), reverse=True)[:70]:
            x = int(float(row["u_left"]) / 4512.0 * 752.0); y = int(float(row["v_left"]) / 4512.0 * 752.0)
            color = (0, 60, 255) if float(row["vertical_error_px"]) > 20.0 else (0, 210, 255)
            cv2.circle(image, (x, y), 3, color, -1, cv2.LINE_AA)
        cv2.putText(image, f"frame {pair.frame_id}: Ours frozen SIFT errors", (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(image, f"frame {pair.frame_id}: Ours frozen SIFT errors", (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (20, 20, 20), 1, cv2.LINE_AA)
        panels.append(image)
    cv2.imwrite(str(output / "epipolar_error_vector_overlay.png"), cv2.hconcat(panels))


def write_report(path: Path, sync_rows: list[dict[str, Any]], selected_pairs: list[Any], synthetic: list[dict[str, Any]], known_stats: list[dict[str, Any]], sift_stats: list[dict[str, Any]], essential_rows: list[dict[str, Any]], direction_rows: list[dict[str, Any]], ours_provenance: str) -> None:
    def select(rows: list[dict[str, Any]], method: str, group: str, source: str | None = None) -> dict[str, Any]:
        for row in rows:
            if row.get("method") == method and row.get("group", row.get("region")) == group and (source is None or row.get("source") == source):
                return row
        return {}
    selected_ids = [pair.frame_id for pair in selected_pairs]
    selected_deltas = ", ".join(f"f{pair.frame_id}={parse_timestamp(pair.right_path) - parse_timestamp(pair.left_path)} ns" for pair in selected_pairs)
    lines = ["# Rectified Epipolar Error Audit", "", "## Scope", "",
             "This audit does not alter Stage5, Stage6, intrinsics, stereo extrinsics, frozen frame selection, or SGBM.",
             f"Ours calibration artifacts: `{ours_provenance}`.",
             "The original SIFT result is retained; this directory only adds diagnostics.", "", "## Synchronization", ""]
    lines += [f"- {len(sync_rows)} strict timestamp-paired left/right images are available for audit. The selected IDs {selected_ids} are filename frame IDs, not ordinal indices.",
              f"- Exact timestamps among strict pairs: {sum(int(row['timestamp_delta_ns']) == 0 for row in sync_rows)}/{len(sync_rows)}. The current frozen selected deltas are {selected_deltas}.", "",
              "## Geometry sanity result", ""]
    for method in ("Kalibr", "Ours"):
        row = select(synthetic, method, "all")
        lines.append(f"- {method} synthetic 3D rectified vertical error: P95={row.get('p95', math.nan):.3e} px, max={row.get('maximum', math.nan):.3e} px.")
    lines += ["- Therefore the DS project/unproject implementation, loaded transform convention, and baseline-aligned rectification are self-consistent. This does not prove that the supplied physical stereo extrinsics match the test images, but it rules out a mathematical rectification-direction bug as the source of 20+ px errors.", "", "## Real correspondences", ""]
    for source, stats in (("frozen SIFT", sift_stats), ("same-ID AprilTag outer corners", known_stats)):
        lines.append(f"### {source}")
        for method in ("Kalibr", "Ours"):
            row = select(stats, method, "all", "known_apriltag_outer_corners" if source.startswith("same") else "frozen_sift")
            lines.append(f"- {method}: n={row.get('count', 0)}, P50={row.get('p50', math.nan):.3f}px, P95={row.get('p95', math.nan):.3f}px, Inlier@1px={100.0 * row.get('inlier_1', math.nan):.2f}%.")
    lines += ["", "## Relative-rotation audit from known corner identities", ""]
    for method in ("Kalibr", "Ours"):
        values = [float(row["rotation_delta_deg"]) for row in essential_rows if row["method"] == method]
        lines.append(f"- {method}: per-frame essential-matrix estimates differ from the supplied rotation by {min(values):.3f}-{max(values):.3f} deg (median {np.median(values):.3f} deg).")
    lines += ["", "## Transform-direction hypothesis", ""]
    for method in ("Kalibr", "Ours"):
        loaded = next(row for row in direction_rows if row["method"] == method and row["transform_hypothesis"] == "loaded_cam1_from_cam0")
        inverse = next(row for row in direction_rows if row["method"] == method and row["transform_hypothesis"] == "inverse_misread_as_cam1_from_cam0")
        lines.append(f"- {method}: known-corner P95 is {loaded['p95']:.3f}px with the documented `cam1<-cam0` transform and {inverse['p95']:.3f}px if its full inverse is incorrectly treated as `cam1<-cam0`.")
    lines += ["", "## Conclusion", "",
              "- Frozen SIFT matches remain a secondary visualization diagnostic; same-ID AprilTag corners are the identity-preserved geometry metric."]
    for method in ("Kalibr", "Ours"):
        known = select(known_stats, method, "all", "known_apriltag_outer_corners")
        p95 = float(known.get("p95", math.nan))
        inlier = 100.0 * float(known.get("inlier_1", math.nan))
        rotations = [float(row["rotation_delta_deg"]) for row in essential_rows if row["method"] == method]
        if math.isfinite(p95) and p95 <= 1.0:
            verdict = "is geometrically consistent with the frozen external sequence"
        elif math.isfinite(p95) and p95 <= 3.0:
            verdict = "is substantially improved but still has residual geometric mismatch"
        else:
            verdict = "does not jointly explain the frozen external correspondences"
        lines.append(
            f"- {method}: known-corner P95={p95:.3f}px and Inlier@1px={inlier:.2f}%; "
            f"the supplied calibration {verdict}. Relative-rotation mismatch is "
            f"{min(rotations):.3f}-{max(rotations):.3f} deg."
        )
    lines.append("- Treat this audit as a geometry-validation gate. A downstream figure is paper-eligible only after same-ID AprilTag errors and calibration-artifact provenance are both reported.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--detector", type=Path, default=DEFAULT_DETECTOR)
    parser.add_argument("--detector-config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--ours-left-intrinsics", type=Path, default=D.DEFAULT_OURS_LEFT)
    parser.add_argument("--ours-right-intrinsics", type=Path, default=D.DEFAULT_OURS_RIGHT)
    parser.add_argument("--ours-extrinsic", type=Path, default=D.DEFAULT_OURS_EXTRINSIC)
    parser.add_argument("--kalibr-camchain", type=Path, default=D.DEFAULT_KALIBR)
    parser.add_argument("--left-dir", type=Path, default=D.DEFAULT_LEFT_DIR)
    parser.add_argument("--right-dir", type=Path, default=D.DEFAULT_RIGHT_DIR)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    ours = D.load_ours_system(args.ours_left_intrinsics, args.ours_right_intrinsics, args.ours_extrinsic)
    kalibr = D.load_kalibr_system(args.kalibr_camchain)
    systems = [kalibr, ours]
    spec = D.RectificationSpec(2048, 1536, 120.0, 0, 192, 7)
    manifest_rows = load_csv(args.input / "frame_manifest.csv")
    pairs = [D.ImagePair(int(row["frame_id"]),
                         Path(row["left_image"]) if Path(row["left_image"]).is_absolute() else ROOT / row["left_image"],
                         Path(row["right_image"]) if Path(row["right_image"]).is_absolute() else ROOT / row["right_image"],
                         float(row["selection_score"]), float(row["texture_score"]), float(row["peripheral_score"]), float(row["line_score"]),
                         int(row.get("timestamp_delta_ns", 0)))
             for row in manifest_rows]
    raw_pairs = D.list_pairs(args.left_dir, args.right_dir, 1_000_000)
    sync_rows = []
    for pair in raw_pairs:
        left, right = pair.left_path, pair.right_path
        sync_rows.append({"frame_id": pair.frame_id, "left_image": str(left), "right_image": str(right),
                          "left_timestamp_ns": parse_timestamp(left), "right_timestamp_ns": parse_timestamp(right),
                          "timestamp_delta_ns": parse_timestamp(right) - parse_timestamp(left)})
    write_csv(args.output / "image_pair_audit.csv", sync_rows)
    write_csv(args.output / "extrinsic_audit.csv", [row for system in systems for row in system_transform_rows(system)])
    rng = np.random.default_rng(20260725)
    roundtrip = [row for system in systems for row in roundtrip_rows(system, rng)]
    synthetic = [row for system in systems for row in synthetic_rows(system, spec, rng)]
    write_csv(args.output / "ds_roundtrip_audit.csv", roundtrip)
    write_csv(args.output / "synthetic_rectification_audit.csv", synthetic)
    frozen_raw = [{key: (int(value) if key == "frame_id" else float(value)) for key, value in row.items()} for row in load_csv(args.input / "frozen_matches.csv")]
    descriptor_rows = []
    for frame_id in ["all", *[str(pair.frame_id) for pair in pairs]]:
        group = frozen_raw if frame_id == "all" else [row for row in frozen_raw if int(row["frame_id"]) == int(frame_id)]
        distances = np.asarray([float(row["descriptor_distance"]) for row in group])
        descriptor_rows.append({"frame_id": frame_id, "count": int(len(distances)), "distance_p50": float(np.percentile(distances, 50)),
                                "distance_p90": float(np.percentile(distances, 90)), "distance_p95": float(np.percentile(distances, 95)),
                                "distance_max": float(np.max(distances)), "lowe_ratio_status": "not_recorded_in_frozen_matches_csv"})
    write_csv(args.output / "frozen_sift_descriptor_audit.csv", descriptor_rows)
    frozen_records = []
    masks = []
    for system in systems:
        rows, stages = rectify_records(frozen_raw, system, spec, "frozen_sift")
        frozen_records.extend(rows); masks.append({"method": system.name, "source": "frozen_sift", **stages})
    write_csv(args.output / "frozen_sift_rectified_audit.csv", frozen_records)
    write_csv(args.output / "match_mask_audit.csv", masks)
    sift_stats = grouped_statistics(frozen_records, "frozen_sift")
    write_csv(args.output / "frozen_sift_statistics.csv", sift_stats)
    if not args.detector.is_file():
        raise RuntimeError(f"missing detector executable: {args.detector}; build target detect_apriltag_internal first")
    known_raw = known_correspondences(pairs, args.detector, args.detector_config, args.ours_left_intrinsics, args.output / "detector_work")
    write_csv(args.output / "known_apriltag_outer_correspondences.csv", known_raw)
    known_records = []
    for system in systems:
        rows, stages = rectify_records(known_raw, system, spec, "known_apriltag_outer_corners")
        known_records.extend(rows); masks.append({"method": system.name, "source": "known_apriltag_outer_corners", **stages})
    write_csv(args.output / "known_apriltag_rectified_audit.csv", known_records)
    write_csv(args.output / "match_mask_audit.csv", masks)
    known_stats = grouped_statistics(known_records, "known_apriltag_outer_corners")
    write_csv(args.output / "known_apriltag_statistics.csv", known_stats)
    essential = essential_rotation_rows(known_raw, systems)
    write_csv(args.output / "known_apriltag_essential_rotation_audit.csv", essential)
    direction = transform_direction_rows(known_raw, systems, spec)
    write_csv(args.output / "transform_direction_hypothesis_audit.csv", direction)
    disparity_rows = []
    for source, records in (("frozen_sift", frozen_records), ("known_apriltag_outer_corners", known_records)):
        for method in ("Kalibr", "Ours"):
            values = np.asarray([float(row["horizontal_disparity_px"]) for row in records if row["source"] == source and row["method"] == method and int(row["valid"]) == 1])
            disparity_rows.append({"source": source, "method": method, "count": int(len(values)), "p5": float(np.percentile(values, 5)), "p50": float(np.percentile(values, 50)), "p95": float(np.percentile(values, 95)), "negative_ratio": float(np.mean(values < 0.0)), "outside_0_192_ratio": float(np.mean((values < 0.0) | (values >= 192.0)))})
    write_csv(args.output / "horizontal_disparity_audit.csv", disparity_rows)
    make_plots(frozen_records, args.output)
    make_match_contact_sheet(frozen_records, pairs, args.output, largest=True)
    make_match_contact_sheet(frozen_records, pairs, args.output, largest=False)
    make_vector_overlay(frozen_records, pairs, args.output)
    write_report(args.output / "diagnostic_report.md", sync_rows, pairs, synthetic, known_stats, sift_stats, essential, direction, str(args.ours_extrinsic.resolve()))
    print(f"output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
