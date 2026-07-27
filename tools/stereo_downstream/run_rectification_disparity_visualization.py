#!/usr/bin/env python3
"""Reproducible Kalibr-vs-Ours DS stereo rectification/disparity visualization.

This tool intentionally consumes completed calibration artifacts only. It does
not invoke Stage5/Stage6 and never uses either calibration while selecting
frames or extracting the frozen raw-image feature correspondences.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "paper_experiments/2026_07_25_stereo_rectification_disparity_timestamp_sync"
DEFAULT_OURS_LEFT = ROOT / "intrintic/catalog/canonical/left/ds/stereo-4-2-3__left__tartancalib__ds.yaml"
DEFAULT_OURS_RIGHT = ROOT / "intrintic/catalog/canonical/right/checkerboard-3-25__right/ds/checkerboard-3-25__right__tartancalib__ds.yaml"
DEFAULT_OURS_EXTRINSIC = ROOT / (
    "paper_experiments/2026_07_02_multiboard_calibration/data_only/"
    "05_ray_curve_and_reprojection/"
    "stage6_residual_compare_hybrid_highpolar_w1_nofinal_rerun2_20260625_"
    "1444190clear_to_144928/stereo_extrinsic.yaml"
)
DEFAULT_KALIBR = ROOT / "config/stereo_4_2-3-camchain.yaml"
DEFAULT_LEFT_DIR = ROOT / "image/mid_far_dataset/stereo_dataset_20260430_144928/left"
DEFAULT_RIGHT_DIR = ROOT / "image/mid_far_dataset/stereo_dataset_20260430_144928/right"
DEFAULT_CALIB_LEFT_DIR = ROOT / "image/datatset_5_1/stereo_dataset_20260430_1444190-clear/left"
DEFAULT_CALIB_RIGHT_DIR = ROOT / "image/datatset_5_1/stereo_dataset_20260430_1444190-clear/right"


@dataclass(frozen=True)
class DoubleSphereCamera:
    name: str
    xi: float
    alpha: float
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int


@dataclass(frozen=True)
class StereoSystem:
    name: str
    left: DoubleSphereCamera
    right: DoubleSphereCamera
    rotation_cam1_cam0: np.ndarray
    translation_cam1_cam0: np.ndarray


@dataclass(frozen=True)
class ImagePair:
    frame_id: int
    left_path: Path
    right_path: Path
    selection_score: float
    texture_score: float
    peripheral_score: float
    line_score: float
    timestamp_delta_ns: int = 0


@dataclass(frozen=True)
class RectificationSpec:
    width: int
    height: int
    hfov_deg: float
    min_disparity: int
    num_disparities: int
    block_size: int

    @property
    def focal(self) -> float:
        return self.width / (2.0 * math.tan(math.radians(self.hfov_deg) / 2.0))

    @property
    def cx(self) -> float:
        return (self.width - 1.0) / 2.0

    @property
    def cy(self) -> float:
        return (self.height - 1.0) / 2.0


def fail(message: str) -> None:
    raise RuntimeError(message)


def read_text(path: Path) -> str:
    if not path.is_file():
        fail(f"missing input file: {path}")
    return path.read_text(encoding="utf-8")


def yaml_block(text: str, key: str) -> str:
    match = re.search(rf"(?ms)^{re.escape(key)}:\s*$\n(.*?)(?=^cam\d+:|\Z)", text)
    if not match:
        fail(f"could not find YAML block '{key}'")
    return match.group(1)


def yaml_scalar(text: str, key: str) -> str:
    match = re.search(rf"(?m)^\s*{re.escape(key)}:\s*([^#\n]+)", text)
    if not match:
        fail(f"could not find YAML scalar '{key}'")
    return match.group(1).strip().strip('"').strip("'")


def yaml_vector(text: str, key: str, count: int) -> np.ndarray:
    match = re.search(rf"(?m)^\s*{re.escape(key)}:\s*\[([^\]]+)\]", text)
    if not match:
        fail(f"could not find YAML vector '{key}'")
    values = [float(value.strip()) for value in match.group(1).split(",")]
    if len(values) != count:
        fail(f"YAML vector '{key}' has {len(values)} values, expected {count}")
    return np.asarray(values, dtype=np.float64)


def yaml_matrix(text: str, key: str, rows: int) -> np.ndarray:
    match = re.search(rf"(?ms)^\s*{re.escape(key)}:\s*$\n((?:\s*-\s*\[[^\n]+\]\s*\n?){{{rows}}})", text)
    if not match:
        fail(f"could not find YAML matrix '{key}'")
    result = []
    for row in re.findall(r"\[([^\]]+)\]", match.group(1)):
        result.append([float(value.strip()) for value in row.split(",")])
    matrix = np.asarray(result, dtype=np.float64)
    if matrix.shape != (rows, rows):
        fail(f"YAML matrix '{key}' has shape {matrix.shape}, expected {(rows, rows)}")
    return matrix


def load_ds_camera(path: Path, name: str, block_name: str = "cam0") -> DoubleSphereCamera:
    block = yaml_block(read_text(path), block_name)
    model = yaml_scalar(block, "camera_model").lower()
    distortion = yaml_scalar(block, "distortion_model").lower()
    if model != "ds" or distortion != "none":
        fail(f"{path} is {model}/{distortion}; this downstream protocol supports ds/none only")
    values = yaml_vector(block, "intrinsics", 6)
    resolution = yaml_vector(block, "resolution", 2).astype(int)
    camera = DoubleSphereCamera(name, *values, int(resolution[0]), int(resolution[1]))
    if not all(math.isfinite(value) for value in values) or camera.fx <= 0.0 or camera.fy <= 0.0:
        fail(f"invalid DS intrinsics in {path}")
    return camera


def load_kalibr_system(path: Path) -> StereoSystem:
    text = read_text(path)
    cam0 = load_ds_camera(path, "Kalibr left", "cam0")
    cam1 = load_ds_camera(path, "Kalibr right", "cam1")
    cam1_block = yaml_block(text, "cam1")
    transform = yaml_matrix(cam1_block, "T_cn_cnm1", 4)
    if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-9):
        fail(f"invalid homogeneous bottom row in {path}")
    return StereoSystem("Kalibr", cam0, cam1, transform[:3, :3], transform[:3, 3])


def load_ours_system(left_path: Path, right_path: Path, extrinsic_path: Path) -> StereoSystem:
    text = read_text(extrinsic_path)
    left = load_ds_camera(left_path, "Ours left")
    right = load_ds_camera(right_path, "Ours right")
    rotation = yaml_matrix(text, "rotation_matrix", 3)
    translation = yaml_vector(text, "translation_xyz", 3)
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=2e-4):
        fail(f"Ours extrinsic rotation is not orthonormal: {extrinsic_path}")
    if np.linalg.det(rotation) <= 0.0 or np.linalg.norm(translation) <= 1e-5:
        fail(f"invalid Ours extrinsic transform: {extrinsic_path}")
    return StereoSystem("Ours", left, right, rotation, translation)


def normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return np.divide(vectors, norms, out=np.zeros_like(vectors), where=norms > 1e-12)


def ds_fov_parameter(camera: DoubleSphereCamera) -> float:
    alpha = camera.alpha
    temp = alpha / (1.0 - alpha) if alpha <= 0.5 else (1.0 - alpha) / alpha
    denominator = math.sqrt(2.0 * temp * camera.xi + camera.xi * camera.xi + 1.0)
    return (temp + camera.xi) / denominator


def ds_project(camera: DoubleSphereCamera, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Equivalent to DoubleSphereProjection::euclideanToKeypoint for ds/none."""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    d1 = np.sqrt(x * x + y * y + z * z)
    k = camera.xi * d1 + z
    d2 = np.sqrt(x * x + y * y + k * k)
    denominator = camera.alpha * d2 + (1.0 - camera.alpha) * k
    valid = (d1 > 1e-12) & (z > -ds_fov_parameter(camera) * d1) & (np.abs(denominator) > 1e-12)
    pixels = np.full((len(points), 2), np.nan, dtype=np.float64)
    pixels[valid, 0] = camera.fx * x[valid] / denominator[valid] + camera.cx
    pixels[valid, 1] = camera.fy * y[valid] / denominator[valid] + camera.cy
    valid &= np.isfinite(pixels).all(axis=1)
    valid &= (pixels[:, 0] >= 0.0) & (pixels[:, 0] < camera.width)
    valid &= (pixels[:, 1] >= 0.0) & (pixels[:, 1] < camera.height)
    return pixels, valid


def ds_unproject(camera: DoubleSphereCamera, pixels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Equivalent to DoubleSphereProjection::keypointToEuclidean for ds/none."""
    pixels = np.asarray(pixels, dtype=np.float64).reshape(-1, 2)
    mx = (pixels[:, 0] - camera.cx) / camera.fx
    my = (pixels[:, 1] - camera.cy) / camera.fy
    r2 = mx * mx + my * my
    valid = np.isfinite(r2)
    if camera.alpha > 0.5:
        valid &= r2 <= 1.0 / (2.0 * camera.alpha - 1.0)
    radicand = 1.0 - (2.0 * camera.alpha - 1.0) * r2
    valid &= radicand >= 0.0
    mz = np.full_like(r2, np.nan)
    denominator = camera.alpha * np.sqrt(np.maximum(radicand, 0.0)) + 1.0 - camera.alpha
    valid &= np.abs(denominator) > 1e-12
    mz[valid] = (1.0 - camera.alpha * camera.alpha * r2[valid]) / denominator[valid]
    radicand2 = mz * mz + (1.0 - camera.xi * camera.xi) * r2
    valid &= radicand2 >= 0.0
    norm = mz * mz + r2
    valid &= norm > 1e-12
    k = np.full_like(r2, np.nan)
    k[valid] = (mz[valid] * camera.xi + np.sqrt(np.maximum(radicand2[valid], 0.0))) / norm[valid]
    rays = np.column_stack([k * mx, k * my, k * mz - camera.xi])
    valid &= np.isfinite(rays).all(axis=1)
    rays = normalize(rays)
    return rays, valid


def rectification_rotations(system: StereoSystem) -> tuple[np.ndarray, np.ndarray]:
    """Build a common baseline-aligned rectified frame from a stereo transform."""
    center_cam1_in_cam0 = -system.rotation_cam1_cam0.T @ system.translation_cam1_cam0
    ex = normalize(center_cam1_in_cam0.reshape(1, 3))[0]
    forward = np.array([0.0, 0.0, 1.0]) + system.rotation_cam1_cam0.T @ np.array([0.0, 0.0, 1.0])
    ey = np.cross(forward, ex)
    if np.linalg.norm(ey) < 1e-8:
        ey = np.cross(np.array([0.0, 1.0, 0.0]), ex)
    ey = normalize(ey.reshape(1, 3))[0]
    ez = normalize(np.cross(ex, ey).reshape(1, 3))[0]
    rect_from_cam0 = np.vstack([ex, ey, ez])
    rect_from_cam1 = rect_from_cam0 @ system.rotation_cam1_cam0.T
    return rect_from_cam0, rect_from_cam1


def virtual_rays(spec: RectificationSpec) -> np.ndarray:
    ys, xs = np.indices((spec.height, spec.width), dtype=np.float64)
    rays = np.stack([(xs - spec.cx) / spec.focal, (ys - spec.cy) / spec.focal, np.ones_like(xs)], axis=-1)
    return normalize(rays)


def build_remap(camera: DoubleSphereCamera, rect_from_camera: np.ndarray, spec: RectificationSpec) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rays_rect = virtual_rays(spec).reshape(-1, 3)
    rays_camera = rays_rect @ rect_from_camera
    pixels, valid = ds_project(camera, rays_camera)
    map_x = pixels[:, 0].reshape(spec.height, spec.width).astype(np.float32)
    map_y = pixels[:, 1].reshape(spec.height, spec.width).astype(np.float32)
    valid = valid.reshape(spec.height, spec.width)
    map_x[~valid] = -1.0
    map_y[~valid] = -1.0
    return map_x, map_y, valid


def rectified_points(camera: DoubleSphereCamera, rect_from_camera: np.ndarray, pixels: np.ndarray, spec: RectificationSpec) -> tuple[np.ndarray, np.ndarray]:
    rays, valid = ds_unproject(camera, pixels)
    rays_rect = rays @ rect_from_camera.T
    projected = np.full((len(pixels), 2), np.nan, dtype=np.float64)
    valid &= rays_rect[:, 2] > 1e-12
    projected[valid, 0] = spec.focal * rays_rect[valid, 0] / rays_rect[valid, 2] + spec.cx
    projected[valid, 1] = spec.focal * rays_rect[valid, 1] / rays_rect[valid, 2] + spec.cy
    valid &= np.isfinite(projected).all(axis=1)
    valid &= (projected[:, 0] >= 0.0) & (projected[:, 0] < spec.width)
    valid &= (projected[:, 1] >= 0.0) & (projected[:, 1] < spec.height)
    return projected, valid


def image_identity_from_path(path: Path) -> tuple[int, int]:
    match = re.match(r"(\d+)_(?:left|right)_(\d+)_mono8\.(?:png|jpg|jpeg|bmp|tiff)$", path.name, re.IGNORECASE)
    if not match:
        fail(f"could not parse frame ID and capture timestamp from {path.name}")
    return int(match.group(1)), int(match.group(2))


def list_pairs(left_dir: Path, right_dir: Path, timestamp_tolerance_ns: int) -> list[ImagePair]:
    if not left_dir.is_dir() or not right_dir.is_dir():
        fail(f"test image directories must exist: {left_dir}, {right_dir}")
    if timestamp_tolerance_ns < 0:
        fail("timestamp tolerance must be non-negative")
    extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}
    left = [(frame_id, timestamp, path) for path in left_dir.iterdir() if path.suffix.lower() in extensions for frame_id, timestamp in [image_identity_from_path(path)]]
    right = [(frame_id, timestamp, path) for path in right_dir.iterdir() if path.suffix.lower() in extensions for frame_id, timestamp in [image_identity_from_path(path)]]
    if not left or not right:
        fail("no readable image candidates for timestamp pairing")

    # Select one-to-one pairs by smallest absolute timestamp difference first.
    # This rule is independent of either calibration and forbids frame-index-only
    # pairing when cameras have dropped or delayed captures.
    candidates = []
    for left_index, (left_frame, left_timestamp, left_path) in enumerate(left):
        for right_index, (right_frame, right_timestamp, right_path) in enumerate(right):
            delta = right_timestamp - left_timestamp
            if abs(delta) <= timestamp_tolerance_ns:
                candidates.append((abs(delta), left_timestamp, right_timestamp, left_frame, right_frame, left_index, right_index, delta, left_path, right_path))
    candidates.sort()
    used_left, used_right, pairs = set(), set(), []
    for _, _, _, left_frame, _, left_index, right_index, delta, left_path, right_path in candidates:
        if left_index in used_left or right_index in used_right:
            continue
        used_left.add(left_index)
        used_right.add(right_index)
        pairs.append(ImagePair(left_frame, left_path, right_path, 0.0, 0.0, 0.0, 0.0, delta))
    if not pairs:
        fail(f"no one-to-one pairs within {timestamp_tolerance_ns} ns")
    return sorted(pairs, key=lambda pair: (pair.frame_id, pair.left_path.name))


def raw_selection_scores(image: np.ndarray) -> tuple[float, float, float, float]:
    max_side = 960
    scale = min(1.0, max_side / max(image.shape[:2]))
    reduced = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1.0 else image
    gradient = cv2.Sobel(reduced, cv2.CV_32F, 1, 0) ** 2 + cv2.Sobel(reduced, cv2.CV_32F, 0, 1) ** 2
    texture = float(np.mean(np.sqrt(gradient)))
    height, width = reduced.shape[:2]
    yy, xx = np.indices((height, width), dtype=np.float32)
    rho = np.sqrt(((xx - (width - 1) / 2.0) / (width / 2.0)) ** 2 + ((yy - (height - 1) / 2.0) / (height / 2.0)) ** 2)
    peripheral = float(np.mean(np.sqrt(gradient)[(rho >= 0.65) & (rho <= 0.90)]))
    edges = cv2.Canny(reduced, 60, 140)
    lines = cv2.HoughLinesP(edges, 1.0, np.pi / 180.0, threshold=40, minLineLength=max(20, width // 18), maxLineGap=8)
    line_score = 0.0 if lines is None else float(len(lines))
    return texture + 0.6 * peripheral + 0.08 * line_score, texture, peripheral, line_score


def choose_blind_pairs(raw_pairs: Sequence[ImagePair], count: int) -> list[ImagePair]:
    if len(raw_pairs) < count:
        fail(f"need {count} synchronized pairs, found {len(raw_pairs)}")
    scored = []
    for pair in raw_pairs:
        image = cv2.imread(str(pair.left_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue
        score, texture, peripheral, line = raw_selection_scores(image)
        scored.append(ImagePair(pair.frame_id, pair.left_path, pair.right_path, score, texture, peripheral, line, pair.timestamp_delta_ns))
    if len(scored) < count:
        fail("too few readable pairs for blind frame selection")
    selected = []
    for bin_items in np.array_split(np.asarray(scored, dtype=object), count):
        selected.append(max(bin_items.tolist(), key=lambda item: item.selection_score))
    return sorted(selected, key=lambda item: item.frame_id)


def write_manifest(pairs: Sequence[ImagePair], output_dir: Path) -> Path:
    csv_path = output_dir / "frame_manifest.csv"
    ordered_for_display = sorted(pairs, key=lambda item: (-item.selection_score, item.frame_id))
    display_rank = {item.frame_id: index + 1 for index, item in enumerate(ordered_for_display)}
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["frame_id", "left_image", "right_image", "timestamp_delta_ns", "selection_score", "texture_score", "peripheral_score", "line_score", "display_rank"])
        writer.writeheader()
        for pair in pairs:
            writer.writerow({"frame_id": pair.frame_id, "left_image": pair.left_path, "right_image": pair.right_path, "timestamp_delta_ns": pair.timestamp_delta_ns, "selection_score": f"{pair.selection_score:.9f}", "texture_score": f"{pair.texture_score:.9f}", "peripheral_score": f"{pair.peripheral_score:.9f}", "line_score": f"{pair.line_score:.9f}", "display_rank": display_rank[pair.frame_id]})
    yaml_path = output_dir / "frame_manifest.yaml"
    lines = ["selection_protocol: raw_image_only_after_timestamp_pairing", "pairing_protocol: global_one_to_one_greedy_minimum_absolute_timestamp_difference", "frame_count: " + str(len(pairs)), "frames:"]
    for pair in pairs:
        lines.extend([f"  - frame_id: {pair.frame_id}", f"    left_image: {pair.left_path}", f"    right_image: {pair.right_path}", f"    timestamp_delta_ns: {pair.timestamp_delta_ns}", f"    selection_score: {pair.selection_score:.9f}", f"    texture_score: {pair.texture_score:.9f}", f"    peripheral_score: {pair.peripheral_score:.9f}", f"    line_score: {pair.line_score:.9f}", f"    display_rank: {display_rank[pair.frame_id]}"])
    yaml_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return csv_path


def load_manifest(path: Path) -> list[ImagePair]:
    if not path.is_file():
        fail(f"missing manifest: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        fail(f"empty manifest: {path}")
    if "timestamp_delta_ns" not in rows[0]:
        fail(f"legacy manifest lacks timestamp pairing data: {path}; rerun with --refresh-freeze")
    return [ImagePair(int(row["frame_id"]), Path(row["left_image"]), Path(row["right_image"]), float(row["selection_score"]), float(row["texture_score"]), float(row["peripheral_score"]), float(row["line_score"]), int(row["timestamp_delta_ns"])) for row in rows]


def extract_frozen_matches(pairs: Sequence[ImagePair], output_path: Path, refresh: bool) -> list[dict[str, float | int]]:
    if output_path.is_file() and not refresh:
        with output_path.open(newline="", encoding="utf-8") as handle:
            return [{key: (int(value) if key == "frame_id" else float(value)) for key, value in row.items()} for row in csv.DictReader(handle)]
    sift = cv2.SIFT_create(nfeatures=6000, contrastThreshold=0.02)
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    records: list[dict[str, float | int]] = []
    for pair in pairs:
        left = cv2.imread(str(pair.left_path), cv2.IMREAD_GRAYSCALE)
        right = cv2.imread(str(pair.right_path), cv2.IMREAD_GRAYSCALE)
        if left is None or right is None:
            fail(f"could not read image pair {pair.frame_id}")
        scale = min(1.0, 2048.0 / max(left.shape[0], left.shape[1], right.shape[0], right.shape[1]))
        left_small = cv2.resize(left, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1.0 else left
        right_small = cv2.resize(right, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1.0 else right
        kpl, desl = sift.detectAndCompute(left_small, None)
        kpr, desr = sift.detectAndCompute(right_small, None)
        if desl is None or desr is None:
            continue
        forward = matcher.knnMatch(desl, desr, k=2)
        backward = matcher.knnMatch(desr, desl, k=2)
        forward_ok = {match.queryIdx: match for candidates in forward if len(candidates) == 2 for match, second in [candidates] if match.distance < 0.75 * second.distance}
        backward_ok = {match.queryIdx: match for candidates in backward if len(candidates) == 2 for match, second in [candidates] if match.distance < 0.75 * second.distance}
        for query_idx, match in forward_ok.items():
            reverse = backward_ok.get(match.trainIdx)
            if reverse is None or reverse.trainIdx != query_idx:
                continue
            x_left, y_left = np.asarray(kpl[query_idx].pt) / scale
            x_right, y_right = np.asarray(kpr[match.trainIdx].pt) / scale
            rho = math.sqrt(((x_left - (left.shape[1] - 1) / 2.0) / (left.shape[1] / 2.0)) ** 2 + ((y_left - (left.shape[0] - 1) / 2.0) / (left.shape[0] / 2.0)) ** 2)
            records.append({"frame_id": pair.frame_id, "u_left": float(x_left), "v_left": float(y_left), "u_right": float(x_right), "v_right": float(y_right), "descriptor_distance": float(match.distance), "raw_left_rho": float(rho)})
    if not records:
        fail("SIFT produced no frozen mutual correspondences")
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        fields = ["frame_id", "u_left", "v_left", "u_right", "v_right", "descriptor_distance", "raw_left_rho"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    return records


def create_matcher(min_disparity: int, num_disparities: int, block_size: int) -> cv2.StereoSGBM:
    return cv2.StereoSGBM_create(minDisparity=min_disparity, numDisparities=num_disparities, blockSize=block_size, P1=8 * block_size * block_size, P2=32 * block_size * block_size, disp12MaxDiff=1, preFilterCap=31, uniquenessRatio=10, speckleWindowSize=100, speckleRange=2, mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)


def compute_disparity(left: np.ndarray, right: np.ndarray, left_valid: np.ndarray, right_valid: np.ndarray, spec: RectificationSpec) -> tuple[np.ndarray, np.ndarray]:
    disparity = create_matcher(spec.min_disparity, spec.num_disparities, spec.block_size).compute(left, right).astype(np.float32) / 16.0
    reverse = create_matcher(-spec.num_disparities, spec.num_disparities, spec.block_size).compute(right, left).astype(np.float32) / 16.0
    height, width = disparity.shape
    yy, xx = np.indices((height, width))
    xr = np.rint(xx - disparity).astype(np.int32)
    inside = (xr >= 0) & (xr < width)
    sampled_reverse = np.full_like(disparity, np.nan)
    sampled_right_valid = np.zeros_like(left_valid)
    sampled_reverse[inside] = reverse[yy[inside], xr[inside]]
    sampled_right_valid[inside] = right_valid[yy[inside], xr[inside]]
    valid = left_valid & sampled_right_valid & inside
    valid &= np.isfinite(disparity) & (disparity >= spec.min_disparity) & (disparity < spec.min_disparity + spec.num_disparities)
    valid &= np.isfinite(sampled_reverse) & (np.abs(disparity + sampled_reverse) <= 1.0)
    return disparity, valid


def percentile(values: np.ndarray, q: float) -> float:
    return float(np.percentile(values, q)) if len(values) else float("nan")


def fixed_regions(spec: RectificationSpec) -> tuple[np.ndarray, np.ndarray]:
    yy, xx = np.indices((spec.height, spec.width), dtype=np.float64)
    rho = np.sqrt(((xx - spec.cx) / (spec.width / 2.0)) ** 2 + ((yy - spec.cy) / (spec.height / 2.0)) ** 2)
    return rho <= 0.90, (rho >= 0.65) & (rho <= 0.90)


def colorize_disparity(disparity: np.ndarray, valid: np.ndarray, spec: RectificationSpec) -> np.ndarray:
    normalized = np.clip((disparity - spec.min_disparity) * 255.0 / spec.num_disparities, 0.0, 255.0).astype(np.uint8)
    color = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
    color[~valid] = (128, 128, 128)
    return color


def box_from_relative(spec: RectificationSpec, values: tuple[float, float, float, float]) -> tuple[int, int, int, int]:
    x, y, w, h = values
    return int(round(x * spec.width)), int(round(y * spec.height)), int(round(w * spec.width)), int(round(h * spec.height))


LINE_ROI = (0.58, 0.16, 0.30, 0.25)
CENTER_ROI = (0.40, 0.40, 0.20, 0.20)
PERIPHERAL_ROI = (0.68, 0.18, 0.22, 0.22)


def draw_line_fit(image: np.ndarray, roi: tuple[int, int, int, int]) -> np.ndarray:
    output = image.copy()
    x, y, w, h = roi
    cv2.rectangle(output, (x, y), (x + w, y + h), (70, 210, 70), 3, cv2.LINE_AA)
    gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray[y:y + h, x:x + w], 70, 150)
    lines = cv2.HoughLinesP(edges, 1.0, np.pi / 180.0, 22, minLineLength=max(24, w // 4), maxLineGap=10)
    if lines is not None:
        candidates = lines.reshape(-1, 4)
        best = max(candidates, key=lambda line: float((line[2] - line[0]) ** 2 + (line[3] - line[1]) ** 2))
        cv2.line(output, (x + int(best[0]), y + int(best[1])), (x + int(best[2]), y + int(best[3])), (50, 230, 50), 4, cv2.LINE_AA)
    return output


def paste_inset(image: np.ndarray, roi: tuple[int, int, int, int], corner: str) -> np.ndarray:
    output = image.copy()
    x, y, w, h = roi
    crop = output[y:y + h, x:x + w]
    if crop.size == 0:
        return output
    inset_width = max(140, output.shape[1] // 4)
    inset_height = max(100, int(inset_width * h / max(w, 1)))
    inset = cv2.resize(crop, (inset_width, inset_height), interpolation=cv2.INTER_CUBIC)
    margin = 14
    ox = margin if corner == "left" else output.shape[1] - inset_width - margin
    oy = margin
    cv2.rectangle(output, (ox - 3, oy - 3), (ox + inset_width + 3, oy + inset_height + 3), (255, 255, 255), -1)
    output[oy:oy + inset_height, ox:ox + inset_width] = inset
    cv2.rectangle(output, (ox, oy), (ox + inset_width, oy + inset_height), (20, 20, 20), 2)
    return output


def panel_label(image: np.ndarray, label: str) -> np.ndarray:
    height, width = image.shape[:2]
    bar = np.full((54, width, 3), 255, dtype=np.uint8)
    cv2.putText(bar, label, (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (24, 24, 24), 2, cv2.LINE_AA)
    return np.vstack([bar, image])


def render_figure(output_dir: Path, images: dict[str, dict[str, np.ndarray]], spec: RectificationSpec) -> Path:
    line_roi = box_from_relative(spec, LINE_ROI)
    center_roi = box_from_relative(spec, CENTER_ROI)
    peripheral_roi = box_from_relative(spec, PERIPHERAL_ROI)
    top_panels = []
    bottom_panels = []
    for method in ("Kalibr", "Ours"):
        rect = draw_line_fit(images[method]["rectified_left"], line_roi)
        rect = paste_inset(rect, line_roi, "right")
        top_panels.append(panel_label(rect, method))
        disp = images[method]["disparity_color"].copy()
        for box, color in ((center_roi, (255, 255, 255)), (peripheral_roi, (45, 225, 245))):
            x, y, w, h = box
            cv2.rectangle(disp, (x, y), (x + w, y + h), color, 3, cv2.LINE_AA)
        disp = paste_inset(disp, center_roi, "left")
        disp = paste_inset(disp, peripheral_roi, "right")
        bottom_panels.append(panel_label(disp, method))
    divider = np.full((8, top_panels[0].shape[1] * 2 + 8, 3), 238, dtype=np.uint8)
    top = np.hstack([top_panels[0], np.full((top_panels[0].shape[0], 8, 3), 238, dtype=np.uint8), top_panels[1]])
    bottom = np.hstack([bottom_panels[0], np.full((bottom_panels[0].shape[0], 8, 3), 238, dtype=np.uint8), bottom_panels[1]])
    figure = np.vstack([top, divider, bottom])
    footer = np.full((54, figure.shape[1], 3), 255, dtype=np.uint8)
    gradient = np.linspace(0, 255, 420, dtype=np.uint8).reshape(1, -1)
    colorbar = cv2.applyColorMap(gradient, cv2.COLORMAP_TURBO)
    x0, y0 = figure.shape[1] // 2 - 210, 12
    footer[y0:y0 + 14, x0:x0 + 420] = colorbar
    cv2.rectangle(footer, (x0, y0), (x0 + 420, y0 + 14), (20, 20, 20), 1)
    for x, value in ((x0, "0"), (x0 + 210, str(spec.num_disparities // 2)), (x0 + 402, str(spec.num_disparities))):
        cv2.putText(footer, value, (x, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (24, 24, 24), 1, cv2.LINE_AA)
    cv2.putText(footer, "Disparity (px)", (x0 - 118, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (24, 24, 24), 1, cv2.LINE_AA)
    figure = np.vstack([figure, footer])
    path = output_dir / "stereo_rectification_disparity_figure.png"
    cv2.imwrite(str(path), figure)
    return path


def write_figure_pdf(output_dir: Path, png_path: Path) -> None:
    if shutil.which("pdflatex") is None:
        return
    tex = output_dir / "stereo_rectification_disparity_figure.tex"
    tex.write_text("\\documentclass{standalone}\n\\usepackage{graphicx}\n\\begin{document}\n\\includegraphics[width=0.99\\linewidth]{" + png_path.name + "}\n\\end{document}\n", encoding="utf-8")
    subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", tex.name], cwd=output_dir, check=True, stdout=subprocess.DEVNULL)
    generated = output_dir / "stereo_rectification_disparity_figure.pdf"
    if not generated.is_file():
        fail("pdflatex did not create the composite figure PDF")


def write_table_tex(path: Path, metrics: dict[str, dict[str, float]]) -> None:
    path.write_text("\n".join(["\\begin{tabular}{lcccc}", "\\toprule", "Method & Epipolar P95 $\\downarrow$ & Peripheral P95 $\\downarrow$ & Valid disparity $\\uparrow$ & Peripheral valid $\\uparrow$ \\\\", "\\midrule", *[f"{name} & {row['epipolar_p95']:.3f} px & {row['peripheral_p95']:.3f} px & {100.0 * row['valid_ratio']:.1f}\\% & {100.0 * row['peripheral_valid_ratio']:.1f}\\% \\\\" for name, row in metrics.items()], "\\bottomrule", "\\end{tabular}", ""]), encoding="utf-8")


def write_latex_includes(output_dir: Path) -> None:
    text = r"""% Stereo rectification and disparity visualization
\begin{figure*}[t]
  \centering
  \includegraphics[width=0.99\textwidth]{figures/stereo_rectification_disparity_figure.pdf}
  \caption{Stereo rectification and disparity visualization on four frozen synchronized test pairs. Kalibr and Ours use their respective DS intrinsics and stereo extrinsics, while the virtual pinhole view, interpolation, SGBM parameters, color range, and regions of interest are shared. Gray denotes invalid disparity. This figure is a qualitative downstream illustration rather than a depth-ground-truth evaluation.}
  \label{fig:stereo-rectification-disparity}
\end{figure*}

% Requires \usepackage{booktabs}.
\begin{table}[t]
  \centering
  \caption{Rectified correspondence and disparity validity on the frozen stereo test pairs. Peripheral metrics use the fixed image annulus $\rho\in[0.65,0.90]$.}
  \label{tab:stereo-rectification-disparity}
  \input{tables/stereo_rectification_disparity_table}
\end{table}
"""
    (output_dir / "latex_includes.tex").write_text(text, encoding="utf-8")


def run_self_checks(camera: DoubleSphereCamera, spec: RectificationSpec) -> dict[str, float]:
    grid_x, grid_y = np.meshgrid(np.linspace(100.0, camera.width - 101.0, 15), np.linspace(100.0, camera.height - 101.0, 15))
    pixels = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    rays, valid = ds_unproject(camera, pixels)
    reproj, projected = ds_project(camera, rays)
    common = valid & projected
    roundtrip = float(np.max(np.linalg.norm(reproj[common] - pixels[common], axis=1))) if np.any(common) else float("inf")
    if not math.isfinite(roundtrip) or roundtrip > 1e-6:
        fail(f"DS project/unproject roundtrip check failed: {roundtrip}")
    base = np.array([0.1, 0.5, 0.2, 0.7])
    if percentile(base, 95.0) >= percentile(base + 1.0, 95.0):
        fail("vertical offset P95 monotonicity check failed")
    _, _, valid_map = build_remap(camera, np.eye(3), spec)
    if valid_map.shape != (spec.height, spec.width) or not np.isfinite(valid_map.astype(float)).all():
        fail("remap validity check failed")
    return {"ds_roundtrip_max_px": roundtrip, "map_valid_ratio": float(np.mean(valid_map))}


def write_protocol(output_dir: Path, args: argparse.Namespace, spec: RectificationSpec, pairs: Sequence[ImagePair], checks: dict[str, float], systems: Sequence[StereoSystem], timestamp_pair_count: int) -> None:
    payload = {"protocol": "Stereo Rectification and Disparity Visualization", "test_left_dir": str(args.left_dir), "test_right_dir": str(args.right_dir), "calibration_left_dir": str(args.calibration_left_dir), "calibration_right_dir": str(args.calibration_right_dir), "timestamp_pairing": {"rule": "global_one_to_one_greedy_minimum_absolute_timestamp_difference", "tolerance_ms": args.timestamp_tolerance_ms, "eligible_pair_count": timestamp_pair_count, "selected_timestamp_deltas_ns": [pair.timestamp_delta_ns for pair in pairs]}, "frame_count": len(pairs), "frame_ids": [pair.frame_id for pair in pairs], "virtual_pinhole": {"width": spec.width, "height": spec.height, "horizontal_fov_deg": spec.hfov_deg, "focal_px": spec.focal, "interpolation": "INTER_LINEAR"}, "sgbm": {"min_disparity": spec.min_disparity, "num_disparities": spec.num_disparities, "block_size": spec.block_size, "P1": 8 * spec.block_size * spec.block_size, "P2": 32 * spec.block_size * spec.block_size, "uniqueness_ratio": 10, "speckle_window_size": 100, "speckle_range": 2, "left_right_consistency_px": 1.0}, "systems": [{"name": item.name, "left_intrinsics": item.left.__dict__, "right_intrinsics": item.right.__dict__, "baseline_m": float(np.linalg.norm(item.translation_cam1_cam0))} for item in systems], "checks": checks}
    (output_dir / "protocol.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def ensure_independent_test(args: argparse.Namespace, pairs: Sequence[ImagePair]) -> None:
    left_test = args.left_dir.resolve()
    right_test = args.right_dir.resolve()
    if left_test == args.calibration_left_dir.resolve() or right_test == args.calibration_right_dir.resolve():
        fail("test image directory equals a calibration image directory")
    calibration_paths = set(args.calibration_left_dir.glob("*")) | set(args.calibration_right_dir.glob("*"))
    overlap = [pair for pair in pairs if pair.left_path in calibration_paths or pair.right_path in calibration_paths]
    if overlap:
        fail("selected test frames overlap a calibration image manifest")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ours-left-intrinsics", type=Path, default=DEFAULT_OURS_LEFT)
    parser.add_argument("--ours-right-intrinsics", type=Path, default=DEFAULT_OURS_RIGHT)
    parser.add_argument("--ours-extrinsic", type=Path, default=DEFAULT_OURS_EXTRINSIC)
    parser.add_argument("--kalibr-camchain", type=Path, default=DEFAULT_KALIBR)
    parser.add_argument("--left-dir", type=Path, default=DEFAULT_LEFT_DIR)
    parser.add_argument("--right-dir", type=Path, default=DEFAULT_RIGHT_DIR)
    parser.add_argument("--calibration-left-dir", type=Path, default=DEFAULT_CALIB_LEFT_DIR)
    parser.add_argument("--calibration-right-dir", type=Path, default=DEFAULT_CALIB_RIGHT_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--frame-count", type=int, default=4)
    parser.add_argument("--timestamp-tolerance-ms", type=float, default=1.0, help="maximum absolute capture-timestamp difference for a stereo pair")
    parser.add_argument("--refresh-freeze", action="store_true", help="replace an existing blind frame manifest and frozen raw matches")
    parser.add_argument("--smoke", action="store_true", help="use one blind-selected pair and a 512x384 virtual pinhole output")
    parser.add_argument("--width", type=int, default=2048)
    parser.add_argument("--height", type=int, default=1536)
    parser.add_argument("--hfov-deg", type=float, default=120.0)
    parser.add_argument("--skip-pdf", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.frame_count < 1 or args.frame_count > 5:
        fail("frame-count must be in [1, 5]")
    if not math.isfinite(args.timestamp_tolerance_ms) or args.timestamp_tolerance_ms < 0.0:
        fail("timestamp tolerance must be finite and non-negative")
    if args.smoke:
        args.frame_count, args.width, args.height = 1, 512, 384
    if args.width < 256 or args.height < 192 or not (30.0 < args.hfov_deg < 170.0):
        fail("invalid virtual pinhole resolution or horizontal FoV")
    if args.output.exists() and args.refresh_freeze:
        for path in (args.output / "frame_manifest.csv", args.output / "frame_manifest.yaml", args.output / "frozen_matches.csv"):
            path.unlink(missing_ok=True)
    args.output.mkdir(parents=True, exist_ok=True)
    spec = RectificationSpec(args.width, args.height, args.hfov_deg, 0, 192, 7)
    ours = load_ours_system(args.ours_left_intrinsics, args.ours_right_intrinsics, args.ours_extrinsic)
    kalibr = load_kalibr_system(args.kalibr_camchain)
    systems = [kalibr, ours]
    checks = run_self_checks(ours.left, spec)
    raw_pairs = list_pairs(args.left_dir, args.right_dir, int(round(args.timestamp_tolerance_ms * 1e6)))
    manifest_path = args.output / "frame_manifest.csv"
    pairs = load_manifest(manifest_path) if manifest_path.exists() else choose_blind_pairs(raw_pairs, args.frame_count)
    if not manifest_path.exists():
        write_manifest(pairs, args.output)
    ensure_independent_test(args, pairs)
    matches = extract_frozen_matches(pairs, args.output / "frozen_matches.csv", args.refresh_freeze)
    matches_by_frame: dict[int, list[dict[str, float | int]]] = {}
    for match in matches:
        matches_by_frame.setdefault(int(match["frame_id"]), []).append(match)
    eval_region, peripheral_region = fixed_regions(spec)
    metrics: dict[str, dict[str, float]] = {}
    per_frame_rows = []
    rectified_match_rows = []
    map_audit_rows = []
    display_frame = min(pairs, key=lambda pair: (-pair.selection_score, pair.frame_id)).frame_id
    figure_images: dict[str, dict[str, np.ndarray]] = {}
    for system in systems:
        rect0, rect1 = rectification_rotations(system)
        left_map_x, left_map_y, left_map_valid = build_remap(system.left, rect0, spec)
        right_map_x, right_map_y, right_map_valid = build_remap(system.right, rect1, spec)
        if not np.isfinite(left_map_x[left_map_valid]).all() or not np.isfinite(right_map_x[right_map_valid]).all():
            fail(f"non-finite valid remap values for {system.name}")
        map_audit_rows.append({"method": system.name, "left_map_valid_ratio": float(np.mean(left_map_valid)), "right_map_valid_ratio": float(np.mean(right_map_valid)), "width": spec.width, "height": spec.height, "horizontal_fov_deg": spec.hfov_deg})
        all_errors, peripheral_errors = [], []
        total_valid, total_count, peripheral_valid, peripheral_count = 0, 0, 0, 0
        for pair in pairs:
            left = cv2.imread(str(pair.left_path), cv2.IMREAD_GRAYSCALE)
            right = cv2.imread(str(pair.right_path), cv2.IMREAD_GRAYSCALE)
            if left is None or right is None:
                fail(f"could not read selected pair {pair.frame_id}")
            left_rect = cv2.remap(left, left_map_x, left_map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=127)
            right_rect = cv2.remap(right, right_map_x, right_map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=127)
            disparity, valid_disp = compute_disparity(left_rect, right_rect, left_map_valid, right_map_valid, spec)
            row_matches = matches_by_frame.get(pair.frame_id, [])
            raw_left = np.asarray([[float(row["u_left"]), float(row["v_left"])] for row in row_matches], dtype=np.float64)
            raw_right = np.asarray([[float(row["u_right"]), float(row["v_right"])] for row in row_matches], dtype=np.float64)
            if len(raw_left):
                mapped_left, valid_left = rectified_points(system.left, rect0, raw_left, spec)
                mapped_right, valid_right = rectified_points(system.right, rect1, raw_right, spec)
                valid_match = valid_left & valid_right
                errors = np.abs(mapped_left[valid_match, 1] - mapped_right[valid_match, 1])
                raw_rhos = np.asarray([float(row["raw_left_rho"]) for row in row_matches])
                rhos = raw_rhos[valid_match]
                all_errors.extend(errors.tolist())
                peripheral_errors.extend(errors[(rhos >= 0.65) & (rhos <= 0.90)].tolist())
                for match_index, match_row in enumerate(row_matches):
                    vertical_error = abs(mapped_left[match_index, 1] - mapped_right[match_index, 1]) if valid_match[match_index] else float("nan")
                    rectified_match_rows.append({"method": system.name, "frame_id": pair.frame_id, "match_index": match_index, "u_left_raw": raw_left[match_index, 0], "v_left_raw": raw_left[match_index, 1], "u_right_raw": raw_right[match_index, 0], "v_right_raw": raw_right[match_index, 1], "raw_left_rho": raw_rhos[match_index], "u_left_rect": mapped_left[match_index, 0], "v_left_rect": mapped_left[match_index, 1], "u_right_rect": mapped_right[match_index, 0], "v_right_rect": mapped_right[match_index, 1], "valid": int(valid_match[match_index]), "vertical_error_px": vertical_error})
            frame_valid = int(np.count_nonzero(valid_disp & eval_region))
            frame_total = int(np.count_nonzero(eval_region))
            frame_peripheral_valid = int(np.count_nonzero(valid_disp & peripheral_region))
            frame_peripheral_total = int(np.count_nonzero(peripheral_region))
            total_valid += frame_valid
            total_count += frame_total
            peripheral_valid += frame_peripheral_valid
            peripheral_count += frame_peripheral_total
            per_frame_rows.append({"method": system.name, "frame_id": pair.frame_id, "frozen_match_count": len(row_matches), "epipolar_match_count": len(errors) if len(raw_left) else 0, "epipolar_p95_px": percentile(errors, 95.0) if len(raw_left) else float("nan"), "peripheral_epipolar_p95_px": percentile(errors[(rhos >= 0.65) & (rhos <= 0.90)], 95.0) if len(raw_left) else float("nan"), "valid_disparity_ratio": frame_valid / frame_total, "peripheral_valid_disparity_ratio": frame_peripheral_valid / frame_peripheral_total})
            pair_dir = args.output / "pairs" / f"frame_{pair.frame_id:06d}" / system.name.lower()
            pair_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(pair_dir / "rectified_left.png"), left_rect)
            cv2.imwrite(str(pair_dir / "rectified_right.png"), right_rect)
            cv2.imwrite(str(pair_dir / "disparity.png"), colorize_disparity(disparity, valid_disp, spec))
            cv2.imwrite(str(pair_dir / "valid_disparity_mask.png"), (valid_disp.astype(np.uint8) * 255))
            if pair.frame_id == display_frame:
                figure_images[system.name] = {"rectified_left": cv2.cvtColor(left_rect, cv2.COLOR_GRAY2BGR), "disparity_color": colorize_disparity(disparity, valid_disp, spec)}
        if not all_errors or total_count == 0 or peripheral_count == 0:
            fail(f"insufficient evaluation support for {system.name}")
        metrics[system.name] = {"epipolar_p95": percentile(np.asarray(all_errors), 95.0), "peripheral_p95": percentile(np.asarray(peripheral_errors), 95.0), "valid_ratio": total_valid / total_count, "peripheral_valid_ratio": peripheral_valid / peripheral_count, "frozen_match_count": float(len(all_errors)), "peripheral_match_count": float(len(peripheral_errors))}
    with (args.output / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["method", "epipolar_p95_px", "peripheral_p95_px", "valid_disparity_ratio", "peripheral_valid_disparity_ratio", "frozen_match_count", "peripheral_match_count"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for method, row in metrics.items():
            writer.writerow({"method": method, "epipolar_p95_px": row["epipolar_p95"], "peripheral_p95_px": row["peripheral_p95"], "valid_disparity_ratio": row["valid_ratio"], "peripheral_valid_disparity_ratio": row["peripheral_valid_ratio"], "frozen_match_count": int(row["frozen_match_count"]), "peripheral_match_count": int(row["peripheral_match_count"])})
    with (args.output / "per_frame_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_frame_rows[0]))
        writer.writeheader()
        writer.writerows(per_frame_rows)
    with (args.output / "rectified_frozen_matches.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rectified_match_rows[0]))
        writer.writeheader()
        writer.writerows(rectified_match_rows)
    with (args.output / "rectification_map_audit.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(map_audit_rows[0]))
        writer.writeheader()
        writer.writerows(map_audit_rows)
    write_table_tex(args.output / "stereo_rectification_disparity_table.tex", metrics)
    write_latex_includes(args.output)
    figure_path = render_figure(args.output, figure_images, spec)
    if not args.skip_pdf:
        write_figure_pdf(args.output, figure_path)
    write_protocol(args.output, args, spec, pairs, checks, systems, len(raw_pairs))
    print(f"output={args.output}")
    for method, row in metrics.items():
        print(f"{method}: epipolar_p95={row['epipolar_p95']:.4f}px peripheral_p95={row['peripheral_p95']:.4f}px valid={100.0 * row['valid_ratio']:.2f}% peripheral_valid={100.0 * row['peripheral_valid_ratio']:.2f}%")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2)
