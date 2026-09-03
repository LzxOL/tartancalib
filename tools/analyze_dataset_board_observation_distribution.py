#!/usr/bin/env python3
"""Dataset-level board observation distribution analysis for Stage5/Stage6 runs."""

from __future__ import annotations

import argparse
import csv
import math
import struct
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
SUMMARY_FIELDS = [
    "stage",
    "run_dir",
    "dataset_label",
    "training_input",
    "holdout_input",
    "observation_scope",
    "frame_or_pair_count",
    "board_observation_count",
    "selected_board_observation_count",
    "unique_board_count",
    "point_count",
    "outer_point_count",
    "internal_point_count",
    "mean_boards_per_frame_or_pair",
    "max_boards_per_frame_or_pair",
    "boards_per_frame_or_pair_hist",
    "single_board_frame_or_pair_ratio",
    "multi_board_frame_or_pair_ratio",
    "four_plus_board_frame_or_pair_ratio",
    "mean_points_per_board_observation",
    "mean_projected_area_ratio",
    "median_projected_area_ratio",
    "p90_projected_area_ratio",
    "large_area_ratio_ge_0p015",
    "large_area_ratio_ge_0p02",
    "large_area_ratio_ge_0p04",
    "large_area_ratio_ge_0p06",
    "mean_polar_angle_deg",
    "median_max_polar_angle_deg",
    "p90_max_polar_angle_deg",
    "high_polar_ratio_ge_50deg",
    "high_polar_ratio_ge_60deg",
    "high_polar_ratio_ge_70deg",
    "edge_margin_ratio_le_0p05",
    "close_edge_like_ratio_area_ge_0p015_polar_ge_60",
    "close_edge_like_ratio_area_ge_0p04_polar_ge_50",
    "mean_outer_rmse",
    "median_outer_rmse",
    "p90_outer_rmse",
    "notes",
]


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def read_key_values(path: Path) -> Dict[str, str]:
    values: Dict[str, str] = {}
    if not path.exists():
        return values
    with path.open(errors="replace") as handle:
        for line in handle:
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            values[key.strip()] = value.strip()
    return values


def parse_run_log(path: Path) -> Dict[str, str]:
    values: Dict[str, str] = {}
    if not path.exists():
        return values
    with path.open(errors="replace") as handle:
        for line in handle:
            if "[stage5_backend]" not in line and "[stage6" not in line:
                continue
            text = line.strip().split("]", 1)[-1].strip()
            if "=" in text:
                key, value = text.split("=", 1)
                values[key.strip()] = value.strip()
    return values


def to_float(value: object) -> Optional[float]:
    try:
        if value is None:
            return None
        text = str(value).strip()
        if text == "" or text.lower() in {"nan", "inf", "-inf"}:
            return None
        number = float(text)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def to_int(value: object) -> Optional[int]:
    number = to_float(value)
    return int(number) if number is not None else None


def fmt(value: object, digits: int = 6) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            return ""
        return f"{value:.{digits}g}"
    return str(value)


def values(rows: Iterable[Dict[str, object]], key: str) -> List[float]:
    out: List[float] = []
    for row in rows:
        value = to_float(row.get(key))
        if value is not None:
            out.append(value)
    return out


def mean(numbers: Sequence[float]) -> Optional[float]:
    return sum(numbers) / len(numbers) if numbers else None


def percentile(numbers: Sequence[float], pct: float) -> Optional[float]:
    if not numbers:
        return None
    ordered = sorted(numbers)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * pct
    lo = math.floor(rank)
    hi = math.ceil(rank)
    if lo == hi:
        return ordered[lo]
    alpha = rank - lo
    return ordered[lo] * (1.0 - alpha) + ordered[hi] * alpha


def ratio(count: int, total: int) -> Optional[float]:
    return count / total if total > 0 else None


def hist_string(counter: Counter) -> str:
    return " ".join(f"{key}:{counter[key]}" for key in sorted(counter))


def find_first_image_size(path_text: str, repo_root: Path) -> Tuple[Optional[int], Optional[int]]:
    if not path_text:
        return None, None
    path = Path(path_text)
    if not path.is_absolute():
        path = repo_root / path
    candidates: List[Path] = []
    if path.is_file():
        candidates = [path]
    elif path.is_dir():
        candidates = sorted(
            child for child in path.iterdir() if child.suffix.lower() in IMAGE_EXTENSIONS
        )
    for candidate in candidates:
        size = read_image_size(candidate)
        if size != (None, None):
            return size
    return None, None


def read_image_size(path: Path) -> Tuple[Optional[int], Optional[int]]:
    try:
        with path.open("rb") as handle:
            header = handle.read(32)
            if header.startswith(b"\x89PNG\r\n\x1a\n") and len(header) >= 24:
                width, height = struct.unpack(">II", header[16:24])
                return int(width), int(height)
            if header[:2] == b"\xff\xd8":
                handle.seek(2)
                while True:
                    marker_prefix = handle.read(1)
                    if not marker_prefix:
                        break
                    if marker_prefix != b"\xff":
                        continue
                    marker = handle.read(1)
                    while marker == b"\xff":
                        marker = handle.read(1)
                    if marker in {b"\xd8", b"\xd9"}:
                        continue
                    length_bytes = handle.read(2)
                    if len(length_bytes) != 2:
                        break
                    length = struct.unpack(">H", length_bytes)[0]
                    if marker in {
                        b"\xc0",
                        b"\xc1",
                        b"\xc2",
                        b"\xc3",
                        b"\xc5",
                        b"\xc6",
                        b"\xc7",
                        b"\xc9",
                        b"\xca",
                        b"\xcb",
                        b"\xcd",
                        b"\xce",
                        b"\xcf",
                    }:
                        data = handle.read(5)
                        if len(data) == 5:
                            height, width = struct.unpack(">HH", data[1:5])
                            return int(width), int(height)
                        break
                    handle.seek(max(0, length - 2), 1)
    except OSError:
        return None, None
    return None, None


def convex_hull(points: Sequence[Tuple[float, float]]) -> List[Tuple[float, float]]:
    unique = sorted(set(points))
    if len(unique) <= 1:
        return list(unique)

    def cross(o: Tuple[float, float], a: Tuple[float, float], b: Tuple[float, float]) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: List[Tuple[float, float]] = []
    for point in unique:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    upper: List[Tuple[float, float]] = []
    for point in reversed(unique):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    return lower[:-1] + upper[:-1]


def polygon_area(points: Sequence[Tuple[float, float]]) -> Optional[float]:
    hull = convex_hull(points)
    if len(hull) < 3:
        return None
    area = 0.0
    for idx, point in enumerate(hull):
        next_point = hull[(idx + 1) % len(hull)]
        area += point[0] * next_point[1] - next_point[0] * point[1]
    return abs(area) * 0.5


def ds_polar_angle_deg(
    x: float,
    y: float,
    intrinsics: Dict[str, float],
) -> Optional[float]:
    xi = intrinsics.get("xi")
    alpha = intrinsics.get("alpha")
    fu = intrinsics.get("fu")
    fv = intrinsics.get("fv")
    cu = intrinsics.get("cu")
    cv = intrinsics.get("cv")
    if None in {xi, alpha, fu, fv, cu, cv} or fu == 0.0 or fv == 0.0:
        return None
    mx = (x - cu) / fu
    my = (y - cv) / fv
    r2 = mx * mx + my * my
    if alpha > 0.5 and r2 > 1.0 / (2.0 * alpha - 1.0):
        return None
    root_arg = 1.0 - (2.0 * alpha - 1.0) * r2
    if root_arg < 0.0:
        return None
    denom = alpha * math.sqrt(root_arg) + 1.0 - alpha
    if denom == 0.0:
        return None
    mz = (1.0 - alpha * alpha * r2) / denom
    mz2 = mz * mz
    k_root = mz2 + (1.0 - xi * xi) * r2
    if k_root < 0.0 or mz2 + r2 == 0.0:
        return None
    k = (mz * xi + math.sqrt(k_root)) / (mz2 + r2)
    ray = (k * mx, k * my, k * mz - xi)
    norm = math.sqrt(ray[0] * ray[0] + ray[1] * ray[1] + ray[2] * ray[2])
    if norm == 0.0:
        return None
    cos_theta = max(-1.0, min(1.0, ray[2] / norm))
    return math.degrees(math.acos(cos_theta))


def camera_intrinsics_from_summary(run_dir: Path) -> Dict[str, float]:
    summary = read_key_values(run_dir / "backend_optimization_summary.txt")
    out: Dict[str, float] = {}
    for target, source in [
        ("xi", "optimized_camera_xi"),
        ("alpha", "optimized_camera_alpha"),
        ("fu", "optimized_camera_fu"),
        ("fv", "optimized_camera_fv"),
        ("cu", "optimized_camera_cu"),
        ("cv", "optimized_camera_cv"),
    ]:
        value = to_float(summary.get(source))
        if value is None:
            value = to_float(summary.get(source.replace("optimized", "anchor")))
        if value is not None:
            out[target] = value
    return out


def choose_stage5_points_csv(run_dir: Path) -> Tuple[Optional[Path], str]:
    for name, scope in [
        ("backend_optimization_cost_parity_initial_points.csv", "stage5_backend_input_points"),
        ("backend_training_points.csv", "stage5_backend_training_points"),
        ("benchmark_training_points.csv", "stage5_benchmark_training_points"),
        ("pre_backend_filter_points.csv", "stage5_pre_backend_filter_points"),
    ]:
        path = run_dir / name
        if path.exists():
            return path, scope
    return None, "stage5_missing_points_csv"


def stage5_summary(run_dir: Path, repo_root: Path) -> Tuple[Dict[str, object], List[Dict[str, object]], List[Dict[str, object]]]:
    run_log = parse_run_log(run_dir / "run.log")
    points_path, scope = choose_stage5_points_csv(run_dir)
    rows = read_csv(points_path) if points_path else []
    polar_rows = read_csv(run_dir / "backend_optimization_residual_type_per_point.csv")
    polar_by_point: Dict[Tuple[int, int, int, str], float] = {}
    for row in polar_rows:
        frame_index = to_int(row.get("frame_index"))
        board_id = to_int(row.get("board_id"))
        point_id = to_int(row.get("point_id"))
        polar = to_float(row.get("polar_angle_deg"))
        if frame_index is None or board_id is None or point_id is None or polar is None:
            continue
        polar_by_point[(frame_index, board_id, point_id, row.get("point_type", ""))] = polar
    training_input = run_log.get("training_input", "")
    holdout_input = run_log.get("holdout_input", "")
    image_width, image_height = find_first_image_size(training_input, repo_root)
    image_area = image_width * image_height if image_width and image_height else None
    intrinsics = camera_intrinsics_from_summary(run_dir)

    grouped: Dict[Tuple[int, str, int], List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        frame_index = to_int(row.get("frame_index"))
        board_id = to_int(row.get("board_id"))
        if frame_index is None or board_id is None:
            continue
        grouped[(frame_index, row.get("frame_label", ""), board_id)].append(row)

    observation_rows: List[Dict[str, object]] = []
    per_board_counts: Dict[int, Dict[str, object]] = defaultdict(lambda: {
        "point_count": 0,
        "outer_point_count": 0,
        "internal_point_count": 0,
        "observation_count": 0,
        "area_ratios": [],
        "max_polars": [],
    })
    frame_to_boards: Dict[int, set] = defaultdict(set)
    total_points = 0
    total_outer = 0
    total_internal = 0

    for (frame_index, frame_label, board_id), point_rows in grouped.items():
        outer_points: List[Tuple[float, float]] = []
        polar_angles: List[float] = []
        outer_count = 0
        internal_count = 0
        for point in point_rows:
            point_type = point.get("point_type", "")
            point_id = to_int(point.get("point_id"))
            x = to_float(point.get("observed_x"))
            y = to_float(point.get("observed_y"))
            if point_type == "outer":
                outer_count += 1
                if x is not None and y is not None:
                    outer_points.append((x, y))
            elif point_type == "internal":
                internal_count += 1
            polar = None
            if point_id is not None:
                polar = polar_by_point.get((frame_index, board_id, point_id, point_type))
            if polar is None and x is not None and y is not None and intrinsics:
                polar = ds_polar_angle_deg(x, y, intrinsics)
            if polar is not None:
                polar_angles.append(polar)
        point_count = len(point_rows)
        total_points += point_count
        total_outer += outer_count
        total_internal += internal_count
        frame_to_boards[frame_index].add(board_id)
        area = polygon_area(outer_points)
        area_ratio = area / image_area if area is not None and image_area else None
        margin_ratio = None
        if outer_points and image_width and image_height:
            xs = [point[0] for point in outer_points]
            ys = [point[1] for point in outer_points]
            margin_px = min(min(xs), min(ys), image_width - max(xs), image_height - max(ys))
            margin_ratio = margin_px / min(image_width, image_height)
        mean_polar = mean(polar_angles)
        max_polar = max(polar_angles) if polar_angles else None
        observation = {
            "stage": "stage5",
            "run_dir": str(run_dir),
            "frame_or_pair_index": frame_index,
            "frame_or_pair_label": frame_label,
            "board_id": board_id,
            "point_count": point_count,
            "outer_point_count": outer_count,
            "internal_point_count": internal_count,
            "projected_area_ratio": area_ratio,
            "mean_polar_angle_deg": mean_polar,
            "max_polar_angle_deg": max_polar,
            "edge_margin_ratio": margin_ratio,
        }
        observation_rows.append(observation)
        per_board = per_board_counts[board_id]
        per_board["observation_count"] = int(per_board["observation_count"]) + 1
        per_board["point_count"] = int(per_board["point_count"]) + point_count
        per_board["outer_point_count"] = int(per_board["outer_point_count"]) + outer_count
        per_board["internal_point_count"] = int(per_board["internal_point_count"]) + internal_count
        if area_ratio is not None:
            per_board["area_ratios"].append(area_ratio)
        if max_polar is not None:
            per_board["max_polars"].append(max_polar)

    boards_per_frame = [len(boards) for boards in frame_to_boards.values()]
    board_observation_count = len(observation_rows)
    area_ratios = values(observation_rows, "projected_area_ratio")
    mean_polars = values(observation_rows, "mean_polar_angle_deg")
    max_polars = values(observation_rows, "max_polar_angle_deg")
    edge_margins = values(observation_rows, "edge_margin_ratio")
    summary = {
        "stage": "stage5",
        "run_dir": str(run_dir),
        "dataset_label": run_log.get("dataset", run_dir.name),
        "training_input": training_input,
        "holdout_input": holdout_input,
        "observation_scope": scope,
        "frame_or_pair_count": len(frame_to_boards),
        "board_observation_count": board_observation_count,
        "selected_board_observation_count": board_observation_count,
        "unique_board_count": len(per_board_counts),
        "point_count": total_points,
        "outer_point_count": total_outer,
        "internal_point_count": total_internal,
        "mean_boards_per_frame_or_pair": mean(boards_per_frame),
        "max_boards_per_frame_or_pair": max(boards_per_frame) if boards_per_frame else None,
        "boards_per_frame_or_pair_hist": hist_string(Counter(boards_per_frame)),
        "single_board_frame_or_pair_ratio": ratio(sum(1 for count in boards_per_frame if count == 1), len(boards_per_frame)),
        "multi_board_frame_or_pair_ratio": ratio(sum(1 for count in boards_per_frame if count >= 2), len(boards_per_frame)),
        "four_plus_board_frame_or_pair_ratio": ratio(sum(1 for count in boards_per_frame if count >= 4), len(boards_per_frame)),
        "mean_points_per_board_observation": total_points / board_observation_count if board_observation_count else None,
        "mean_projected_area_ratio": mean(area_ratios),
        "median_projected_area_ratio": percentile(area_ratios, 0.5),
        "p90_projected_area_ratio": percentile(area_ratios, 0.9),
        "large_area_ratio_ge_0p015": ratio(sum(1 for value in area_ratios if value >= 0.015), len(area_ratios)),
        "large_area_ratio_ge_0p02": ratio(sum(1 for value in area_ratios if value >= 0.02), len(area_ratios)),
        "large_area_ratio_ge_0p04": ratio(sum(1 for value in area_ratios if value >= 0.04), len(area_ratios)),
        "large_area_ratio_ge_0p06": ratio(sum(1 for value in area_ratios if value >= 0.06), len(area_ratios)),
        "mean_polar_angle_deg": mean(mean_polars),
        "median_max_polar_angle_deg": percentile(max_polars, 0.5),
        "p90_max_polar_angle_deg": percentile(max_polars, 0.9),
        "high_polar_ratio_ge_50deg": ratio(sum(1 for value in max_polars if value >= 50.0), len(max_polars)),
        "high_polar_ratio_ge_60deg": ratio(sum(1 for value in max_polars if value >= 60.0), len(max_polars)),
        "high_polar_ratio_ge_70deg": ratio(sum(1 for value in max_polars if value >= 70.0), len(max_polars)),
        "edge_margin_ratio_le_0p05": ratio(sum(1 for value in edge_margins if value <= 0.05), len(edge_margins)),
        "close_edge_like_ratio_area_ge_0p015_polar_ge_60": ratio(
            sum(
                1
                for row in observation_rows
                if to_float(row.get("projected_area_ratio")) is not None
                and to_float(row.get("projected_area_ratio")) >= 0.015
                and to_float(row.get("max_polar_angle_deg")) is not None
                and to_float(row.get("max_polar_angle_deg")) >= 60.0
            ),
            board_observation_count,
        ),
        "close_edge_like_ratio_area_ge_0p04_polar_ge_50": ratio(
            sum(
                1
                for row in observation_rows
                if to_float(row.get("projected_area_ratio")) is not None
                and to_float(row.get("projected_area_ratio")) >= 0.04
                and to_float(row.get("max_polar_angle_deg")) is not None
                and to_float(row.get("max_polar_angle_deg")) >= 50.0
            ),
            board_observation_count,
        ),
        "mean_outer_rmse": None,
        "median_outer_rmse": None,
        "p90_outer_rmse": None,
        "notes": "" if rows else "missing Stage5 point CSV",
    }
    per_board_rows = []
    for board_id, stats in sorted(per_board_counts.items()):
        area_values = stats["area_ratios"]
        polar_values = stats["max_polars"]
        per_board_rows.append({
            "stage": "stage5",
            "run_dir": str(run_dir),
            "board_id": board_id,
            "observation_count": stats["observation_count"],
            "point_count": stats["point_count"],
            "outer_point_count": stats["outer_point_count"],
            "internal_point_count": stats["internal_point_count"],
            "mean_projected_area_ratio": mean(area_values),
            "p90_projected_area_ratio": percentile(area_values, 0.9),
            "mean_max_polar_angle_deg": mean(polar_values),
            "p90_max_polar_angle_deg": percentile(polar_values, 0.9),
        })
    return summary, observation_rows, per_board_rows


def stage6_summary(run_dir: Path) -> Tuple[Dict[str, object], List[Dict[str, object]], List[Dict[str, object]]]:
    summary_txt = read_key_values(run_dir / "stereo_extrinsic_summary.txt")
    rows = read_csv(run_dir / "stereo_pair_board_consistency.csv")
    selected_rows = read_csv(run_dir / "stereo_pair_board_trial_selected_boards.csv")
    selected_keys = {
        (to_int(row.get("pair_index")), to_int(row.get("board_id")))
        for row in selected_rows
    }
    selected_keys.discard((None, None))

    pair_to_boards: Dict[int, set] = defaultdict(set)
    selected_pair_to_boards: Dict[int, set] = defaultdict(set)
    per_board_counts: Dict[int, Dict[str, object]] = defaultdict(lambda: {
        "observation_count": 0,
        "selected_count": 0,
        "point_count": 0,
        "cam0_outer_point_count": 0,
        "cam1_outer_point_count": 0,
        "outer_rmses": [],
    })
    observation_rows: List[Dict[str, object]] = []
    total_points = 0
    total_cam0 = 0
    total_cam1 = 0
    outer_rmses: List[float] = []

    for row in rows:
        if row.get("split", "training") != "training":
            continue
        pair_index = to_int(row.get("pair_index"))
        board_id = to_int(row.get("board_id"))
        if pair_index is None or board_id is None:
            continue
        key = (pair_index, board_id)
        selected = key in selected_keys
        pair_to_boards[pair_index].add(board_id)
        if selected:
            selected_pair_to_boards[pair_index].add(board_id)
        cam0 = to_int(row.get("cam0_outer_point_count")) or 0
        cam1 = to_int(row.get("cam1_outer_point_count")) or 0
        points = to_int(row.get("global_outer_point_count"))
        if points is None:
            points = cam0 + cam1
        total_points += points
        total_cam0 += cam0
        total_cam1 += cam1
        rmse = to_float(row.get("global_outer_rmse"))
        if rmse is not None:
            outer_rmses.append(rmse)
        observation_rows.append({
            "stage": "stage6",
            "run_dir": str(run_dir),
            "frame_or_pair_index": pair_index,
            "frame_or_pair_label": f"{row.get('left_frame_label', '')}|{row.get('right_frame_label', '')}",
            "board_id": board_id,
            "selected": 1 if selected else 0,
            "point_count": points,
            "outer_point_count": points,
            "internal_point_count": 0,
            "cam0_outer_point_count": cam0,
            "cam1_outer_point_count": cam1,
            "outer_rmse": rmse,
            "shared_board": row.get("shared_board", ""),
        })
        per_board = per_board_counts[board_id]
        per_board["observation_count"] = int(per_board["observation_count"]) + 1
        per_board["selected_count"] = int(per_board["selected_count"]) + (1 if selected else 0)
        per_board["point_count"] = int(per_board["point_count"]) + points
        per_board["cam0_outer_point_count"] = int(per_board["cam0_outer_point_count"]) + cam0
        per_board["cam1_outer_point_count"] = int(per_board["cam1_outer_point_count"]) + cam1
        if rmse is not None:
            per_board["outer_rmses"].append(rmse)

    boards_per_pair = [len(boards) for boards in pair_to_boards.values()]
    selected_boards_per_pair = [
        len(selected_pair_to_boards.get(pair_index, set())) for pair_index in pair_to_boards
    ]
    selected_count = sum(1 for row in observation_rows if row.get("selected") == 1)
    pair_count = len(pair_to_boards)
    summary = {
        "stage": "stage6",
        "run_dir": str(run_dir),
        "dataset_label": run_dir.name,
        "training_input": f"{summary_txt.get('left_image_path', '')}|{summary_txt.get('right_image_path', '')}",
        "holdout_input": "",
        "observation_scope": "stage6_training_pair_board_consistency",
        "frame_or_pair_count": pair_count,
        "board_observation_count": len(observation_rows),
        "selected_board_observation_count": selected_count,
        "unique_board_count": len(per_board_counts),
        "point_count": total_points,
        "outer_point_count": total_points,
        "internal_point_count": 0,
        "mean_boards_per_frame_or_pair": mean(boards_per_pair),
        "max_boards_per_frame_or_pair": max(boards_per_pair) if boards_per_pair else None,
        "boards_per_frame_or_pair_hist": hist_string(Counter(boards_per_pair)),
        "single_board_frame_or_pair_ratio": ratio(sum(1 for count in boards_per_pair if count == 1), pair_count),
        "multi_board_frame_or_pair_ratio": ratio(sum(1 for count in boards_per_pair if count >= 2), pair_count),
        "four_plus_board_frame_or_pair_ratio": ratio(sum(1 for count in boards_per_pair if count >= 4), pair_count),
        "mean_points_per_board_observation": total_points / len(observation_rows) if observation_rows else None,
        "mean_projected_area_ratio": None,
        "median_projected_area_ratio": None,
        "p90_projected_area_ratio": None,
        "large_area_ratio_ge_0p015": None,
        "large_area_ratio_ge_0p02": None,
        "large_area_ratio_ge_0p04": None,
        "large_area_ratio_ge_0p06": None,
        "mean_polar_angle_deg": None,
        "median_max_polar_angle_deg": None,
        "p90_max_polar_angle_deg": None,
        "high_polar_ratio_ge_50deg": None,
        "high_polar_ratio_ge_60deg": None,
        "high_polar_ratio_ge_70deg": None,
        "edge_margin_ratio_le_0p05": None,
        "close_edge_like_ratio_area_ge_0p015_polar_ge_60": None,
        "close_edge_like_ratio_area_ge_0p04_polar_ge_50": None,
        "mean_outer_rmse": mean(outer_rmses),
        "median_outer_rmse": percentile(outer_rmses, 0.5),
        "p90_outer_rmse": percentile(outer_rmses, 0.9),
        "notes": (
            "Stage6 current outputs do not include image-space area/polar fields; "
            f"selected_boards_per_pair_hist={hist_string(Counter(selected_boards_per_pair))}"
        ),
    }
    per_board_rows = []
    for board_id, stats in sorted(per_board_counts.items()):
        rmses = stats["outer_rmses"]
        per_board_rows.append({
            "stage": "stage6",
            "run_dir": str(run_dir),
            "board_id": board_id,
            "observation_count": stats["observation_count"],
            "selected_count": stats["selected_count"],
            "point_count": stats["point_count"],
            "outer_point_count": stats["point_count"],
            "internal_point_count": 0,
            "cam0_outer_point_count": stats["cam0_outer_point_count"],
            "cam1_outer_point_count": stats["cam1_outer_point_count"],
            "mean_outer_rmse": mean(rmses),
            "p90_outer_rmse": percentile(rmses, 0.9),
        })
    return summary, observation_rows, per_board_rows


def detect_stage(run_dir: Path) -> Optional[str]:
    if (run_dir / "backend_training_points.csv").exists() or (
        run_dir / "benchmark_training_points.csv"
    ).exists():
        return "stage5"
    if (run_dir / "stereo_pair_board_consistency.csv").exists():
        return "stage6"
    return None


def collect_run_dirs(args: argparse.Namespace) -> List[Path]:
    runs: List[Path] = []
    for run_dir in args.run_dir:
        path = run_dir.resolve()
        if path.is_dir():
            runs.append(path)
    for input_dir in args.input_dir:
        root = input_dir.resolve()
        if not root.is_dir():
            continue
        for child in sorted(root.iterdir()):
            if child.is_dir() and detect_stage(child) is not None:
                runs.append(child)
    seen = set()
    unique: List[Path] = []
    for run in runs:
        if run not in seen:
            unique.append(run)
            seen.add(run)
    return unique


def write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: fmt(row.get(field)) for field in fieldnames})


def write_markdown(path: Path, summaries: List[Dict[str, object]]) -> None:
    stage5 = [row for row in summaries if row.get("stage") == "stage5"]
    stage6 = [row for row in summaries if row.get("stage") == "stage6"]
    lines: List[str] = []
    lines.append("# Dataset-Level Board Observation Distribution Analysis")
    lines.append("")
    lines.append("This report summarizes board-observation structure for existing Stage5 and Stage6 output directories.")
    lines.append("")
    lines.append("## Interpretation Notes")
    lines.append("")
    lines.append("- Stage5 area ratio and polar angle are recomputed from backend training observed pixels and optimized DS intrinsics.")
    lines.append("- Stage5 `close_edge_like` is reported with two thresholds: area >= 0.015 + max polar >= 60 deg, and the stricter area >= 0.04 + max polar >= 50 deg.")
    lines.append("- Stage6 current outputs do not contain image-space board corners, so area/polar fields are NA there.")
    lines.append("- Stage6 still reports pair-board distribution, selected pair-board count, point count, and outer RMSE structure.")
    lines.append("")

    def append_table(title: str, rows: List[Dict[str, object]], fields: Sequence[str]) -> None:
        lines.append(f"## {title}")
        lines.append("")
        if not rows:
            lines.append("No runs found.")
            lines.append("")
            return
        lines.append("| " + " | ".join(fields) + " |")
        lines.append("|" + "|".join("---" for _ in fields) + "|")
        for row in rows:
            lines.append("| " + " | ".join(fmt(row.get(field), 5) for field in fields) + " |")
        lines.append("")

    stage5_sorted = sorted(
        stage5,
        key=lambda row: (
            str(row.get("training_input", "")),
            str(row.get("run_dir", "")),
        ),
    )
    stage6_sorted = sorted(
        stage6,
        key=lambda row: (
            str(row.get("training_input", "")),
            str(row.get("run_dir", "")),
        ),
    )
    append_table(
        "Stage5 Runs",
        stage5_sorted,
        [
            "dataset_label",
            "frame_or_pair_count",
            "board_observation_count",
            "boards_per_frame_or_pair_hist",
            "mean_projected_area_ratio",
            "p90_projected_area_ratio",
            "p90_max_polar_angle_deg",
            "large_area_ratio_ge_0p015",
            "high_polar_ratio_ge_60deg",
            "close_edge_like_ratio_area_ge_0p015_polar_ge_60",
        ],
    )
    append_table(
        "Stage6 Runs",
        stage6_sorted,
        [
            "dataset_label",
            "frame_or_pair_count",
            "board_observation_count",
            "selected_board_observation_count",
            "boards_per_frame_or_pair_hist",
            "mean_outer_rmse",
            "p90_outer_rmse",
        ],
    )
    if stage5:
        high_close = sorted(
            stage5,
            key=lambda row: to_float(row.get("close_edge_like_ratio_area_ge_0p015_polar_ge_60")) or -1.0,
            reverse=True,
        )[:8]
        append_table(
            "Stage5 Highest Close-Edge-Like Runs",
            high_close,
            [
                "dataset_label",
                "close_edge_like_ratio_area_ge_0p015_polar_ge_60",
                "large_area_ratio_ge_0p015",
                "high_polar_ratio_ge_60deg",
                "p90_projected_area_ratio",
                "p90_max_polar_angle_deg",
            ],
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze dataset-level board observation distributions for Stage5/Stage6 outputs."
    )
    parser.add_argument("--input-dir", action="append", type=Path, default=[])
    parser.add_argument("--run-dir", action="append", type=Path, default=[])
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    repo_root = Path.cwd()
    run_dirs = collect_run_dirs(args)
    summaries: List[Dict[str, object]] = []
    observations: List[Dict[str, object]] = []
    per_board: List[Dict[str, object]] = []
    hist_rows: List[Dict[str, object]] = []

    for run_dir in run_dirs:
        stage = detect_stage(run_dir)
        if stage == "stage5":
            summary, obs_rows, board_rows = stage5_summary(run_dir, repo_root)
        elif stage == "stage6":
            summary, obs_rows, board_rows = stage6_summary(run_dir)
        else:
            continue
        summaries.append(summary)
        observations.extend(obs_rows)
        per_board.extend(board_rows)
        counts = Counter()
        for obs in obs_rows:
            key = int(obs.get("frame_or_pair_index", -1))
            counts[key] += 1
        hist = Counter(counts.values())
        for bin_value, count in sorted(hist.items()):
            hist_rows.append({
                "stage": summary["stage"],
                "run_dir": summary["run_dir"],
                "dataset_label": summary["dataset_label"],
                "boards_per_frame_or_pair": bin_value,
                "frame_or_pair_count": count,
            })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(
        args.output_dir / "dataset_board_observation_distribution_summary.csv",
        summaries,
        SUMMARY_FIELDS,
    )
    observation_fields = [
        "stage",
        "run_dir",
        "frame_or_pair_index",
        "frame_or_pair_label",
        "board_id",
        "selected",
        "point_count",
        "outer_point_count",
        "internal_point_count",
        "cam0_outer_point_count",
        "cam1_outer_point_count",
        "projected_area_ratio",
        "mean_polar_angle_deg",
        "max_polar_angle_deg",
        "edge_margin_ratio",
        "outer_rmse",
        "shared_board",
    ]
    write_csv(
        args.output_dir / "dataset_board_observation_distribution_per_observation.csv",
        observations,
        observation_fields,
    )
    per_board_fields = [
        "stage",
        "run_dir",
        "board_id",
        "observation_count",
        "selected_count",
        "point_count",
        "outer_point_count",
        "internal_point_count",
        "cam0_outer_point_count",
        "cam1_outer_point_count",
        "mean_projected_area_ratio",
        "p90_projected_area_ratio",
        "mean_max_polar_angle_deg",
        "p90_max_polar_angle_deg",
        "mean_outer_rmse",
        "p90_outer_rmse",
    ]
    write_csv(
        args.output_dir / "dataset_board_observation_distribution_per_board.csv",
        per_board,
        per_board_fields,
    )
    write_csv(
        args.output_dir / "dataset_board_observation_distribution_histogram.csv",
        hist_rows,
        [
            "stage",
            "run_dir",
            "dataset_label",
            "boards_per_frame_or_pair",
            "frame_or_pair_count",
        ],
    )
    write_markdown(
        args.output_dir / "dataset_board_observation_distribution_report.md",
        summaries,
    )
    print(f"analyzed_run_count: {len(summaries)}")
    print(f"summary_csv: {args.output_dir / 'dataset_board_observation_distribution_summary.csv'}")
    print(f"report_md: {args.output_dir / 'dataset_board_observation_distribution_report.md'}")


if __name__ == "__main__":
    main()
