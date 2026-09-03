#!/usr/bin/env python3
"""Audit image-plane corner distributions in BabelCalib MAT exports.

The report is intentionally independent of camera intrinsics.  The radial
statistics are image-normalized radius proxies, not calibrated polar angles.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import loadmat


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mat_files", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--grid-rows", type=int, default=8)
    parser.add_argument("--grid-cols", type=int, default=8)
    return parser.parse_args()


def field(obj: Any, name: str) -> Any:
    if hasattr(obj, name):
        return getattr(obj, name)
    if isinstance(obj, np.void) and obj.dtype.names and name in obj.dtype.names:
        return obj[name]
    raise KeyError(name)


def struct_list(value: Any) -> list[Any]:
    arr = np.asarray(value)
    return list(arr.reshape(-1))


def finite_points(corner: Any) -> tuple[np.ndarray, np.ndarray]:
    xy = np.asarray(field(corner, "x"), dtype=float)
    correspondence = np.asarray(field(corner, "cspond"), dtype=float)
    if xy.size == 0:
        return np.empty((0, 2)), np.empty((0, 2), dtype=int)
    if xy.ndim == 1:
        xy = xy.reshape(2, -1)
    if correspondence.ndim == 1:
        correspondence = correspondence.reshape(2, -1)
    if xy.shape[0] != 2 or correspondence.shape[0] != 2:
        raise ValueError(f"expected 2xN arrays, got x={xy.shape}, cspond={correspondence.shape}")
    if xy.shape[1] != correspondence.shape[1]:
        raise ValueError("x and cspond have different point counts")
    valid = np.isfinite(xy).all(axis=0) & np.isfinite(correspondence).all(axis=0)
    return xy[:, valid].T, correspondence[:, valid].T.astype(int)


def quantiles(values: np.ndarray) -> dict[str, float | None]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not values.size:
        return {key: None for key in ("min", "p05", "median", "p95", "max", "mean", "std")}
    return {
        "min": float(np.min(values)),
        "p05": float(np.percentile(values, 5)),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze(mat_path: Path, args: argparse.Namespace) -> dict[str, Any]:
    data = loadmat(mat_path, squeeze_me=False, struct_as_record=False)
    for name in ("corners", "boards", "imgsize"):
        if name not in data:
            raise ValueError(f"{mat_path}: missing {name}")

    corners = struct_list(data["corners"])
    boards = struct_list(data["boards"])
    image_size = np.asarray(data["imgsize"], dtype=float).reshape(-1)
    if image_size.size != 2:
        raise ValueError(f"{mat_path}: imgsize must contain width and height")
    width, height = float(image_size[0]), float(image_size[1])
    center = np.array([width / 2.0, height / 2.0])
    half_size = np.array([width / 2.0, height / 2.0])

    board_point_counts = {
        board_idx: int(np.asarray(field(board, "X")).reshape(2, -1).shape[1])
        for board_idx, board in enumerate(boards, start=1)
    }
    board_rows: dict[int, dict[str, Any]] = {
        board_idx: {
            "mat_file": mat_path.name,
            "board_id": board_idx,
            "board_definition_point_count": point_count,
            "observed_frame_count": 0,
            "observed_point_count": 0,
            "frame_point_count_mean": 0.0,
            "x": [],
            "y": [],
            "r_norm": [],
        }
        for board_idx, point_count in board_point_counts.items()
    }
    frame_rows: list[dict[str, Any]] = []
    all_xy: list[np.ndarray] = []
    all_r: list[float] = []
    grid = np.zeros((args.grid_rows, args.grid_cols), dtype=int)
    radial_edges = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.8, np.inf])
    radial_counts = np.zeros(radial_edges.size - 1, dtype=int)

    for frame_idx, corner in enumerate(corners, start=1):
        xy, correspondence = finite_points(corner)
        r = np.linalg.norm((xy - center) / half_size, axis=1) if len(xy) else np.empty(0)
        all_xy.append(xy)
        all_r.extend(r.tolist())
        if len(xy):
            all_valid = (xy[:, 0] >= 0) & (xy[:, 0] <= width) & (xy[:, 1] >= 0) & (xy[:, 1] <= height)
            grid_x = np.clip((xy[:, 0] / width * args.grid_cols).astype(int), 0, args.grid_cols - 1)
            grid_y = np.clip((xy[:, 1] / height * args.grid_rows).astype(int), 0, args.grid_rows - 1)
            for gx, gy in zip(grid_x[all_valid], grid_y[all_valid]):
                grid[gy, gx] += 1
            radial_counts += np.histogram(r, radial_edges)[0]

        counts = {board_idx: 0 for board_idx in board_point_counts}
        for point_id, board_idx in correspondence:
            del point_id
            if board_idx in counts:
                counts[board_idx] += 1
        for board_idx, count in counts.items():
            if count:
                board_rows[board_idx]["observed_frame_count"] += 1
        for point, (_, board_idx) in zip(xy, correspondence):
            if board_idx not in board_rows:
                continue
            row = board_rows[board_idx]
            row["observed_point_count"] += 1
            row["x"].append(float(point[0]))
            row["y"].append(float(point[1]))
            row["r_norm"].append(float(np.linalg.norm((point - center) / half_size)))

        if len(xy):
            bbox_min = xy.min(axis=0)
            bbox_max = xy.max(axis=0)
            centroid = xy.mean(axis=0)
        else:
            bbox_min = bbox_max = centroid = np.array([np.nan, np.nan])
        frame_rows.append({
            "mat_file": mat_path.name,
            "frame_id": frame_idx,
            "point_count": int(len(xy)),
            "board_count": int(np.count_nonzero(list(counts.values()))),
            "board_ids": ";".join(str(i) for i, count in counts.items() if count),
            "centroid_x": float(centroid[0]),
            "centroid_y": float(centroid[1]),
            "bbox_min_x": float(bbox_min[0]),
            "bbox_min_y": float(bbox_min[1]),
            "bbox_max_x": float(bbox_max[0]),
            "bbox_max_y": float(bbox_max[1]),
            "bbox_width": float(bbox_max[0] - bbox_min[0]) if len(xy) else 0.0,
            "bbox_height": float(bbox_max[1] - bbox_min[1]) if len(xy) else 0.0,
            "r_norm_median": float(np.median(r)) if len(r) else np.nan,
            "r_norm_p95": float(np.percentile(r, 95)) if len(r) else np.nan,
            "point_count_by_board": ";".join(f"{i}:{counts[i]}" for i in counts if counts[i]),
        })

    xy_all = np.vstack([points for points in all_xy if len(points)]) if any(len(points) for points in all_xy) else np.empty((0, 2))
    for board_idx, row in board_rows.items():
        row["frame_point_count_mean"] = row["observed_point_count"] / max(row["observed_frame_count"], 1)
        for key in ("x", "y", "r_norm"):
            values = np.asarray(row.pop(key), dtype=float)
            row[f"{key}_min"] = quantiles(values)["min"]
            row[f"{key}_median"] = quantiles(values)["median"]
            row[f"{key}_p95"] = quantiles(values)["p95"]
            row[f"{key}_max"] = quantiles(values)["max"]
    global_r = np.asarray(all_r, dtype=float)
    global_bbox = {
        "min_x": float(xy_all[:, 0].min()) if len(xy_all) else None,
        "min_y": float(xy_all[:, 1].min()) if len(xy_all) else None,
        "max_x": float(xy_all[:, 0].max()) if len(xy_all) else None,
        "max_y": float(xy_all[:, 1].max()) if len(xy_all) else None,
    }
    global_bbox["width"] = global_bbox["max_x"] - global_bbox["min_x"] if len(xy_all) else None
    global_bbox["height"] = global_bbox["max_y"] - global_bbox["min_y"] if len(xy_all) else None
    frame_counts = np.array([row["point_count"] for row in frame_rows], dtype=float)
    board_count_per_frame = np.array([row["board_count"] for row in frame_rows], dtype=float)
    summary = {
        "mat_file": str(mat_path.resolve()),
        "frame_count": len(corners),
        "board_count": len(boards),
        "imgsize_width": int(width),
        "imgsize_height": int(height),
        "board_definition_point_counts": board_point_counts,
        "total_point_count": int(len(xy_all)),
        "point_count_per_frame": quantiles(frame_counts),
        "board_count_per_frame": quantiles(board_count_per_frame),
        "frames_with_all_defined_boards": int(np.count_nonzero(board_count_per_frame == len(boards))),
        "global_bbox": global_bbox,
        "global_x": quantiles(xy_all[:, 0]) if len(xy_all) else quantiles(np.empty(0)),
        "global_y": quantiles(xy_all[:, 1]) if len(xy_all) else quantiles(np.empty(0)),
        "image_normalized_radius_proxy": quantiles(global_r),
        "coverage_grid_rows": args.grid_rows,
        "coverage_grid_cols": args.grid_cols,
        "coverage_occupied_cells": int(np.count_nonzero(grid)),
        "coverage_total_cells": int(grid.size),
        "coverage_occupied_fraction": float(np.count_nonzero(grid) / grid.size),
        "radial_bins_normalized_radius": [float(v) if np.isfinite(v) else "inf" for v in radial_edges],
        "radial_bin_counts": radial_counts.tolist(),
        "board_stats": list(board_rows.values()),
        "note": "r_norm is radius normalized by half image width/height; it is not a calibrated polar angle.",
    }
    return {"summary": summary, "frames": frame_rows, "boards": list(board_rows.values()), "grid": grid}


def main() -> None:
    args = parse_args()
    if args.grid_rows < 1 or args.grid_cols < 1:
        raise ValueError("grid dimensions must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    combined = {"datasets": []}
    for mat_path in args.mat_files:
        result = analyze(mat_path.resolve(), args)
        stem = mat_path.parent.name
        (args.output_dir / f"{stem}_summary.json").write_text(json.dumps(result["summary"], indent=2) + "\n")
        write_csv(args.output_dir / f"{stem}_frame_stats.csv", result["frames"])
        write_csv(args.output_dir / f"{stem}_board_stats.csv", result["boards"])
        grid_rows = []
        for row_idx in range(args.grid_rows):
            for col_idx in range(args.grid_cols):
                grid_rows.append({"row": row_idx, "col": col_idx, "point_count": int(result["grid"][row_idx, col_idx])})
        write_csv(args.output_dir / f"{stem}_coverage_grid.csv", grid_rows)
        combined["datasets"].append(result["summary"])
    (args.output_dir / "combined_summary.json").write_text(json.dumps(combined, indent=2) + "\n")
    print(json.dumps(combined, indent=2))


if __name__ == "__main__":
    main()
