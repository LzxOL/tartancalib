#!/usr/bin/env python3
"""Compute normalized image-plane coverage metrics for a holdout point CSV.

The input is the frozen ``benchmark_holdout_points.csv`` artifact.  This
script intentionally measures the observed image locations only; it does not
rerun calibration or pose fitting.  Coordinates are normalized by the
inscribed image radius and measured on a fixed unit-disc grid, so results from
different image resolutions/aspect ratios are comparable.  The implementation
uses only NumPy so it can be called by the repository's normal ``python3``
wrapper without a SciPy installation.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def convex_hull_area(points: np.ndarray) -> float:
    """Return the 2-D convex-hull area using a monotonic-chain hull."""
    if len(points) < 3:
        return 0.0
    unique = sorted({(float(x), float(y)) for x, y in points})
    if len(unique) < 3:
        return 0.0

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for point in unique:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    upper = []
    for point in reversed(unique):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    hull = lower[:-1] + upper[:-1]
    return abs(sum(
        hull[i][0] * hull[(i + 1) % len(hull)][1]
        - hull[(i + 1) % len(hull)][0] * hull[i][1]
        for i in range(len(hull))
    )) * 0.5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--points", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--width", type=float, default=4512.0)
    parser.add_argument("--height", type=float, default=4512.0)
    parser.add_argument("--grid", type=int, default=20)
    parser.add_argument("--angular-sectors", type=int, default=36)
    parser.add_argument("--peripheral-rho", type=float, default=0.7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.points.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"No points found in {args.points}")

    x = np.asarray([float(row["observed_x"]) for row in rows], dtype=float)
    y = np.asarray([float(row["observed_y"]) for row in rows], dtype=float)
    cx, cy = args.width * 0.5, args.height * 0.5
    radius = min(cx, cy)
    rho = np.hypot((x - cx) / radius, (y - cy) / radius)
    angle = np.mod(np.arctan2(y - cy, x - cx), 2.0 * np.pi)
    finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(rho)
    valid = finite & (rho <= 1.0)
    if not np.any(valid):
        raise RuntimeError("No finite points inside the normalized image disc")

    normalized_x = (x[valid] - cx) / radius
    normalized_y = (y[valid] - cy) / radius
    grid_x = np.clip(((normalized_x + 1.0) * 0.5 * args.grid).astype(int), 0, args.grid - 1)
    grid_y = np.clip(((normalized_y + 1.0) * 0.5 * args.grid).astype(int), 0, args.grid - 1)
    valid_cells = {
        (cell_x, cell_y)
        for cell_y in range(args.grid)
        for cell_x in range(args.grid)
        if np.hypot(
            -1.0 + (cell_x + 0.5) * 2.0 / args.grid,
            -1.0 + (cell_y + 0.5) * 2.0 / args.grid,
        ) <= 1.0
    }
    occupied = len(set(zip(grid_x.tolist(), grid_y.tolist())) & valid_cells)
    sector = np.floor(angle[valid] / (2.0 * np.pi) * args.angular_sectors).astype(int)
    sector_coverage = len(set(np.clip(sector, 0, args.angular_sectors - 1).tolist()))
    point_types = {}
    for row in rows:
        point_types[row.get("point_type", "unknown")] = point_types.get(row.get("point_type", "unknown"), 0) + 1

    summary = {
        "point_count": len(rows),
        "point_count_by_type": point_types,
        "frame_count": len({row["frame_index"] for row in rows}),
        "frame_board_count": len({(row["frame_index"], row["board_id"]) for row in rows}),
        "valid_unit_disc_fraction": float(np.mean(valid)),
        "radial_min": float(np.min(rho[finite])),
        "radial_max": float(np.max(rho[finite])),
        "radial_p05": float(np.percentile(rho[finite], 5)),
        "radial_p50": float(np.percentile(rho[finite], 50)),
        "radial_p95": float(np.percentile(rho[finite], 95)),
        "peripheral_rho_threshold": args.peripheral_rho,
        "peripheral_fraction_rho_ge_threshold": float(np.mean(rho[valid] >= args.peripheral_rho)),
        "angular_sector_count": args.angular_sectors,
        "angular_coverage_fraction": float(sector_coverage / args.angular_sectors),
        "grid_occupied_cells": occupied,
        "grid_total_cells": len(valid_cells),
        "grid_occupancy_fraction": float(occupied / len(valid_cells)),
        "convex_hull_area_fraction": float(convex_hull_area(np.c_[normalized_x, normalized_y]) / np.pi),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "holdout_distribution_metrics.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    with (args.output_dir / "holdout_distribution_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        writer.writerows(summary.items())
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
