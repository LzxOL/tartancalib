#!/usr/bin/env python3
"""Compare frontend outer-corner coverage curves across Stage5 results.

Each input is an ``auto_camera_initialization_bootstrap_views.csv`` file.  A
frame-board group contributes its four observed physical outer corners.  The
solid curve uses all available board groups (multi-board); the dashed curve is
the median of board-specific single-board baselines.  Both curves use the same
20x20 valid-image grid and deterministic random-order permutations as the
existing ``corner_distribution_and_coverage_efficiency`` experiment.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter


DATASET_COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#6A3D9A"]
BUDGET_FRACTIONS = np.linspace(0.0, 1.0, 21)


def stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def parse_dataset(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("dataset must have NAME=VIEWS_CSV form")
    name, raw_path = value.split("=", 1)
    if not name.strip() or not raw_path.strip():
        raise argparse.ArgumentTypeError("dataset must have NAME=VIEWS_CSV form")
    return name.strip(), Path(raw_path.strip())


def parse_size(value: str) -> tuple[str, tuple[float, float]]:
    if "=" not in value or "x" not in value.lower():
        raise argparse.ArgumentTypeError("dataset size must have NAME=WIDTHxHEIGHT form")
    name, raw_size = value.split("=", 1)
    width_text, height_text = raw_size.lower().split("x", 1)
    try:
        size = (float(width_text), float(height_text))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("dataset size must have numeric WIDTHxHEIGHT") from exc
    if min(size) <= 0:
        raise argparse.ArgumentTypeError("dataset size must be positive")
    return name.strip(), size


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", action="append", type=parse_dataset, required=True,
                        help="Repeat NAME=auto_camera_initialization_bootstrap_views.csv")
    parser.add_argument("--dataset-size", action="append", type=parse_size, default=[],
                        help="Optional per-dataset NAME=WIDTHxHEIGHT override")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--width", type=float, default=4512.0)
    parser.add_argument("--height", type=float, default=4512.0)
    parser.add_argument("--grid-size", type=int, default=20)
    parser.add_argument("--permutations", type=int, default=100)
    parser.add_argument("--coordinate-space", choices=("normalized_disk", "pixel"),
                        default="normalized_disk",
                        help="Resolution-independent unit-disk grid (default) or legacy pixel grid.")
    parser.add_argument("--dpi", type=int, default=400)
    parser.add_argument("--include-unsuccessful", action="store_true",
                        help="Include rows marked pose_init_success=0")
    return parser.parse_args()


def load_rows(path: Path, include_unsuccessful: bool) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {"frame_index", "frame_label", "board_id", "corner_index", "x", "y", "pose_init_success"}
    missing = required - set(rows[0]) if rows else required
    if missing:
        raise RuntimeError(f"Missing columns in {path}: {sorted(missing)}")
    if not include_unsuccessful:
        rows = [row for row in rows if row.get("pose_init_success", "1") == "1"]
    if not rows:
        raise RuntimeError(f"No rows remain after filtering {path}")
    return rows


def valid_grid(width: float, height: float, grid_size: int,
               coordinate_space: str) -> set[tuple[int, int]]:
    if coordinate_space == "normalized_disk":
        return {
            (gx, gy)
            for gy in range(grid_size)
            for gx in range(grid_size)
            if math.hypot(
                -1.0 + (gx + 0.5) * 2.0 / grid_size,
                -1.0 + (gy + 0.5) * 2.0 / grid_size,
            ) <= 1.0
        }
    cu, cv = width / 2.0, height / 2.0
    radius = min(cu, cv, width - cu, height - cv)
    valid = set()
    for gy in range(grid_size):
        for gx in range(grid_size):
            u = (gx + 0.5) * width / grid_size
            v = (gy + 0.5) * height / grid_size
            if math.hypot(u - cu, v - cv) <= radius:
                valid.add((gx, gy))
    return valid


def occupied(rows: list[dict[str, str]], width: float, height: float, grid_size: int,
             valid: set[tuple[int, int]], coordinate_space: str) -> set[tuple[int, int]]:
    cells = set()
    radius = min(width / 2.0, height / 2.0)
    for row in rows:
        x, y = float(row["x"]), float(row["y"])
        if not (math.isfinite(x) and math.isfinite(y)):
            continue
        if coordinate_space == "normalized_disk":
            u = (x - width / 2.0) / radius
            v = (y - height / 2.0) / radius
            if u * u + v * v > 1.0:
                continue
            gx = int((u + 1.0) * 0.5 * grid_size)
            gy = int((v + 1.0) * 0.5 * grid_size)
        else:
            gx = int(x / width * grid_size)
            gy = int(y / height * grid_size)
        gx = min(grid_size - 1, max(0, gx))
        gy = min(grid_size - 1, max(0, gy))
        if (gx, gy) in valid:
            cells.add((gx, gy))
    return cells


def group_rows(rows: list[dict[str, str]]) -> dict[tuple[str, int], list[dict[str, str]]]:
    grouped: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["frame_label"], int(row["board_id"]))].append(row)
    return grouped


def curves_for_groups(groups: dict[tuple[str, int], list[dict[str, str]]], width: float,
                      height: float, grid_size: int, valid: set[tuple[int, int]],
                      permutations: int, tag: str, coordinate_space: str) -> np.ndarray:
    keys = list(groups)
    if not keys:
        raise RuntimeError(f"No frame-board groups found for {tag}")
    values = []
    for seed in range(permutations):
        ordered = sorted(keys, key=lambda item: stable_hash(f"coverage|{tag}|{seed}|{item[0]}|B{item[1]}"))
        curve = []
        for fraction in BUDGET_FRACTIONS:
            if fraction == 0:
                curve.append(0.0)
                continue
            count = max(1, math.ceil(fraction * len(ordered)))
            selected = [row for key in ordered[:count] for row in groups[key]]
            curve.append(len(occupied(selected, width, height, grid_size, valid,
                                      coordinate_space)) / len(valid))
        values.append(curve)
    return np.asarray(values, dtype=float)


def dataset_curves(rows: list[dict[str, str]], width: float, height: float, grid_size: int,
                   valid: set[tuple[int, int]], permutations: int, tag: str,
                   coordinate_space: str) -> tuple[np.ndarray, np.ndarray, dict]:
    all_groups = group_rows(rows)
    multi = curves_for_groups(all_groups, width, height, grid_size, valid, permutations,
                              f"{tag}|multi", coordinate_space)
    by_board: dict[int, dict[tuple[str, int], list[dict[str, str]]] ] = defaultdict(dict)
    for key, value in all_groups.items():
        by_board[key[1]][key] = value
    single = []
    for board_id, groups in sorted(by_board.items()):
        single.append(curves_for_groups(groups, width, height, grid_size, valid, permutations,
                                        f"{tag}|single|B{board_id}", coordinate_space))
    single_samples = np.concatenate(single, axis=0)
    summary = {
        "point_count": len(rows),
        "frame_count": len({row["frame_index"] for row in rows}),
        "frame_board_count": len(all_groups),
        "board_ids": sorted(by_board),
        "valid_grid_cells": len(valid),
        "coordinate_space": coordinate_space,
        "multi_final_median": float(np.median(multi[:, -1])),
        "single_final_median": float(np.median(single_samples[:, -1])),
        "final_median_gain": float(np.median(multi[:, -1]) - np.median(single_samples[:, -1])),
        "multi_auc": float(np.trapezoid(np.median(multi, axis=0), BUDGET_FRACTIONS)),
        "single_auc": float(np.trapezoid(np.median(single_samples, axis=0), BUDGET_FRACTIONS)),
    }
    return multi, single_samples, summary


def configure_style() -> None:
    plt.rcParams.update({
        "font.family": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 1.5,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def make_figure(results: list[tuple[str, np.ndarray, np.ndarray, dict]], output_dir: Path,
                dpi: int) -> None:
    configure_style()
    x = 100.0 * BUDGET_FRACTIONS
    fig, ax = plt.subplots(figsize=(10.5, 6.2), constrained_layout=True)
    for index, (name, multi, single, _summary) in enumerate(results):
        color = DATASET_COLORS[index % len(DATASET_COLORS)]
        multi_median = np.median(multi, axis=0)
        multi_q25, multi_q75 = np.quantile(multi, [0.25, 0.75], axis=0)
        single_median = np.median(single, axis=0)
        single_q25, single_q75 = np.quantile(single, [0.25, 0.75], axis=0)
        ax.fill_between(x, multi_q25, multi_q75, color=color, alpha=0.10, linewidth=0)
        ax.fill_between(x, single_q25, single_q75, color=color, alpha=0.045, linewidth=0)
        ax.plot(x, multi_median, color=color, lw=2.2, marker="o", ms=4.0,
                markevery=4, label=f"{name} · multi-board")
        ax.plot(x, single_median, color=color, lw=1.7, ls=(0, (5, 2.5)), marker="s",
                ms=3.4, markevery=4, alpha=0.80, label=f"{name} · single-board")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("Frame–board observation budget (%)")
    ax.set_ylabel("Valid-grid coverage")
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax.grid(axis="y", color="#D9DDE3", linewidth=0.7, alpha=0.8)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", fontsize=8, ncol=2, handlelength=2.5)
    ax.set_title("Frontend outer-corner coverage efficiency across datasets")
    fig.savefig(output_dir / "frontend_outer_corner_coverage_curve_comparison.png",
                dpi=dpi, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(output_dir / "frontend_outer_corner_coverage_curve_comparison.pdf",
                bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def write_curve_csv(results: list[tuple[str, np.ndarray, np.ndarray, dict]], output_dir: Path) -> None:
    with (output_dir / "frontend_outer_corner_coverage_curve.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["dataset", "budget_percent", "multi_median", "multi_q25", "multi_q75",
                         "single_median", "single_q25", "single_q75"])
        for name, multi, single, _summary in results:
            for index, budget in enumerate(100.0 * BUDGET_FRACTIONS):
                writer.writerow([
                    name, budget,
                    float(np.median(multi[:, index])), float(np.quantile(multi[:, index], 0.25)),
                    float(np.quantile(multi[:, index], 0.75)), float(np.median(single[:, index])),
                    float(np.quantile(single[:, index], 0.25)), float(np.quantile(single[:, index], 0.75)),
                ])


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    size_by_name = dict(args.dataset_size)
    results = []
    summaries = {}
    for name, path in args.dataset:
        width, height = size_by_name.get(name, (args.width, args.height))
        valid = valid_grid(width, height, args.grid_size, args.coordinate_space)
        rows = load_rows(path, args.include_unsuccessful)
        multi, single, summary = dataset_curves(rows, width, height, args.grid_size,
                                                valid, args.permutations, name,
                                                args.coordinate_space)
        summary = {"image_width": width, "image_height": height, **summary}
        results.append((name, multi, single, summary))
        summaries[name] = {"views_csv": str(path.resolve()), **summary}
    write_curve_csv(results, args.output_dir)
    (args.output_dir / "frontend_outer_corner_coverage_curve_summary.json").write_text(
        json.dumps(summaries, indent=2) + "\n", encoding="utf-8"
    )
    make_figure(results, args.output_dir, args.dpi)
    print(json.dumps(summaries, indent=2))
    print(f"png: {args.output_dir / 'frontend_outer_corner_coverage_curve_comparison.png'}")
    print(f"pdf: {args.output_dir / 'frontend_outer_corner_coverage_curve_comparison.pdf'}")


if __name__ == "__main__":
    main()
