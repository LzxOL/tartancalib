#!/usr/bin/env python3
"""Plot spatial reprojection-error scatter heatmaps from Stage5 point diagnostics.

Each plotted marker is one observed calibration point. Its image location
preserves spatial coverage, while its color encodes the corresponding
reprojection residual in pixels. No polar-angle quantity is used.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


VALID_METHODS = ("ours", "kalibr")
VALID_POINT_TYPES = ("all", "outer", "internal")
LOW_ERROR_COLOR_BGR = np.array([255.0, 245.0, 230.0])
HIGH_ERROR_COLOR_BGR = np.array([190.0, 86.0, 30.0])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path,
                        help="Stage5 result directory containing benchmark_holdout_points.csv.")
    parser.add_argument("--points-csv", type=Path,
                        help="Optional explicit point CSV; overrides RESULT_DIR input.")
    parser.add_argument("--output-dir", type=Path,
                        help="Defaults to RESULT_DIR/reprojection_error_heatmaps.")
    parser.add_argument("--methods", default="ours,kalibr",
                        help="Comma-separated methods: ours, kalibr, or both.")
    parser.add_argument("--point-types", default="all,outer,internal",
                        help="Comma-separated point groups: all, outer, internal.")
    parser.add_argument("--split", default="holdout",
                        help="Point CSV split column to retain; use 'all' to keep every split.")
    parser.add_argument("--frame-label",
                        help="Optional exact image stem. Limits the plot to one frame.")
    parser.add_argument("--board-id", type=int,
                        help="Optional board ID. Limits the plot to one board.")
    parser.add_argument("--color-max", type=float,
                        help="Shared colorbar upper bound in pixels. Defaults to selected-data P99.")
    parser.add_argument("--color-percentile", type=float, default=99.0,
                        help="Percentile for automatic shared colorbar maximum.")
    parser.add_argument("--marker-size", type=float, default=12.0,
                        help="Scatter marker area in points squared.")
    parser.add_argument("--alpha", type=float, default=0.88,
                        help="Marker opacity in [0, 1].")
    parser.add_argument("--image-width", type=int,
                        help="Optional fixed image width for common plot extents.")
    parser.add_argument("--image-height", type=int,
                        help="Optional fixed image height for common plot extents.")
    parser.add_argument("--dpi", type=int, default=300,
                        help="PNG resolution.")
    return parser.parse_args()


def comma_values(value: str, allowed: tuple[str, ...], argument: str) -> list[str]:
    values = [item.strip().lower() for item in value.split(",") if item.strip()]
    if not values or any(item not in allowed for item in values):
        raise ValueError(f"{argument} must contain only: {', '.join(allowed)}")
    return list(dict.fromkeys(values))


def load_rows(path: Path, args: argparse.Namespace, methods: list[str]) -> list[dict[str, object]]:
    required = {
        "method", "split", "frame_label", "board_id", "point_type",
        "observed_x", "observed_y", "residual_norm",
    }
    rows: list[dict[str, object]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise RuntimeError(f"{path} lacks required Stage5 point columns")
        for row in reader:
            if row["method"].lower() not in methods:
                continue
            if args.split != "all" and row["split"] != args.split:
                continue
            if args.frame_label and row["frame_label"] != Path(args.frame_label).stem:
                continue
            if args.board_id is not None and int(row["board_id"]) != args.board_id:
                continue
            residual = float(row["residual_norm"])
            observed_x = float(row["observed_x"])
            observed_y = float(row["observed_y"])
            if not (math.isfinite(residual) and math.isfinite(observed_x) and
                    math.isfinite(observed_y) and residual >= 0.0):
                continue
            rows.append({
                "method": row["method"].lower(),
                "point_type": row["point_type"].lower(),
                "frame_label": row["frame_label"],
                "board_id": int(row["board_id"]),
                "x": observed_x,
                "y": observed_y,
                "residual": residual,
            })
    if not rows:
        raise RuntimeError("No finite point observations remain after filtering")
    return rows


def plot_extent(rows: list[dict[str, object]], args: argparse.Namespace) -> tuple[float, float]:
    width = float(args.image_width) if args.image_width else max(float(row["x"]) for row in rows) + 1.0
    height = float(args.image_height) if args.image_height else max(float(row["y"]) for row in rows) + 1.0
    if width <= 1.0 or height <= 1.0:
        raise RuntimeError("Invalid image extent inferred from point observations")
    return width, height


def error_color(value: float) -> np.ndarray:
    """Use one restrained blue scale to keep dense point maps visually quiet."""
    return LOW_ERROR_COLOR_BGR + value * (HIGH_ERROR_COLOR_BGR - LOW_ERROR_COLOR_BGR)


def write_summary(path: Path, rows: list[dict[str, object]], color_max: float) -> None:
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["method"]), str(row["point_type"]))].append(float(row["residual"]))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["method", "point_type", "point_count", "rmse_px", "median_px", "p95_px", "max_px", "color_max_px"])
        for (method, point_type), values in sorted(grouped.items()):
            array = np.asarray(values, dtype=float)
            writer.writerow([
                method, point_type, len(array),
                f"{math.sqrt(float(np.mean(array ** 2))):.9f}",
                f"{float(np.median(array)):.9f}",
                f"{float(np.percentile(array, 95.0)):.9f}",
                f"{float(np.max(array)):.9f}",
                f"{color_max:.9f}",
            ])


def render_plot(rows: list[dict[str, object]], method: str, point_type: str,
                color_max: float, width: float, height: float, args: argparse.Namespace,
                output_path: Path) -> None:
    selected = [
        row for row in rows
        if row["method"] == method and
        (point_type == "all" or row["point_type"] == point_type)
    ]
    if not selected:
        return
    plot_width = 900
    plot_height = max(1, int(round(plot_width * height / width)))
    canvas_height = plot_height + 80
    colorbar_x0 = plot_width + 72
    canvas_width = colorbar_x0 + 220
    canvas = np.full((canvas_height, canvas_width, 3), 255, dtype=np.uint8)
    origin_x, origin_y = 20, 20
    radius = max(1, int(round(math.sqrt(args.marker_size / math.pi))))
    background = np.array([255.0, 255.0, 255.0])

    for row in selected:
        px = origin_x + int(round(float(row["x"]) / width * (plot_width - 1)))
        py = origin_y + int(round(float(row["y"]) / height * (plot_height - 1)))
        value = min(1.0, float(row["residual"]) / color_max)
        color = error_color(value)
        blended = tuple(int(round(component)) for component in (
            args.alpha * color + (1.0 - args.alpha) * background))
        cv2.circle(canvas, (px, py), radius, blended, cv2.FILLED, cv2.LINE_AA)

    colorbar_height = plot_height
    for index in range(colorbar_height):
        value = 1.0 - index / max(1, colorbar_height - 1)
        color = error_color(value).astype(np.uint8)
        canvas[origin_y + index, colorbar_x0:colorbar_x0 + 26] = color
    cv2.rectangle(canvas, (colorbar_x0, origin_y),
                  (colorbar_x0 + 26, origin_y + colorbar_height),
                  (90, 90, 90), 1, cv2.LINE_AA)
    for fraction in np.linspace(0.0, 1.0, 6):
        y_tick = origin_y + int(round((1.0 - fraction) * (colorbar_height - 1)))
        cv2.line(canvas, (colorbar_x0 + 27, y_tick),
                 (colorbar_x0 + 33, y_tick), (70, 70, 70), 1, cv2.LINE_AA)
        label = f"{fraction * color_max:.3g}"
        cv2.putText(canvas, label, (colorbar_x0 + 40, y_tick + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (55, 55, 55), 1, cv2.LINE_AA)
    cv2.putText(canvas, "Reprojection error [px]", (colorbar_x0 - 10, canvas_height - 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (45, 45, 45), 1, cv2.LINE_AA)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    render_scale = args.dpi / 300.0
    if render_scale != 1.0:
        canvas = cv2.resize(canvas, None, fx=render_scale, fy=render_scale,
                            interpolation=cv2.INTER_CUBIC)
    if not cv2.imwrite(str(output_path), canvas):
        raise RuntimeError(f"Failed to write heatmap: {output_path}")


def main() -> int:
    args = parse_args()
    if args.points_csv is None and args.result_dir is None:
        raise ValueError("Specify --result-dir or --points-csv")
    if not 0.0 < args.color_percentile <= 100.0:
        raise ValueError("--color-percentile must be in (0, 100]")
    if not 0.0 <= args.alpha <= 1.0 or args.marker_size <= 0.0 or args.dpi <= 0:
        raise ValueError("--alpha, --marker-size, and --dpi are invalid")

    methods = comma_values(args.methods, VALID_METHODS, "--methods")
    point_types = comma_values(args.point_types, VALID_POINT_TYPES, "--point-types")
    points_csv = (args.points_csv or args.result_dir / "benchmark_holdout_points.csv").resolve()
    if not points_csv.is_file():
        raise FileNotFoundError(f"Point CSV does not exist: {points_csv}")
    output_dir = (args.output_dir or
                  args.result_dir / "reprojection_error_heatmaps").resolve()
    rows = load_rows(points_csv, args, methods)
    selected_residuals = np.asarray([float(row["residual"]) for row in rows])
    color_max = (args.color_max if args.color_max is not None else
                 float(np.percentile(selected_residuals, args.color_percentile)))
    if not math.isfinite(color_max) or color_max <= 0.0:
        color_max = max(1e-6, float(np.max(selected_residuals)))
    width, height = plot_extent(rows, args)

    for method in methods:
        for point_type in point_types:
            render_plot(rows, method, point_type, color_max, width, height, args,
                        output_dir / f"{method}_{point_type}_reprojection_error_heatmap.png")
    output_dir.mkdir(parents=True, exist_ok=True)
    write_summary(output_dir / "reprojection_error_heatmap_summary.csv", rows, color_max)
    (output_dir / "README.txt").write_text(
        "Markers are observed image points; color is that point's reprojection residual "
        "in pixels. All figures share one color range. No polar-angle quantity is plotted.\n",
        encoding="utf-8")
    print(f"Wrote reprojection-error heatmaps: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
