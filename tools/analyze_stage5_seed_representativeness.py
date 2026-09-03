#!/usr/bin/env python3
"""Audit whether a Stage5 persistent seed represents its training frames."""

import argparse
import csv
import math
from collections import Counter, defaultdict
from pathlib import Path


def percentile(values, fraction):
    values = sorted(values)
    if not values:
        return float("nan")
    position = (len(values) - 1) * fraction
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return values[lower]
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def finite(value):
    return value is not None and math.isfinite(value)


def to_float(row, name):
    try:
        return float(row.get(name, ""))
    except ValueError:
        return None


def polygon_area(points):
    if len(points) < 3:
        return None
    return abs(sum(
        points[index][0] * points[(index + 1) % len(points)][1]
        - points[(index + 1) % len(points)][0] * points[index][1]
        for index in range(len(points))) * 0.5)


def quantile_text(values):
    return " / ".join(
        "{:.3f}".format(percentile(values, fraction))
        for fraction in (0.10, 0.50, 0.90))


def summarize(rows):
    frames = {row["frame_index"] for row in rows}
    board_counts = Counter(row["board_id"] for row in rows)
    visible = defaultdict(set)
    for row in rows:
        visible[row["frame_index"]].add(row["board_id"])
    values = {
        "tilt_deg": [row["tilt_deg"] for row in rows if finite(row["tilt_deg"])],
        "normalized_radius": [
            row["normalized_radius"] for row in rows
            if finite(row["normalized_radius"])
        ],
        "projected_area_px2": [
            row["projected_area_px2"] for row in rows
            if finite(row["projected_area_px2"])
        ],
        "pose_rmse_px": [row["pose_rmse_px"] for row in rows if finite(row["pose_rmse_px"])],
        "visible_boards_per_frame": [len(boards) for boards in visible.values()],
    }
    return frames, board_counts, values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", required=True, type=Path)
    parser.add_argument("--pose-samples", required=True, type=Path)
    parser.add_argument("--outer-corners", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--image-width", default=3648.0, type=float)
    parser.add_argument("--image-height", default=2736.0, type=float)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    seed_keys = set()
    all_keys = set()
    with args.selection.open(newline="") as handle:
        for row in csv.DictReader(handle):
            key = (int(row["frame_index"]), int(row["board_id"]))
            all_keys.add(key)
            if row.get("baseline_seed") == "1":
                seed_keys.add(key)

    pose_rows = {}
    half_width = args.image_width * 0.5
    half_height = args.image_height * 0.5
    with args.pose_samples.open(newline="") as handle:
        for row in csv.DictReader(handle):
            key = (int(row["frame_index"]), int(row["board_id"]))
            centroid_x = to_float(row, "centroid_x")
            centroid_y = to_float(row, "centroid_y")
            radius = None
            if finite(centroid_x) and finite(centroid_y):
                radius = math.hypot(
                    (centroid_x - half_width) / half_width,
                    (centroid_y - half_height) / half_height)
            pose_rows[key] = {
                "frame_label": row["frame_label"],
                "pose_rmse_px": to_float(row, "pose_rmse"),
                "tilt_deg": to_float(row, "tilt_deg"),
                "centroid_x": centroid_x,
                "centroid_y": centroid_y,
                "normalized_radius": radius,
            }

    corners = defaultdict(list)
    with args.outer_corners.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("used_in_lm") != "1":
                continue
            key = (int(row["frame_index"]), int(row["board_id"]))
            corners[key].append((to_float(row, "x"), to_float(row, "y")))
    areas = {
        key: polygon_area([point for point in points if finite(point[0]) and finite(point[1])])
        for key, points in corners.items()
    }

    records = []
    for key in sorted(all_keys):
        pose = pose_rows.get(key, {})
        records.append({
            "frame_index": key[0],
            "frame_label": pose.get("frame_label", ""),
            "board_id": key[1],
            "baseline_seed": int(key in seed_keys),
            "pose_rmse_px": pose.get("pose_rmse_px"),
            "tilt_deg": pose.get("tilt_deg"),
            "centroid_x": pose.get("centroid_x"),
            "centroid_y": pose.get("centroid_y"),
            "normalized_radius": pose.get("normalized_radius"),
            "projected_area_px2": areas.get(key),
        })

    output_csv = args.output_dir / "seed_vs_training_observations.csv"
    fieldnames = list(records[0])
    with output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    seed_rows = [row for row in records if row["baseline_seed"]]
    groups = (("seed", seed_rows), ("all_training", records))
    summary_rows = []
    for name, rows in groups:
        frames, board_counts, values = summarize(rows)
        for metric, metric_values in values.items():
            summary_rows.append({
                "group": name,
                "metric": metric,
                "count": len(metric_values),
                "p10": percentile(metric_values, 0.10),
                "median": percentile(metric_values, 0.50),
                "p90": percentile(metric_values, 0.90),
            })
        for board_id, count in sorted(board_counts.items()):
            summary_rows.append({
                "group": name,
                "metric": "board_{}_observations".format(board_id),
                "count": count,
                "p10": count / max(1, len(rows)),
                "median": len(frames),
                "p90": float("nan"),
            })
    summary_csv = args.output_dir / "seed_representativeness_summary.csv"
    with summary_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=("group", "metric", "count", "p10", "median", "p90"))
        writer.writeheader()
        writer.writerows(summary_rows)

    report = args.output_dir / "seed_representativeness_report.md"
    with report.open("w") as handle:
        handle.write("# Seed Representativeness Audit\n\n")
        handle.write("The seed is the existing model-aware persistent seed, not the full Outer4 initializer input. ")
        handle.write("Projected area is an image-scale proxy, not physical distance.\n\n")
        handle.write("| Group | Frames | Board observations |\n|---|---:|---:|\n")
        for name, rows in groups:
            frames, _, _ = summarize(rows)
            handle.write("| {} | {} | {} |\n".format(name, len(frames), len(rows)))
        handle.write("\n| Group | Tilt p10 / median / p90 (deg) | Radius p10 / median / p90 | Area p10 / median / p90 (px^2) | Pose RMSE p10 / median / p90 (px) | Visible boards p10 / median / p90 |\n")
        handle.write("|---|---|---|---|---|---|\n")
        for name, rows in groups:
            _, _, values = summarize(rows)
            handle.write("| {} | {} | {} | {} | {} | {} |\n".format(
                name,
                quantile_text(values["tilt_deg"]),
                quantile_text(values["normalized_radius"]),
                quantile_text(values["projected_area_px2"]),
                quantile_text(values["pose_rmse_px"]),
                quantile_text(values["visible_boards_per_frame"])))
        handle.write("\n| Group | Board observation counts |\n|---|---|\n")
        for name, rows in groups:
            _, board_counts, _ = summarize(rows)
            counts = ", ".join("B{}={}".format(board, count)
                               for board, count in sorted(board_counts.items()))
            handle.write("| {} | {} |\n".format(name, counts))


if __name__ == "__main__":
    main()
