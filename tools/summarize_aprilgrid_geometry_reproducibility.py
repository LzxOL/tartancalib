#!/usr/bin/env python3
"""Summarize fixed AprilGrid geometry regions across saved Ours/Kalibr audits."""

import argparse
import csv
import math
from pathlib import Path
from statistics import median


REGIONS = {
    "all_valid": lambda row: True,
    "near_large_outer": lambda row: (
        row["tag_size"] >= 155.5
        and row["radius_p95"] >= 0.533
    ),
    "far_small_outer": lambda row: (
        row["distance"] >= 0.421
        and row["tag_size"] <= 136.8
        and row["radius_p95"] >= 0.511
    ),
}


def finite_float(row, name):
    value = float(row[name])
    return value if math.isfinite(value) else None


def summarize(rows):
    deltas = [row["delta"] for row in rows]
    return {
        "frame_count": len(rows),
        "ours_win_count": sum(delta < 0.0 for delta in deltas),
        "ours_win_rate": sum(delta < 0.0 for delta in deltas) / len(deltas),
        "strong_ours_win_count": sum(delta <= -0.25 for delta in deltas),
        "kalibr_win_count": sum(delta > 0.0 for delta in deltas),
        "mean_ours_minus_kalibr_rmse_px": sum(deltas) / len(deltas),
        "median_ours_minus_kalibr_rmse_px": median(deltas),
        "mean_distance_m": sum(row["distance"] for row in rows) / len(rows),
        "mean_tag_size_px": sum(row["tag_size"] for row in rows) / len(rows),
        "mean_radius_p95_norm": sum(row["radius_p95"] for row in rows) / len(rows),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="append", required=True, type=Path,
                        help="Saved per_frame_geometry.csv file; repeat for each audit.")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--exclude-dataset", action="append", default=["aprilgrid_8_21-7"])
    args = parser.parse_args()

    records = []
    for input_path in args.input:
        run_name = input_path.parent.name
        with input_path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                if row["dataset"] in args.exclude_dataset:
                    continue
                ours = finite_float(row, "ours_rmse_px")
                kalibr = finite_float(row, "kalibr_rmse_px")
                if ours is None or kalibr is None:
                    continue
                records.append({
                    "run": run_name,
                    "dataset": row["dataset"],
                    "delta": ours - kalibr,
                    "distance": finite_float(row, "distance"),
                    "tag_size": finite_float(row, "tag_pixel_size_median"),
                    "radius_p95": finite_float(row, "point_radius_p95_norm"),
                })

    output_rows = []
    for run in sorted({record["run"] for record in records}):
        run_records = [record for record in records if record["run"] == run]
        for region, predicate in REGIONS.items():
            selected = [record for record in run_records if predicate(record)]
            if selected:
                output_rows.append({"run": run, "region": region, **summarize(selected)})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        fieldnames = ["run", "region", "frame_count", "ours_win_count", "ours_win_rate",
                      "strong_ours_win_count", "kalibr_win_count",
                      "mean_ours_minus_kalibr_rmse_px",
                      "median_ours_minus_kalibr_rmse_px", "mean_distance_m",
                      "mean_tag_size_px", "mean_radius_p95_norm"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)


if __name__ == "__main__":
    main()
