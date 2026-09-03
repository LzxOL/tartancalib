#!/usr/bin/env python3
"""Compare trial-backend accepted frame-board observations between two runs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


KEY = Tuple[int, int]


def read_decisions(run_dir: Path) -> Dict[KEY, dict]:
    csv_path = run_dir / "trial_backend_frame_board_selection_decisions.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing decisions CSV: {csv_path}")
    rows: Dict[KEY, dict] = {}
    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            key = (int(row["frame_index"]), int(row["board_id"]))
            rows[key] = row
    return rows


def is_incremental_accept(row: dict) -> bool:
    return row.get("reason", "") == "accepted_incremental_trial"


def accepted_rows(rows: Dict[KEY, dict]) -> Dict[KEY, dict]:
    return {key: row for key, row in rows.items() if is_incremental_accept(row)}


def write_rows(path: Path, rows: Iterable[dict]) -> None:
    fieldnames = [
        "frame_index",
        "frame_label",
        "board_id",
        "reason",
        "candidate_score",
        "coverage_gain",
        "mean_polar_angle_deg",
        "max_polar_angle_deg",
        "global_rmse_before",
        "global_rmse_after",
        "global_rmse_delta",
        "outer_rmse_delta",
        "internal_rmse_delta",
        "point_count",
        "outer_point_count",
        "internal_point_count",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted(rows, key=lambda r: (int(r["frame_index"]), int(r["board_id"]))):
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def frame_counts(rows: Dict[KEY, dict]) -> Dict[int, int]:
    counts: Dict[int, int] = {}
    for frame_index, _ in rows:
        counts[frame_index] = counts.get(frame_index, 0) + 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cap1-run", required=True, type=Path)
    parser.add_argument("--cap3-run", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    cap1_all = read_decisions(args.cap1_run)
    cap3_all = read_decisions(args.cap3_run)
    cap1 = accepted_rows(cap1_all)
    cap3 = accepted_rows(cap3_all)

    cap1_keys = set(cap1)
    cap3_keys = set(cap3)
    added_keys = cap3_keys - cap1_keys
    removed_keys = cap1_keys - cap3_keys
    common_keys = cap1_keys & cap3_keys

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_rows(args.output_dir / "cap1_accepted_incremental.csv", cap1.values())
    write_rows(args.output_dir / "cap3_accepted_incremental.csv", cap3.values())
    write_rows(args.output_dir / "cap3_added_vs_cap1.csv", [cap3[key] for key in added_keys])
    write_rows(args.output_dir / "cap3_removed_vs_cap1.csv", [cap1[key] for key in removed_keys])

    cap1_frame_counts = frame_counts(cap1)
    cap3_frame_counts = frame_counts(cap3)
    all_frames = sorted(set(cap1_frame_counts) | set(cap3_frame_counts))
    with (args.output_dir / "per_frame_incremental_accept_count_delta.csv").open(
        "w", newline=""
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["frame_index", "cap1_count", "cap3_count", "delta"])
        for frame_index in all_frames:
            c1 = cap1_frame_counts.get(frame_index, 0)
            c3 = cap3_frame_counts.get(frame_index, 0)
            writer.writerow([frame_index, c1, c3, c3 - c1])

    with (args.output_dir / "cap_compare_summary.txt").open("w") as handle:
        handle.write(f"cap1_run: {args.cap1_run}\n")
        handle.write(f"cap3_run: {args.cap3_run}\n")
        handle.write(f"cap1_accepted_incremental_count: {len(cap1)}\n")
        handle.write(f"cap3_accepted_incremental_count: {len(cap3)}\n")
        handle.write(f"common_accepted_count: {len(common_keys)}\n")
        handle.write(f"cap3_added_vs_cap1_count: {len(added_keys)}\n")
        handle.write(f"cap3_removed_vs_cap1_count: {len(removed_keys)}\n")
        handle.write("outputs:\n")
        handle.write("  cap1_accepted_incremental.csv\n")
        handle.write("  cap3_accepted_incremental.csv\n")
        handle.write("  cap3_added_vs_cap1.csv\n")
        handle.write("  cap3_removed_vs_cap1.csv\n")
        handle.write("  per_frame_incremental_accept_count_delta.csv\n")


if __name__ == "__main__":
    main()
