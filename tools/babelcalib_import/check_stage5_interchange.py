#!/usr/bin/env python3
"""Validate a Stage5 precomputed-observation interchange directory."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    root = args.directory.resolve()
    metadata = json.loads((root / "metadata.json").read_text())
    boards = rows(root / "boards.csv")
    frames = rows(root / "frames.csv")
    points = rows(root / "points.csv")
    if metadata["schema_version"] != "stage5_precomputed_observations_v1":
        raise RuntimeError("unsupported schema_version")
    if len(frames) != int(metadata["frame_count"]):
        raise RuntimeError("frame count mismatch")
    if len(points) != int(metadata["point_count"]):
        raise RuntimeError("point count mismatch")
    outer_definitions: dict[int, set[int]] = {}
    for row in boards:
        if int(row["is_outer"]) != 0:
            outer_definitions.setdefault(int(row["board_id"]), set()).add(
                int(row["outer_corner_index"])
            )
    if any(indices != {0, 1, 2, 3} for indices in outer_definitions.values()):
        raise RuntimeError("every board must define all four geometric outer corners")
    board_observations = {
        (int(row["frame_index"]), int(row["board_id"])) for row in points
    }
    outer_observations = Counter(
        (int(row["frame_index"]), int(row["board_id"]))
        for row in points
        if row["point_type"] == "outer"
    )
    allows_partial_outer = bool(
        metadata.get("allows_partial_outer_observations", False)
    )
    if allows_partial_outer:
        if any(count < 0 or count > 4 for count in outer_observations.values()):
            raise RuntimeError("partial outer observation count must be in [0, 4]")
    elif any(outer_observations.get(key, 0) != 4 for key in board_observations):
        raise RuntimeError("every observed frame-board must contain four outer points")
    board_point_counts = Counter(
        (int(row["frame_index"]), int(row["board_id"])) for row in points
    )
    if any(count < 4 for count in board_point_counts.values()):
        raise RuntimeError("every observed frame-board must contain at least four points")
    frame_point_counts = Counter(int(row["frame_index"]) for row in points)
    for frame in frames:
        if frame_point_counts[int(frame["frame_index"])] != int(frame["point_count"]):
            raise RuntimeError("per-frame point count mismatch")
    print(
        json.dumps(
            {
                "success": True,
                "frames": len(frames),
                "boards": len(outer_definitions),
                "board_observations": len(board_observations),
                "outer_points": sum(outer_observations.values()),
                "internal_points": sum(row["point_type"] == "internal" for row in points),
                "partial_outer_board_observations": sum(
                    outer_observations.get(key, 0) != 4
                    for key in board_observations
                ),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
