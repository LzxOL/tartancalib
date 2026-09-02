#!/usr/bin/env python3
"""Export Stage5 point diagnostics to the frozen-observation interchange.

This is intentionally a diagnostics utility.  It preserves the image points
and target topology emitted by a completed Stage5 frontend so they can be
supplied as an initialization-only auxiliary session to another run.
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


BOARD_FIELDS = [
    "board_id", "fiducial_index", "point_id", "target_x", "target_y",
    "target_z", "is_outer", "outer_corner_index",
] + [f"rt{row}{col}" for row in range(3) for col in range(4)]
POINT_FIELDS = [
    "frame_index", "frame_label", "board_id", "fiducial_index", "point_id",
    "point_type", "outer_corner_index", "observed_x", "observed_y",
    "target_x", "target_y", "target_z", "quality",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True,
                        help="Stage5 benchmark_*_points.csv file")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument(
        "--tag-size", type=float, default=0.15625,
        help="outer tag side length in target units when importing bootstrap-view CSV",
    )
    parser.add_argument("--reference-board-id", type=int, default=1)
    parser.add_argument(
        "--board-ids",
        default="",
        help="optional comma-separated board IDs to export",
    )
    parser.add_argument(
        "--expected-board-ids",
        default="",
        help=("comma-separated complete target board IDs; missing boards are "
              "declared with the shared canonical Outer4 geometry but have no observations"),
    )
    parser.add_argument("--split-label", default="auxiliary_initialization")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    requested_board_ids = {
        int(value) for value in args.board_ids.split(",") if value.strip()
    }
    with args.input.open(newline="") as stream:
        source_rows = list(csv.DictReader(stream))

    if source_rows and "corner_index" in source_rows[0]:
        # auto_camera_initialization_bootstrap_views.csv already contains the
        # raw four-corner measurements used by the initializer.  Preserve all
        # `used_in_lm` observations; no later selection/BA filtering leaks in.
        point_ids = (0, 10, 120, 110)
        target_xy = (
            (0.0, 0.0),
            (args.tag_size, 0.0),
            (args.tag_size, args.tag_size),
            (0.0, args.tag_size),
        )
        rows = []
        for row in source_rows:
            if row.get("used_in_lm") != "1":
                continue
            corner_index = int(row["corner_index"])
            if corner_index < 0 or corner_index >= 4:
                raise RuntimeError(f"invalid bootstrap corner index {corner_index}")
            converted = {
                "frame_index": row["frame_index"],
                "frame_label": row["frame_label"],
                "board_id": row["board_id"],
                "point_id": str(point_ids[corner_index]),
                "point_type": "outer",
                "observed_x": row["x"],
                "observed_y": row["y"],
                "target_x": str(target_xy[corner_index][0]),
                "target_y": str(target_xy[corner_index][1]),
                "target_z": "0.0",
                "debug_quality": "1.0",
            }
            rows.append(converted)
    else:
        rows = [
            row for row in source_rows
            if row.get("method") in ("ours", "backend")
            and row.get("evaluation_included", "1") == "1"
            and row.get("point_type") == "outer"
        ]
    if requested_board_ids:
        rows = [row for row in rows if int(row["board_id"]) in requested_board_ids]
    if not rows:
        raise RuntimeError("no included outer points found in input")

    board_points = {}
    board_outer_targets = defaultdict(dict)
    frames = defaultdict(list)
    for row in rows:
        board_id = int(row["board_id"])
        point_id = int(row["point_id"])
        key = (board_id, point_id)
        target = (row["target_x"], row["target_y"], row["target_z"])
        previous = board_points.setdefault(key, (target, row["point_type"]))
        if previous[0] != target:
            raise RuntimeError(f"inconsistent target coordinates for {key}")
        board_outer_targets[board_id][point_id] = tuple(
            float(value) for value in target[:2]
        )
        frames[(int(row["frame_index"]), row["frame_label"])].append(row)

    # Stage5's canonical Outer4 order is TL, TR, BR, BL in target space.
    # Point IDs encode lattice slots and are deliberately not corner indices.
    outer_corner_index = {}
    for board_id, targets in board_outer_targets.items():
        xs = [target[0] for target in targets.values()]
        ys = [target[1] for target in targets.values()]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        expected = {
            (min_x, min_y): 0,
            (max_x, min_y): 1,
            (max_x, max_y): 2,
            (min_x, max_y): 3,
        }
        if len(expected) != 4 or set(targets.values()) != set(expected):
            raise RuntimeError(
                f"board {board_id} does not define exactly four geometric outer corners"
            )
        for point_id, target in targets.items():
            outer_corner_index[(board_id, point_id)] = expected[target]

    expected_board_ids = {
        int(value) for value in args.expected_board_ids.split(",") if value.strip()
    }
    if expected_board_ids:
        observed_board_ids = set(board_outer_targets)
        if not observed_board_ids.issubset(expected_board_ids):
            raise RuntimeError("--expected-board-ids omits an observed board")
        template_board_id = min(observed_board_ids)
        template_targets = board_outer_targets[template_board_id]
        for board_id in sorted(expected_board_ids - observed_board_ids):
            board_outer_targets[board_id] = dict(template_targets)
            for point_id, target in template_targets.items():
                target_strings = tuple(str(value) for value in (*target, 0.0))
                board_points[(board_id, point_id)] = (target_strings, "outer")
                outer_corner_index[(board_id, point_id)] = outer_corner_index[
                    (template_board_id, point_id)
                ]

    args.output.mkdir(parents=True, exist_ok=True)
    identity_rt = {
        f"rt{row}{col}": "1" if row == col else "0"
        for row in range(3) for col in range(4)
    }
    with (args.output / "boards.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=BOARD_FIELDS)
        writer.writeheader()
        for (board_id, point_id), (target, _) in sorted(board_points.items()):
            writer.writerow({
                "board_id": board_id,
                "fiducial_index": point_id,
                "point_id": point_id,
                "target_x": target[0],
                "target_y": target[1],
                "target_z": target[2],
                "is_outer": 1,
                "outer_corner_index": outer_corner_index[(board_id, point_id)],
                **identity_rt,
            })

    point_count = 0
    observation_keys = set()
    with (args.output / "frames.csv").open("w", newline="") as frame_stream, \
            (args.output / "points.csv").open("w", newline="") as point_stream:
        frame_writer = csv.DictWriter(
            frame_stream,
            fieldnames=["frame_index", "frame_label", "split_label", "point_count"],
        )
        point_writer = csv.DictWriter(point_stream, fieldnames=POINT_FIELDS)
        frame_writer.writeheader()
        point_writer.writeheader()
        for (frame_index, frame_label), frame_rows in sorted(frames.items()):
            frame_writer.writerow({
                "frame_index": frame_index,
                "frame_label": frame_label,
                "split_label": args.split_label,
                "point_count": len(frame_rows),
            })
            for row in frame_rows:
                point_id = int(row["point_id"])
                point_writer.writerow({
                    "frame_index": frame_index,
                    "frame_label": frame_label,
                    "board_id": row["board_id"],
                    "fiducial_index": point_id,
                    "point_id": point_id,
                    "point_type": "outer",
                    "outer_corner_index": outer_corner_index[(int(row["board_id"]), point_id)],
                    "observed_x": row["observed_x"],
                    "observed_y": row["observed_y"],
                    "target_x": row["target_x"],
                    "target_y": row["target_y"],
                    "target_z": row["target_z"],
                    "quality": row.get("debug_quality", "1.0") or "1.0",
                })
                observation_keys.add((frame_index, int(row["board_id"])))
                point_count += 1

    metadata = [
        "%YAML:1.0",
        "---",
        'schema_version: "stage5_precomputed_observations_v1"',
        f"image_width: {args.width}",
        f"image_height: {args.height}",
        f"reference_board_id: {args.reference_board_id}",
        f"frame_count: {len(frames)}",
        f"board_count: {len({board_id for board_id, _ in board_points})}",
        f"board_observation_count: {len(observation_keys)}",
        f"point_count: {point_count}",
        'source: "stage5 benchmark diagnostic export; outer points only"',
    ]
    (args.output / "metadata.yaml").write_text("\n".join(metadata) + "\n")
    (args.output / "metadata.json").write_text(json.dumps({
        "schema_version": "stage5_precomputed_observations_v1",
        "image_width": args.width,
        "image_height": args.height,
        "reference_board_id": args.reference_board_id,
        "frame_count": len(frames),
        "board_count": len({board_id for board_id, _ in board_points}),
        "board_observation_count": len(observation_keys),
        "point_count": point_count,
        "allows_partial_outer_observations": False,
        "source": "stage5 benchmark diagnostic export; outer points only",
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
