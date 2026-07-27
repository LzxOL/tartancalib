#!/usr/bin/env python3
"""Convert BabelCalib MAT observations into the Stage5 interchange format."""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import loadmat


SCHEMA_VERSION = "stage5_precomputed_observations_v1"


@dataclass(frozen=True)
class Board:
    board_id: int
    points: np.ndarray
    original_point_ids: tuple[int, ...]
    rt: np.ndarray
    outer_index_by_fiducial: dict[int, int]
    observation_count_by_fiducial: dict[int, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert BabelCalib corners/boards/imgsize MAT data for Stage5."
    )
    parser.add_argument("--mat", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--frame-metadata", type=Path)
    parser.add_argument("--split-label", default="training")
    parser.add_argument(
        "--view-indices-file",
        type=Path,
        help="Optional JSON array of zero-based MAT view indices to export.",
    )
    return parser.parse_args()


def struct_items(value: np.ndarray, name: str) -> list[Any]:
    array = np.asarray(value)
    if array.dtype.names is None:
        raise RuntimeError(f"{name} must be a MATLAB struct array")
    return list(array.flat)


def field(item: Any, name: str, *, required: bool = True) -> Any:
    if name not in item.dtype.names:
        if required:
            raise RuntimeError(f"MATLAB struct is missing required field '{name}'")
        return None
    value = item[name]
    while isinstance(value, np.ndarray) and value.dtype == object and value.size == 1:
        value = value.item()
    return value


def finite_array(value: Any, shape_prefix: tuple[int, ...], label: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim != len(shape_prefix) or any(
        expected >= 0 and actual != expected
        for actual, expected in zip(array.shape, shape_prefix)
    ):
        raise RuntimeError(f"{label} has invalid shape {array.shape}, expected {shape_prefix}")
    if not np.all(np.isfinite(array)):
        raise RuntimeError(f"{label} contains NaN or Inf")
    return array


def outer_candidate_indices(
    points: np.ndarray,
    targets: list[tuple[float, float]],
) -> list[list[int]]:
    selected: list[list[int]] = []
    extent = max(float(np.ptp(points[0])), float(np.ptp(points[1])), 1.0)
    tolerance = max(1e-10, extent * 1e-7)
    for target_x, target_y in targets:
        squared = (points[0] - target_x) ** 2 + (points[1] - target_y) ** 2
        candidates = [
            index
            for index, distance_squared in enumerate(squared)
            if math.sqrt(float(distance_squared)) <= tolerance
        ]
        if not candidates:
            raise RuntimeError(
                f"board has no control point at outer target ({target_x}, {target_y})"
            )
        selected.append(candidates)
    return selected


def load_boards(
    data: dict[str, Any], observation_counts: dict[tuple[int, int], int]
) -> list[Board]:
    boards: list[Board] = []
    seen_ids: set[int] = set()
    for fallback_id, item in enumerate(struct_items(data["boards"], "boards"), start=1):
        points = finite_array(field(item, "X"), (2, -1), f"boards({fallback_id}).X")
        if points.shape[1] < 4:
            raise RuntimeError(f"board {fallback_id} has fewer than four points")
        rt = finite_array(field(item, "Rt"), (3, 4), f"boards({fallback_id}).Rt")
        board_id_value = field(item, "board_id", required=False)
        board_id = (
            int(np.asarray(board_id_value).reshape(-1)[0])
            if board_id_value is not None
            else fallback_id
        )
        if board_id <= 0 or board_id in seen_ids:
            raise RuntimeError(f"invalid or duplicate board id {board_id}")
        seen_ids.add(board_id)
        ids_value = field(item, "original_point_ids", required=False)
        if ids_value is None:
            original_ids = tuple(range(points.shape[1]))
        else:
            original_ids = tuple(int(v) for v in np.asarray(ids_value).reshape(-1))
            if len(original_ids) != points.shape[1]:
                raise RuntimeError(
                    f"board {board_id} original_point_ids has {len(original_ids)} entries, "
                    f"expected {points.shape[1]}"
                )
        board_observation_counts = {
            fiducial: count
            for (babel_board, fiducial), count in observation_counts.items()
            if babel_board == fallback_id
        }
        observed_indices = [
            index
            for index in range(points.shape[1])
            if board_observation_counts.get(index + 1, 0) > 0
        ]
        if len(observed_indices) < 4:
            raise RuntimeError(
                f"board {board_id} has fewer than four globally observed control points"
            )
        observed_points = points[:, observed_indices]
        min_x, max_x = (
            float(np.min(observed_points[0])),
            float(np.max(observed_points[0])),
        )
        min_y, max_y = (
            float(np.min(observed_points[1])),
            float(np.max(observed_points[1])),
        )
        outer_candidates = outer_candidate_indices(
            points,
            [(min_x, min_y), (max_x, min_y), (max_x, max_y), (min_x, max_y)],
        )
        outer_index_by_fiducial = {
            index + 1: corner
            for corner, candidates in enumerate(outer_candidates)
            for index in candidates
        }
        for corner, candidates in enumerate(outer_candidates):
            if not any(board_observation_counts.get(index + 1, 0) > 0 for index in candidates):
                raise RuntimeError(
                    f"board {board_id} outer corner {corner} has no observed fiducial"
                )
        boards.append(
            Board(
                board_id=board_id,
                points=points,
                original_point_ids=original_ids,
                rt=rt,
                outer_index_by_fiducial=outer_index_by_fiducial,
                observation_count_by_fiducial=board_observation_counts,
            )
        )
    if not boards:
        raise RuntimeError("MAT file contains no boards")
    return boards


def load_frame_metadata(path: Path | None, count: int) -> list[tuple[int, str]]:
    if path is None or not path.is_file():
        return [(index, f"mat_frame_{index:06d}") for index in range(count)]
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if len(records) != count:
        raise RuntimeError(
            f"frame metadata has {len(records)} records but MAT contains {count} views"
        )
    result: list[tuple[int, str]] = []
    seen_indices: set[int] = set()
    seen_labels: set[str] = set()
    for fallback, record in enumerate(records):
        frame_index = int(record.get("frame_index", fallback))
        frame_label = str(record.get("frame_label", f"mat_frame_{fallback:06d}"))
        if frame_index in seen_indices or frame_label in seen_labels or not frame_label:
            raise RuntimeError("frame metadata contains duplicate/empty frame identity")
        seen_indices.add(frame_index)
        seen_labels.add(frame_label)
        result.append((frame_index, frame_label))
    return result


def infer_sidecar(mat_path: Path) -> Path | None:
    candidate = mat_path.with_name(f"frames_{mat_path.stem}.jsonl")
    return candidate if candidate.is_file() else None


def load_view_indices(path: Path | None, view_count: int) -> list[int]:
    if path is None:
        return list(range(view_count))
    values = json.loads(path.read_text())
    if not isinstance(values, list) or not values:
        raise RuntimeError("view-indices file must contain a non-empty JSON array")
    indices = [int(value) for value in values]
    if len(set(indices)) != len(indices):
        raise RuntimeError("view-indices file contains duplicate indices")
    if min(indices) < 0 or max(indices) >= view_count:
        raise RuntimeError(
            f"view-indices must be in [0, {view_count - 1}]"
        )
    return indices


def count_correspondences(corner_items: list[Any]) -> dict[tuple[int, int], int]:
    counts: dict[tuple[int, int], int] = {}
    for view_index, item in enumerate(corner_items, start=1):
        correspondence = finite_array(
            field(item, "cspond"), (2, -1), f"corners({view_index}).cspond"
        )
        for fiducial_value, board_value in correspondence.T:
            fiducial = int(round(float(fiducial_value)))
            board = int(round(float(board_value)))
            if fiducial <= 0 or board <= 0:
                raise RuntimeError(
                    f"corners({view_index}).cspond contains non-positive indices"
                )
            counts[(board, fiducial)] = counts.get((board, fiducial), 0) + 1
    return counts


def write_interchange(args: argparse.Namespace) -> None:
    mat_path = args.mat.resolve()
    data = loadmat(mat_path, squeeze_me=False, struct_as_record=True)
    for name in ("corners", "boards", "imgsize"):
        if name not in data:
            raise RuntimeError(f"{mat_path} is missing required variable '{name}'")
    imgsize = np.asarray(data["imgsize"], dtype=float).reshape(-1)
    if imgsize.size != 2 or not np.all(np.isfinite(imgsize)):
        raise RuntimeError("imgsize must contain finite [height, width]")
    image_height, image_width = (int(round(float(v))) for v in imgsize)
    if image_height <= 0 or image_width <= 0:
        raise RuntimeError(f"invalid imgsize {imgsize.tolist()}")

    all_corner_items = struct_items(data["corners"], "corners")
    selected_view_indices = load_view_indices(
        args.view_indices_file, len(all_corner_items)
    )
    corner_items = [all_corner_items[index] for index in selected_view_indices]
    boards = load_boards(data, count_correspondences(corner_items))
    board_by_babel_index = {index + 1: board for index, board in enumerate(boards)}
    metadata_path = args.frame_metadata or infer_sidecar(mat_path)
    all_frames = load_frame_metadata(metadata_path, len(all_corner_items))
    frames = [all_frames[index] for index in selected_view_indices]

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / "boards.csv").open("w", newline="") as stream:
        names = [
            "board_id", "fiducial_index", "point_id", "target_x", "target_y", "target_z",
            "is_outer", "outer_corner_index",
        ] + [f"rt{r}{c}" for r in range(3) for c in range(4)]
        writer = csv.DictWriter(stream, fieldnames=names)
        writer.writeheader()
        for board in boards:
            for point_index in range(board.points.shape[1]):
                fiducial = point_index + 1
                outer_corner = board.outer_index_by_fiducial.get(fiducial, -1)
                row = {
                    "board_id": board.board_id,
                    "fiducial_index": fiducial,
                    "point_id": board.original_point_ids[point_index],
                    "target_x": repr(float(board.points[0, point_index])),
                    "target_y": repr(float(board.points[1, point_index])),
                    "target_z": "0.0",
                    "is_outer": int(outer_corner >= 0),
                    "outer_corner_index": outer_corner,
                }
                row.update(
                    {f"rt{r}{c}": repr(float(board.rt[r, c])) for r in range(3) for c in range(4)}
                )
                writer.writerow(row)

    point_count = 0
    observed_board_keys: set[tuple[int, int]] = set()
    partial_outer_board_observation_count = 0
    with (output / "frames.csv").open("w", newline="") as frame_stream, (
        output / "points.csv"
    ).open("w", newline="") as point_stream:
        frame_writer = csv.DictWriter(
            frame_stream, fieldnames=["frame_index", "frame_label", "split_label", "point_count"]
        )
        point_writer = csv.DictWriter(
            point_stream,
            fieldnames=[
                "frame_index", "frame_label", "board_id", "fiducial_index", "point_id",
                "point_type", "outer_corner_index", "observed_x", "observed_y",
                "target_x", "target_y", "target_z", "quality",
            ],
        )
        frame_writer.writeheader()
        point_writer.writeheader()
        for view_index, (item, (frame_index, frame_label)) in enumerate(zip(corner_items, frames), start=1):
            pixels = finite_array(field(item, "x"), (2, -1), f"corners({view_index}).x")
            correspondence = finite_array(
                field(item, "cspond"), (2, pixels.shape[1]), f"corners({view_index}).cspond"
            )
            selected_outer: dict[tuple[int, int], int] = {}
            for babel_board, board in board_by_babel_index.items():
                observed_fiducials = {
                    int(round(float(fiducial_value)))
                    for fiducial_value, board_value in correspondence.T
                    if int(round(float(board_value))) == babel_board
                }
                if not observed_fiducials:
                    continue
                selected_corner_count = 0
                for corner in range(4):
                    candidates = [
                        fiducial
                        for fiducial in observed_fiducials
                        if board.outer_index_by_fiducial.get(fiducial) == corner
                    ]
                    if not candidates:
                        continue
                    selected = max(
                        candidates,
                        key=lambda fiducial: (
                            board.observation_count_by_fiducial.get(fiducial, 0),
                            -fiducial,
                        ),
                    )
                    selected_outer[(babel_board, selected)] = corner
                    selected_corner_count += 1
                if selected_corner_count < 4:
                    partial_outer_board_observation_count += 1
            frame_writer.writerow(
                {
                    "frame_index": frame_index,
                    "frame_label": frame_label,
                    "split_label": args.split_label,
                    "point_count": pixels.shape[1],
                }
            )
            for observation_index in range(pixels.shape[1]):
                fiducial = int(round(float(correspondence[0, observation_index])))
                babel_board = int(round(float(correspondence[1, observation_index])))
                if babel_board not in board_by_babel_index:
                    raise RuntimeError(
                        f"view {view_index} observation {observation_index}: invalid board index {babel_board}"
                    )
                board = board_by_babel_index[babel_board]
                if fiducial <= 0 or fiducial > board.points.shape[1]:
                    raise RuntimeError(
                        f"view {view_index}: invalid fiducial {fiducial} for board {board.board_id}"
                    )
                point_index = fiducial - 1
                outer_corner = selected_outer.get((babel_board, fiducial), -1)
                point_writer.writerow(
                    {
                        "frame_index": frame_index,
                        "frame_label": frame_label,
                        "board_id": board.board_id,
                        "fiducial_index": fiducial,
                        "point_id": board.original_point_ids[point_index],
                        "point_type": "outer" if outer_corner >= 0 else "internal",
                        "outer_corner_index": outer_corner,
                        "observed_x": repr(float(pixels[0, observation_index])),
                        "observed_y": repr(float(pixels[1, observation_index])),
                        "target_x": repr(float(board.points[0, point_index])),
                        "target_y": repr(float(board.points[1, point_index])),
                        "target_z": "0.0",
                        "quality": "1.0",
                    }
                )
                observed_board_keys.add((frame_index, board.board_id))
                point_count += 1

    metadata = {
        "schema_version": SCHEMA_VERSION,
        "source_mat": str(mat_path),
        "source_frame_metadata": str(metadata_path.resolve()) if metadata_path else "",
        "split_label": args.split_label,
        "image_width": image_width,
        "image_height": image_height,
        "reference_board_id": boards[0].board_id,
        "frame_count": len(corner_items),
        "source_view_count": len(all_corner_items),
        "source_view_indices": selected_view_indices,
        "board_count": len(boards),
        "board_observation_count": len(observed_board_keys),
        "point_count": point_count,
        "outer_point_definition": "four_globally_observable_support_bbox_corners",
        "allows_partial_outer_observations":
            partial_outer_board_observation_count > 0,
        "partial_outer_board_observation_count":
            partial_outer_board_observation_count,
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    with (output / "metadata.yaml").open("w") as stream:
        stream.write("%YAML:1.0\n---\n")
        for key, value in metadata.items():
            rendered = json.dumps(value) if isinstance(value, str) else str(value)
            stream.write(f"{key}: {rendered}\n")
    print(json.dumps(metadata, indent=2, sort_keys=True))


def main() -> int:
    args = parse_args()
    write_interchange(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
