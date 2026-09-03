#!/usr/bin/env python3
"""Export Stage5 multi-board observations to BabelCalib .mat inputs.

The exporter consumes point-level Stage5 CSV outputs plus a fixed multi-board
layout CSV and writes MATLAB structs compatible with BabelCalib:

  corners(i).x      : 2 x N image points
  corners(i).cspond : 2 x N [point_index; board_index], both 1-based
  boards(b).X       : 2 x K board-local planar control points
  boards(b).Rt      : 3 x 4 fixed T_reference_board
  imgsize           : [height, width]

It intentionally keeps frame camera poses out of the exported board geometry.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from scipy.io import savemat


Coord = Tuple[float, float, float]


@dataclass(frozen=True)
class Observation:
    split: str
    frame_index: int
    frame_label: str
    board_id: int
    raw_point_id: int
    point_type: str
    observed_x: float
    observed_y: float
    target: Coord
    source_kind: str


def parse_args() -> argparse.Namespace:
    default_source = (
        Path(__file__).resolve().parents[1]
        / "result_may"
        / "stage5_babelcalib_7030_20260706_1444190clear_ds"
    )
    default_out = (
        Path(__file__).resolve().parents[1]
        / "image"
        / "babelcalib_multiboard_export_1444190clear"
    )
    parser = argparse.ArgumentParser(
        description="Export multi-board Stage5 observations to BabelCalib .mat files."
    )
    parser.add_argument("--source-run", type=Path, default=default_source)
    parser.add_argument("--output-dir", type=Path, default=default_out)
    parser.add_argument("--image-height", type=int, default=4512)
    parser.add_argument("--image-width", type=int, default=4512)
    parser.add_argument("--reference-board-id", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--test-ratio", type=float, default=0.30)
    parser.add_argument(
        "--complete-canonical-grid",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "When possible, export a complete square canonical lattice in boards(b).X "
            "instead of only the observed subset. The lattice dimension is inferred "
            "from raw point IDs, e.g. max point id 120 -> 11x11."
        ),
    )
    parser.add_argument(
        "--use-source-split",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use backend_training_points/backend_holdout_points as train/test when present.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove an existing output directory before exporting.",
    )
    parser.add_argument(
        "--include-metadata-fields",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Include non-BabelCalib metadata fields in MATLAB structs. Disabled "
            "by default so generated .mat files match official BabelCalib data "
            "fields: corners.x/cspond and boards.X/Rt only."
        ),
    )
    parser.add_argument(
        "--point-type-filter",
        choices=("all", "outer", "internal"),
        default="all",
        help="Export all points, only outer board corners, or only internal points.",
    )
    return parser.parse_args()


def read_csv_dicts(path: Path) -> List[dict]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def parse_float(value: str, *, default: float = math.nan) -> float:
    try:
        return float(value)
    except Exception:
        return default


def parse_int(value: str, *, default: int = -1) -> int:
    try:
        return int(float(value))
    except Exception:
        return default


def read_points(path: Path, split: str) -> List[Observation]:
    rows = read_csv_dicts(path)
    observations: List[Observation] = []
    for row in rows:
        frame_label = row.get("frame_label", "").strip()
        if not frame_label:
            continue
        board_id = parse_int(row.get("board_id", ""))
        raw_point_id = parse_int(row.get("point_id", ""))
        x = parse_float(row.get("observed_x", ""))
        y = parse_float(row.get("observed_y", ""))
        target = (
            parse_float(row.get("target_x", "")),
            parse_float(row.get("target_y", "")),
            parse_float(row.get("target_z", "0")),
        )
        if board_id <= 0 or raw_point_id < 0 or not np.isfinite([x, y, *target]).all():
            continue
        observations.append(
            Observation(
                split=split,
                frame_index=parse_int(row.get("frame_index", "")),
                frame_label=frame_label,
                board_id=board_id,
                raw_point_id=raw_point_id,
                point_type=row.get("point_type", ""),
                observed_x=x,
                observed_y=y,
                target=target,
                source_kind=row.get("source_kind", ""),
            )
        )
    return observations


def load_observations(source_run: Path) -> Tuple[List[Observation], List[Observation]]:
    train = read_points(source_run / "backend_training_points.csv", "train")
    test = read_points(source_run / "backend_holdout_points.csv", "test")
    if train or test:
        return train, test

    all_points = read_points(source_run / "benchmark_holdout_points.csv", "all")
    if all_points:
        return all_points, []
    raise FileNotFoundError(
        f"No supported point CSV found in {source_run}. Expected "
        "backend_training_points.csv/backend_holdout_points.csv."
    )


def load_layout(source_run: Path, reference_board_id: int) -> Dict[int, np.ndarray]:
    path = source_run / "backend_board_poses.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Cannot find fixed multi-board layout: {path}. "
            "Please provide a Stage5 output with backend_board_poses.csv or a "
            "5-board T_reference_board layout."
        )

    layout: Dict[int, np.ndarray] = {}
    for row in read_csv_dicts(path):
        board_id = parse_int(row.get("board_id", ""))
        initialized = parse_int(row.get("initialized", "0"))
        if board_id <= 0:
            continue
        if initialized != 1:
            # Stage5 writes placeholder rows for configured-but-never-observed
            # boards. They have no layout to export and must not reject a
            # valid subset whose actually observed boards are initialized.
            observation_count = parse_int(row.get("observation_count", "0"))
            if observation_count == 0:
                continue
            raise RuntimeError(f"Layout board {board_id} is not initialized in {path}")
        vals: List[float] = []
        if "T_reference_board_16" in row:
            vals.append(parse_float(row["T_reference_board_16"]))
            for idx in range(1, 16):
                vals.append(parse_float(row.get(None, [])[idx - 1]) if None in row else math.nan)
        else:
            cols = [k for k in row.keys() if k and k.startswith("T_reference_board")]
            vals = [parse_float(row[k]) for k in cols]
        if len(vals) != 16 or not np.isfinite(vals).all():
            # csv.DictReader places duplicate unnamed columns under key None when
            # the header only names the first matrix entry.
            raw_line_vals = [parse_float(v) for k, v in row.items() if k not in {"board_id", "initialized", "observation_count", "rmse"} for v in (v if isinstance(v, list) else [v])]
            vals = raw_line_vals
        if len(vals) != 16 or not np.isfinite(vals).all():
            raise RuntimeError(f"Could not parse 4x4 T_reference_board for board {board_id}")
        T = np.array(vals, dtype=np.float64).reshape(4, 4)
        layout[board_id] = T

    if reference_board_id not in layout:
        raise RuntimeError(f"Reference board {reference_board_id} missing from layout")
    if not np.allclose(layout[reference_board_id], np.eye(4), atol=1e-8):
        raise RuntimeError(
            f"Reference board {reference_board_id} layout is not identity; "
            "refusing to guess a different reference frame."
        )
    return layout


def coordinate_key(coord: Coord, ndigits: int = 12) -> Coord:
    return tuple(round(float(v), ndigits) for v in coord)  # type: ignore[return-value]


def infer_complete_square_lattice(observations: Sequence[Observation]) -> Tuple[int, float] | None:
    raw_ids = [obs.raw_point_id for obs in observations if obs.raw_point_id >= 0]
    if not raw_ids:
        return None
    point_count = max(raw_ids) + 1
    dim = int(round(math.sqrt(point_count)))
    if dim * dim != point_count or dim <= 1:
        return None

    max_coord = max(max(obs.target[0], obs.target[1]) for obs in observations)
    if max_coord <= 0:
        return None
    pitch = max_coord / float(dim - 1)
    if pitch <= 0:
        return None

    # Check that observed target coordinates are compatible with this lattice.
    for obs in observations:
        u = obs.target[0] / pitch
        v = obs.target[1] / pitch
        if abs(u - round(u)) > 1e-5 or abs(v - round(v)) > 1e-5:
            return None
        if round(u) < 0 or round(u) >= dim or round(v) < 0 or round(v) >= dim:
            return None
    return dim, pitch


def build_point_maps(
    observations: Sequence[Observation],
    *,
    complete_canonical_grid: bool,
) -> Tuple[Dict[int, List[Coord]], Dict[int, Dict[Coord, int]], Dict[int, Dict[int, List[int]]], dict]:
    coords_by_board: Dict[int, set] = defaultdict(set)
    raw_to_babel: Dict[int, Dict[int, set]] = defaultdict(lambda: defaultdict(set))
    for obs in observations:
        key = coordinate_key(obs.target)
        coords_by_board[obs.board_id].add(key)
        raw_to_babel[obs.board_id][obs.raw_point_id].add(key)

    board_points: Dict[int, List[Coord]] = {}
    coord_to_index: Dict[int, Dict[Coord, int]] = {}
    raw_mapping: Dict[int, Dict[int, List[int]]] = defaultdict(dict)
    grid_info = {
        "complete_canonical_grid_enabled": bool(complete_canonical_grid),
        "complete_canonical_grid_used": False,
        "grid_dimension": None,
        "grid_pitch": None,
        "fallback_reason": "",
    }
    lattice = infer_complete_square_lattice(observations) if complete_canonical_grid else None
    if complete_canonical_grid and lattice is None:
        grid_info["fallback_reason"] = "could_not_infer_square_lattice_from_raw_point_ids_and_coordinates"

    for board_id, coords in coords_by_board.items():
        if lattice is not None:
            dim, pitch = lattice
            ordered = [
                coordinate_key((pitch * float(u), pitch * float(v), 0.0))
                for v in range(dim)
                for u in range(dim)
            ]
            grid_info["complete_canonical_grid_used"] = True
            grid_info["grid_dimension"] = dim
            grid_info["grid_pitch"] = pitch
        else:
            ordered = sorted(coords, key=lambda p: (p[1], p[0], p[2]))
        board_points[board_id] = ordered
        coord_to_index[board_id] = {coord: idx + 1 for idx, coord in enumerate(ordered)}
        missing_coords = sorted(coords.difference(coord_to_index[board_id].keys()))
        if missing_coords:
            raise RuntimeError(f"Board {board_id} has observed points outside exported board lattice: {missing_coords[:5]}")
        for raw_id, raw_coords in raw_to_babel[board_id].items():
            raw_mapping[board_id][raw_id] = sorted(coord_to_index[board_id][c] for c in raw_coords)
    return board_points, coord_to_index, raw_mapping, grid_info


def group_by_frame(observations: Iterable[Observation]) -> Dict[str, List[Observation]]:
    frames: Dict[str, List[Observation]] = defaultdict(list)
    for obs in observations:
        frames[obs.frame_label].append(obs)
    return dict(frames)


def sorted_frame_labels(frames: Dict[str, List[Observation]]) -> List[str]:
    def key(label: str) -> Tuple[int, str]:
        values = frames[label]
        idxs = [v.frame_index for v in values if v.frame_index >= 0]
        return (min(idxs) if idxs else 10**9, label)

    return sorted(frames.keys(), key=key)


def make_corners_struct(
    frame_labels: Sequence[str],
    frames: Dict[str, List[Observation]],
    coord_to_index: Dict[int, Dict[Coord, int]],
    board_id_to_matlab: Dict[int, int],
    *,
    include_metadata_fields: bool,
) -> np.ndarray:
    dtype = [("x", "O"), ("cspond", "O")]
    if include_metadata_fields:
        dtype += [("frame_label", "O"), ("frame_index", "O")]
    corners = np.empty((1, len(frame_labels)), dtype=dtype)
    for i, label in enumerate(frame_labels):
        obs_list = sorted(
            frames[label],
            key=lambda o: (board_id_to_matlab[o.board_id], coord_to_index[o.board_id][coordinate_key(o.target)], o.observed_x, o.observed_y),
        )
        x = np.array([[o.observed_x for o in obs_list], [o.observed_y for o in obs_list]], dtype=np.float64)
        cspond = np.array(
            [
                [coord_to_index[o.board_id][coordinate_key(o.target)] for o in obs_list],
                [board_id_to_matlab[o.board_id] for o in obs_list],
            ],
            dtype=np.uint16,
        )
        corners[0, i]["x"] = x
        corners[0, i]["cspond"] = cspond
        if include_metadata_fields:
            corners[0, i]["frame_label"] = np.array(label, dtype=object)
            frame_index = obs_list[0].frame_index if obs_list else -1
            corners[0, i]["frame_index"] = np.array([[frame_index]], dtype=np.int32)
    return corners


def make_boards_struct(
    board_ids: Sequence[int],
    board_points: Dict[int, List[Coord]],
    layout: Dict[int, np.ndarray],
    *,
    include_metadata_fields: bool,
) -> np.ndarray:
    dtype = [("X", "O"), ("Rt", "O")]
    if include_metadata_fields:
        dtype += [("source_board_id", "O")]
    boards = np.empty((1, len(board_ids)), dtype=dtype)
    for i, board_id in enumerate(board_ids):
        points = board_points[board_id]
        # BabelCalib's extract_pt_from_corners expects boards(b).X as 2 x K.
        X = np.array([[p[0] for p in points], [p[1] for p in points]], dtype=np.float64)
        Rt = layout[board_id][:3, :4].astype(np.float64)
        boards[0, i]["X"] = X
        boards[0, i]["Rt"] = Rt
        if include_metadata_fields:
            boards[0, i]["source_board_id"] = np.array([[board_id]], dtype=np.int32)
    return boards


def save_mat(path: Path, corners: np.ndarray, boards: np.ndarray, imgsize: Tuple[int, int]) -> None:
    savemat(
        path,
        {
            "corners": corners,
            "boards": boards,
            "imgsize": np.array([[imgsize[0], imgsize[1]]], dtype=np.uint16),
        },
        do_compression=True,
        long_field_names=True,
    )


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_text_lines(path: Path, lines: Sequence[str]) -> None:
    path.write_text("".join(f"{line}\n" for line in lines))


def split_frames_random(all_labels: Sequence[str], test_ratio: float, seed: int) -> Tuple[List[str], List[str]]:
    labels = list(all_labels)
    rng = random.Random(seed)
    shuffled = labels[:]
    rng.shuffle(shuffled)
    test_count = max(1, int(round(len(shuffled) * test_ratio))) if shuffled else 0
    test = set(shuffled[:test_count])
    train = [label for label in labels if label not in test]
    test_list = [label for label in labels if label in test]
    return train, test_list


def point_count(frames: Dict[str, List[Observation]], labels: Sequence[str]) -> int:
    return sum(len(frames[label]) for label in labels)


def main() -> None:
    args = parse_args()
    source_run = args.source_run.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and args.overwrite:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_obs, test_obs = load_observations(source_run)
    if args.point_type_filter != "all":
        train_obs = [obs for obs in train_obs if obs.point_type == args.point_type_filter]
        test_obs = [obs for obs in test_obs if obs.point_type == args.point_type_filter]
    all_obs = train_obs + test_obs
    if not all_obs:
        raise RuntimeError("No valid observations found")

    layout = load_layout(source_run, args.reference_board_id)
    observed_board_ids = sorted({obs.board_id for obs in all_obs})
    missing_layout = [b for b in observed_board_ids if b not in layout]
    if missing_layout:
        raise RuntimeError(f"Missing T_reference_board layout for boards {missing_layout}")

    board_ids = [args.reference_board_id] + [b for b in observed_board_ids if b != args.reference_board_id]
    board_id_to_matlab = {board_id: i + 1 for i, board_id in enumerate(board_ids)}

    board_points, coord_to_index, raw_mapping, grid_info = build_point_maps(
        all_obs, complete_canonical_grid=args.complete_canonical_grid
    )
    all_frames = group_by_frame(all_obs)
    all_labels = sorted_frame_labels(all_frames)

    if args.use_source_split and train_obs and test_obs:
        train_frames = group_by_frame(train_obs)
        test_frames = group_by_frame(test_obs)
        train_labels = sorted_frame_labels(train_frames)
        test_labels = sorted_frame_labels(test_frames)
        split_mode = "source_backend_training_holdout"
    else:
        train_labels, test_labels = split_frames_random(all_labels, args.test_ratio, args.seed)
        train_frames = {label: all_frames[label] for label in train_labels}
        test_frames = {label: all_frames[label] for label in test_labels}
        split_mode = "random"

    backend_frames = group_by_frame(train_obs if train_obs else [obs for label in train_labels for obs in all_frames[label]])
    backend_labels = sorted_frame_labels(backend_frames)

    boards = make_boards_struct(
        board_ids,
        board_points,
        layout,
        include_metadata_fields=args.include_metadata_fields,
    )
    imgsize = (args.image_height, args.image_width)

    datasets = {
        "all": (all_labels, all_frames),
        "train": (train_labels, train_frames),
        "test": (test_labels, test_frames),
        "backend": (backend_labels, backend_frames),
    }
    frame_counts = {}
    for name, (labels, frames) in datasets.items():
        corners = make_corners_struct(
            labels,
            frames,
            coord_to_index,
            board_id_to_matlab,
            include_metadata_fields=args.include_metadata_fields,
        )
        save_mat(output_dir / f"{name}.mat", corners, boards, imgsize)
        frame_counts[name] = len(labels)

    valid_images = all_labels
    source_failed_images = source_run / "failed_images.txt"
    failed_images = (
        [line.strip() for line in source_failed_images.read_text().splitlines() if line.strip()]
        if source_failed_images.exists()
        else []
    )
    write_text_lines(output_dir / "valid_images.txt", valid_images)
    write_text_lines(output_dir / "failed_images.txt", failed_images)

    split_payload = {
        "mode": split_mode,
        "seed": args.seed,
        "test_ratio": args.test_ratio,
        "all": all_labels,
        "train": train_labels,
        "test": test_labels,
        "backend": backend_labels,
    }
    write_json(output_dir / "split.json", split_payload)

    board_point_counts = {str(board_id): len(board_points[board_id]) for board_id in board_ids}
    board_observation_counts = Counter(obs.board_id for obs in all_obs)
    point_type_counts = Counter(obs.point_type or "unknown" for obs in all_obs)

    raw_point_mapping_json = {
        str(board_id): {str(raw): vals for raw, vals in sorted(raw_mapping[board_id].items())}
        for board_id in board_ids
    }

    layout_json = {
        str(board_id): {
            "matlab_board_index": board_id_to_matlab[board_id],
            "T_reference_board_4x4": layout[board_id].tolist(),
            "Rt_3x4": layout[board_id][:3, :4].tolist(),
        }
        for board_id in board_ids
    }

    conversion_report = {
        "source_run": str(source_run),
        "output_dir": str(output_dir),
        "babelcalib_source_checked": str(Path(__file__).resolve().parents[2] / "babelcalib"),
        "coordinate_convention": "p_camera = T_camera_reference(frame) * T_reference_board(board) * p_board",
        "reference_board": {
            "source_board_id": args.reference_board_id,
            "matlab_board_index": board_id_to_matlab[args.reference_board_id],
            "Rt": "[I | 0]",
        },
        "indexing": {
            "matlab_1_based": True,
            "corners_cspond_row_1": "BabelCalib point index into boards(b).X columns",
            "corners_cspond_row_2": "BabelCalib board index into boards",
            "raw_point_id_policy": "Raw project point IDs are mapped by unique board-local target coordinates to contiguous BabelCalib point indices.",
        },
        "boards_X_shape": "2xK board-local planar coordinates; target_z is checked to be zero and stored in point_mapping diagnostics",
        "canonical_grid": grid_info,
        "point_type_filter": args.point_type_filter,
        "boards_Rt_shape": "3x4 fixed T_reference_board, not per-frame camera pose",
        "matlab_struct_policy": {
            "strict_babelcalib_fields": not args.include_metadata_fields,
            "corners_fields": ["x", "cspond"] if not args.include_metadata_fields else ["x", "cspond", "frame_label", "frame_index"],
            "boards_fields": ["X", "Rt"] if not args.include_metadata_fields else ["X", "Rt", "source_board_id"],
            "corners_x_dtype": "double",
            "corners_cspond_dtype": "uint16",
            "boards_X_dtype": "double",
            "boards_Rt_dtype": "double",
            "imgsize_dtype": "uint16",
            "metadata_location": "JSON reports and text files; not MATLAB structs in strict mode",
        },
        "layout_source": str(source_run / "backend_board_poses.csv"),
        "board_count": len(board_ids),
        "board_id_order": board_ids,
        "board_point_counts": board_point_counts,
        "board_observation_counts": {str(k): v for k, v in sorted(board_observation_counts.items())},
        "raw_point_id_to_babel_point_index": raw_point_mapping_json,
        "layout": layout_json,
        "imgsize": [args.image_height, args.image_width],
        "frame_counts": frame_counts,
        "total_observation_count": len(all_obs),
    }
    write_json(output_dir / "conversion_report.json", conversion_report)

    detection_summary = {
        "source_run": str(source_run),
        "valid_image_count": len(valid_images),
        "failed_image_count": len(failed_images),
        "total_observation_count": len(all_obs),
        "point_type_counts": dict(sorted(point_type_counts.items())),
        "board_observation_counts": {str(k): v for k, v in sorted(board_observation_counts.items())},
        "frame_observation_counts": {label: len(all_frames[label]) for label in all_labels},
        "frame_counts": frame_counts,
    }
    write_json(output_dir / "detection_summary.json", detection_summary)

    readme = f"""# BabelCalib Multi-Board Export

This folder was generated from:

`{source_run}`

## Files

- `all.mat`: all exported valid frames.
- `train.mat`: training split.
- `test.mat`: test split.
- `backend.mat`: Stage5 backend/committed training frames.
- `conversion_report.json`: coordinate, indexing, board-layout, and point-mapping details.
- `detection_summary.json`: observation counts and valid/failed image lists.
- `split.json`: exact frame split.
- `valid_images.txt`: exported frame labels.
- `failed_images.txt`: failed/omitted frame labels.

## BabelCalib Format

- `imgsize = [height, width] = [{args.image_height}, {args.image_width}]`.
- `corners(i).x` is `2 x N`.
- `corners(i).cspond(1,:)` is the 1-based point index into `boards(b).X`.
- `corners(i).cspond(2,:)` is the 1-based board index.
- `boards(b).X` is `2 x K` board-local planar control points.
- `boards(b).Rt` is fixed `T_reference_board` as a `3 x 4` transform.

The reference board is source board `{args.reference_board_id}` and is exported as
`boards(1).Rt = [I | 0]`. Other board transforms come from
`backend_board_poses.csv` and are not per-frame camera poses.

Raw project point IDs are not assumed to be contiguous. They are mapped by each
unique board-local target coordinate to contiguous BabelCalib point indices; the
mapping is recorded in `conversion_report.json`.

Canonical grid export:

```text
complete_canonical_grid_used = {grid_info["complete_canonical_grid_used"]}
grid_dimension = {grid_info["grid_dimension"]}
grid_pitch = {grid_info["grid_pitch"]}
```
"""
    (output_dir / "README.md").write_text(readme)

    # MATLAB helper for users with MATLAB/Octave.
    matlab_check = """function check_babelcalib_multiboard_export(export_dir)
if nargin < 1
    export_dir = fileparts(mfilename('fullpath'));
end
files = {'all.mat','train.mat','test.mat','backend.mat'};
for f = 1:numel(files)
    p = fullfile(export_dir, files{f});
    S = load(p);
    assert(isfield(S, 'corners') && isfield(S, 'boards') && isfield(S, 'imgsize'));
    assert(numel(S.imgsize) == 2);
    for b = 1:numel(S.boards)
        assert(all(size(S.boards(b).Rt) == [3 4]));
        assert(size(S.boards(b).X, 1) == 2);
    end
    assert(norm(S.boards(1).Rt - [eye(3), zeros(3,1)], 'fro') < 1e-9);
    for i = 1:numel(S.corners)
        assert(size(S.corners(i).x, 1) == 2);
        assert(size(S.corners(i).cspond, 1) == 2);
        assert(size(S.corners(i).x, 2) == size(S.corners(i).cspond, 2));
        for j = 1:size(S.corners(i).cspond, 2)
            pid = S.corners(i).cspond(1,j);
            bid = S.corners(i).cspond(2,j);
            assert(bid >= 1 && bid <= numel(S.boards));
            assert(pid >= 1 && pid <= size(S.boards(bid).X, 2));
        end
    end
    fprintf('OK %s: %d frames, %d boards\\n', files{f}, numel(S.corners), numel(S.boards));
end
end
"""
    (output_dir / "check_babelcalib_multiboard_export.m").write_text(matlab_check)

    matlab_minimal = f"""% Minimal BabelCalib smoke test for this multi-board export.
% Run from MATLAB with BabelCalib available locally:
%
%   cd('/Users/linzhaoxian/lzx-ws/project/calibr/babelcalib');
%   init;
%   run('{(output_dir / "run_babelcalib_minimal_multiboard.m").as_posix()}');

export_dir = fileparts(mfilename('fullpath'));
train = load(fullfile(export_dir, 'train.mat'));
test = load(fullfile(export_dir, 'test.mat'));

cfg = calib_cfg('target_model', 'ds', 'target_complexity', 2);
model = calibrate(train.corners, train.boards, train.imgsize, cfg{{:}}, ...
                  'refine', 1, 'debug', 0);
test_model = get_poses(model, test.corners, test.boards, test.imgsize, cfg{{:}}, ...
                       'refine', 1, 'debug', 0);

disp(model);
disp(test_model);
"""
    (output_dir / "run_babelcalib_minimal_multiboard.m").write_text(matlab_minimal)

    print(json.dumps({
        "output_dir": str(output_dir),
        "valid_images": len(valid_images),
        "board_count": len(board_ids),
        "board_point_counts": board_point_counts,
        "total_observation_count": len(all_obs),
        "frame_counts": frame_counts,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
