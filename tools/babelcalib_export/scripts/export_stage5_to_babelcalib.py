#!/usr/bin/env python3
"""Export TartanCalib multi-board Stage5 point CSVs to BabelCalib .mat files.

The exporter intentionally consumes already-produced Stage5 diagnostics instead
of reimplementing the detector.  It converts observed 2D points plus board-local
target coordinates into BabelCalib's `corners`, `boards`, and `imgsize`
variables.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy.io import savemat


@dataclass(frozen=True)
class PointRow:
    split: str
    frame_index: int
    frame_label: str
    board_id: int
    point_id: int
    point_type: str
    observed_x: float
    observed_y: float
    target_x: float
    target_y: float
    target_z: float
    source_file: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export Stage5 multi-board points to BabelCalib all/train/test .mat files."
    )
    parser.add_argument("--stage5-run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=None,
        help="Optional source image directory. When provided, failed_frames.txt lists images without exported observations.",
    )
    parser.add_argument(
        "--all-points-csv",
        type=Path,
        default=None,
        help="Optional single point CSV to treat as the whole dataset before random train/test split.",
    )
    parser.add_argument(
        "--points-source",
        choices=("benchmark", "backend"),
        default="benchmark",
        help="Use benchmark_*_points.csv or backend_*_points.csv.",
    )
    parser.add_argument("--method", default="ours")
    parser.add_argument(
        "--layout-csv",
        type=Path,
        default=None,
        help="CSV with board_id and T_reference_board_16 or T_world_board_16.",
    )
    parser.add_argument(
        "--layout-summary",
        type=Path,
        default=None,
        help="run_multi_board_outer_bootstrap text summary containing boards and 4x4 matrices.",
    )
    parser.add_argument(
        "--allow-identity-layout-for-debug",
        action="store_true",
        help="Write all boards at identity. Only for format debugging; invalid for real multi-board calibration.",
    )
    parser.add_argument("--img-width", type=int, default=None)
    parser.add_argument("--img-height", type=int, default=None)
    parser.add_argument("--split-ratio", type=float, default=0.30)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--use-existing-stage5-split",
        action="store_true",
        help="Use Stage5 training/holdout CSV split as train/test. Default rebuilds split from all frames.",
    )
    parser.add_argument("--min-points-per-image", type=int, default=4)
    parser.add_argument("--max-abs-target-z", type=float, default=1e-9)
    parser.add_argument(
        "--point-types",
        default="outer,internal",
        help="Comma-separated point_type allowlist. Default exports outer and internal.",
    )
    parser.add_argument(
        "--backend-points-csv",
        type=Path,
        default=None,
        help="Optional backend accepted observation CSV. Defaults to backend_training_points.csv when present.",
    )
    parser.add_argument(
        "--allow-backend-fallback",
        choices=("none", "train", "all"),
        default="none",
        help="Fallback for backend.mat when no backend point CSV exists. Default refuses fallback.",
    )
    return parser.parse_args()


def read_key_value_summary(path: Path) -> Dict[str, str]:
    values: Dict[str, str] = {}
    if not path.exists():
        return values
    for line in path.read_text(errors="replace").splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        values[key.strip()] = value.strip()
    return values


def infer_imgsize(run_dir: Path, width: Optional[int], height: Optional[int]) -> Tuple[int, int]:
    if width is not None and height is not None:
        return height, width
    for name in ("stage5_bundle_summary.txt", "backend_training_summary.txt", "benchmark_training_summary.txt"):
        summary = read_key_value_summary(run_dir / name)
        w = summary.get("camera_resolution_width")
        h = summary.get("camera_resolution_height")
        if w and h:
            return int(float(h)), int(float(w))
    if width is None or height is None:
        raise RuntimeError(
            "Could not infer imgsize from summaries. Pass --img-width and --img-height."
        )
    return height, width


def point_csv_paths(run_dir: Path, points_source: str) -> Tuple[Path, Path]:
    prefix = "benchmark" if points_source == "benchmark" else "backend"
    train = run_dir / f"{prefix}_training_points.csv"
    test = run_dir / f"{prefix}_holdout_points.csv"
    missing = [str(p) for p in (train, test) if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing Stage5 point CSV(s): " + ", ".join(missing))
    return train, test


def source_point_csvs(args: argparse.Namespace, run_dir: Path) -> Tuple[Optional[Path], Optional[Path], Optional[Path]]:
    if args.all_points_csv is not None:
        if not args.all_points_csv.exists():
            raise FileNotFoundError(f"Missing --all-points-csv: {args.all_points_csv}")
        return args.all_points_csv, None, None
    train, holdout = point_csv_paths(run_dir, args.points_source)
    return None, train, holdout


def parse_float(row: Dict[str, str], key: str) -> float:
    value = row.get(key, "")
    if value == "":
        return math.nan
    return float(value)


def read_point_rows(
    csv_path: Path,
    method: str,
    point_types: Sequence[str],
    max_abs_target_z: float,
    warnings: List[str],
) -> List[PointRow]:
    rows: List[PointRow] = []
    allowed = set(point_types)
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        required = {
            "method",
            "split",
            "frame_index",
            "frame_label",
            "board_id",
            "point_id",
            "point_type",
            "observed_x",
            "observed_y",
            "target_x",
            "target_y",
            "target_z",
        }
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise RuntimeError(f"{csv_path} missing columns: {sorted(missing)}")
        for raw in reader:
            if raw["method"] != method:
                continue
            if raw["point_type"] not in allowed:
                continue
            target_z = parse_float(raw, "target_z")
            if not math.isfinite(target_z) or abs(target_z) > max_abs_target_z:
                warnings.append(
                    f"skip non-planar point in {csv_path.name}: frame={raw['frame_index']} "
                    f"board={raw['board_id']} point={raw['point_id']} target_z={target_z}"
                )
                continue
            values = [
                parse_float(raw, "observed_x"),
                parse_float(raw, "observed_y"),
                parse_float(raw, "target_x"),
                parse_float(raw, "target_y"),
            ]
            if not all(math.isfinite(v) for v in values):
                warnings.append(
                    f"skip non-finite point in {csv_path.name}: frame={raw['frame_index']} "
                    f"board={raw['board_id']} point={raw['point_id']}"
                )
                continue
            rows.append(
                PointRow(
                    split=raw["split"],
                    frame_index=int(raw["frame_index"]),
                    frame_label=raw["frame_label"],
                    board_id=int(raw["board_id"]),
                    point_id=int(raw["point_id"]),
                    point_type=raw["point_type"],
                    observed_x=values[0],
                    observed_y=values[1],
                    target_x=values[2],
                    target_y=values[3],
                    target_z=target_z,
                    source_file=csv_path.name,
                )
            )
    return rows


def matrix_from_csv_cells(cells: Sequence[str]) -> np.ndarray:
    values: List[float] = []
    for cell in cells:
        if cell == "":
            continue
        parts = re.split(r"[;\s]+", cell.strip())
        for part in parts:
            if part:
                values.append(float(part))
    if len(values) != 16:
        raise ValueError(f"expected 16 matrix values, got {len(values)}")
    return np.array(values, dtype=np.float64).reshape(4, 4)


def parse_layout_csv(path: Path) -> Dict[int, np.ndarray]:
    poses: Dict[int, np.ndarray] = {}
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "board_id" not in reader.fieldnames:
            raise RuntimeError(f"{path} must contain a board_id column")
        matrix_col = None
        for candidate in ("T_reference_board_16", "T_world_board_16", "Rt_16", "T_16"):
            if candidate in reader.fieldnames:
                matrix_col = candidate
                break
        for row in reader:
            board_id = int(row["board_id"])
            if matrix_col is not None:
                matrix_cells = [row[matrix_col]]
                if None in row:
                    matrix_cells.extend(row[None])
                poses[board_id] = matrix_from_csv_cells(matrix_cells)
            else:
                matrix_columns = [c for c in reader.fieldnames if c and c.startswith(("T_", "Rt_"))]
                if len(matrix_columns) >= 16:
                    poses[board_id] = matrix_from_csv_cells([row[c] for c in matrix_columns[:16]])
                else:
                    raise RuntimeError(
                        f"{path} must contain T_reference_board_16, T_world_board_16, Rt_16, "
                        "T_16, or 16 matrix columns."
                    )
    return poses


def parse_layout_summary(path: Path) -> Dict[int, np.ndarray]:
    poses: Dict[int, np.ndarray] = {}
    lines = path.read_text(errors="replace").splitlines()
    for i, line in enumerate(lines):
        match = re.match(r"\s*board\s+(-?\d+):", line)
        if not match:
            continue
        board_id = int(match.group(1))
        matrix_rows: List[List[float]] = []
        for offset in range(1, 8):
            if i + offset >= len(lines):
                break
            nums = re.findall(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?", lines[i + offset])
            if len(nums) == 4:
                matrix_rows.append([float(x) for x in nums])
            if len(matrix_rows) == 4:
                poses[board_id] = np.array(matrix_rows, dtype=np.float64)
                break
    if not poses:
        raise RuntimeError(f"Could not parse any board matrices from {path}")
    return poses


def load_layout(args: argparse.Namespace, run_dir: Path, board_ids: Sequence[int]) -> Tuple[Dict[int, np.ndarray], str]:
    candidates: List[Tuple[Path, str]] = []
    if args.layout_csv:
        candidates.append((args.layout_csv, "layout_csv"))
    default_backend_csv = run_dir / "backend_board_poses.csv"
    if default_backend_csv.exists():
        candidates.append((default_backend_csv, "backend_optimized_scene"))
    default_csv = run_dir / "stage5_intermediate_model" / "intermediate_board_poses.csv"
    if default_csv.exists():
        candidates.append((default_csv, "stage5_intermediate_model"))
    if args.layout_summary:
        return parse_layout_summary(args.layout_summary), f"layout_summary:{args.layout_summary}"
    for path, source in candidates:
        if path.exists():
            return parse_layout_csv(path), f"{source}:{path}"
    if args.allow_identity_layout_for_debug:
        return {board_id: np.eye(4, dtype=np.float64) for board_id in board_ids}, "debug_identity_layout"
    raise RuntimeError(
        "Missing multi-board layout. Provide --layout-csv with T_reference_board_16/T_world_board_16 "
        "or --layout-summary from run_multi_board_outer_bootstrap. Refusing to export identity layouts "
        "unless --allow-identity-layout-for-debug is set."
    )


def validate_layout(poses: Dict[int, np.ndarray], board_ids: Sequence[int]) -> None:
    missing = sorted(set(board_ids).difference(poses))
    if missing:
        raise RuntimeError(f"Layout missing board ids: {missing}")
    for board_id in board_ids:
        matrix = poses[board_id]
        if matrix.shape != (4, 4):
            raise RuntimeError(f"Layout board {board_id} has shape {matrix.shape}, expected 4x4")
        if not np.all(np.isfinite(matrix)):
            raise RuntimeError(f"Layout board {board_id} contains non-finite values")


def build_board_maps(rows: Sequence[PointRow], warnings: List[str]):
    coords: Dict[Tuple[int, int], Tuple[float, float]] = {}
    point_types: Dict[Tuple[int, int], set] = defaultdict(set)
    for row in rows:
        key = (row.board_id, row.point_id)
        xy = (row.target_x, row.target_y)
        if key in coords:
            old = coords[key]
            if abs(old[0] - xy[0]) > 1e-9 or abs(old[1] - xy[1]) > 1e-9:
                warnings.append(
                    f"inconsistent target coordinate for board={row.board_id} point={row.point_id}: "
                    f"{old} vs {xy}; keeping first"
                )
        else:
            coords[key] = xy
        point_types[key].add(row.point_type)
    board_ids = sorted({b for b, _ in coords})
    board_index_by_id = {board_id: i + 1 for i, board_id in enumerate(board_ids)}
    fiducial_index: Dict[Tuple[int, int], int] = {}
    fiducial_metadata: List[Dict[str, object]] = []
    board_points: Dict[int, List[Tuple[int, float, float]]] = {}
    for board_id in board_ids:
        pids = sorted(pid for b, pid in coords if b == board_id)
        board_points[board_id] = []
        for dense_idx, pid in enumerate(pids, start=1):
            fiducial_index[(board_id, pid)] = dense_idx
            x, y = coords[(board_id, pid)]
            board_points[board_id].append((pid, x, y))
            fiducial_metadata.append(
                {
                    "board_id": board_id,
                    "babel_board_index": board_index_by_id[board_id],
                    "original_point_id": pid,
                    "fiducial_index": dense_idx,
                    "target_x": x,
                    "target_y": y,
                    "point_types": ",".join(sorted(point_types[(board_id, pid)])),
                }
            )
    return board_ids, board_index_by_id, fiducial_index, board_points, fiducial_metadata


def matlab_struct_array(records: Sequence[Dict[str, object]], fields: Sequence[str]) -> np.ndarray:
    arr = np.empty((1, len(records)), dtype=[(field, "O") for field in fields])
    for i, record in enumerate(records):
        for field in fields:
            arr[0, i][field] = record.get(field)
    return arr


def make_boards(
    board_ids: Sequence[int],
    board_index_by_id: Dict[int, int],
    board_points: Dict[int, List[Tuple[int, float, float]]],
    layout: Dict[int, np.ndarray],
) -> np.ndarray:
    records: List[Dict[str, object]] = []
    for board_id in board_ids:
        points = board_points[board_id]
        X = np.array([[x for _, x, _ in points], [y for _, _, y in points]], dtype=np.float64)
        Rt = np.array(layout[board_id], dtype=np.float64)[:3, :4]
        records.append(
            {
                "X": X,
                "Rt": Rt,
                "board_id": np.array([[board_id]], dtype=np.int32),
                "babel_board_index": np.array([[board_index_by_id[board_id]]], dtype=np.int32),
                "original_point_ids": np.array([[pid for pid, _, _ in points]], dtype=np.int32),
            }
        )
    return matlab_struct_array(records, ("X", "Rt", "board_id", "babel_board_index", "original_point_ids"))


def group_rows_by_frame(rows: Sequence[PointRow]) -> Dict[Tuple[int, str], List[PointRow]]:
    grouped: Dict[Tuple[int, str], List[PointRow]] = defaultdict(list)
    for row in rows:
        grouped[(row.frame_index, row.frame_label)].append(row)
    return grouped


def make_corners(
    grouped: Dict[Tuple[int, str], List[PointRow]],
    board_index_by_id: Dict[int, int],
    fiducial_index: Dict[Tuple[int, int], int],
    min_points_per_image: int,
    warnings: List[str],
) -> Tuple[np.ndarray, List[Dict[str, object]]]:
    records: List[Dict[str, object]] = []
    frame_metadata: List[Dict[str, object]] = []
    for (frame_index, frame_label), points in sorted(grouped.items()):
        points = sorted(points, key=lambda r: (r.board_id, r.point_id, r.point_type, r.observed_x, r.observed_y))
        if len(points) < min_points_per_image:
            warnings.append(
                f"drop frame={frame_index} label={frame_label}: only {len(points)} points "
                f"(< --min-points-per-image {min_points_per_image})"
            )
            continue
        x = np.array(
            [[p.observed_x for p in points], [p.observed_y for p in points]],
            dtype=np.float64,
        )
        cspond = np.array(
            [
                [fiducial_index[(p.board_id, p.point_id)] for p in points],
                [board_index_by_id[p.board_id] for p in points],
            ],
            dtype=np.float64,
        )
        records.append({"x": x, "cspond": cspond})
        frame_metadata.append(
            {
                "frame_index": frame_index,
                "frame_label": frame_label,
                "point_count": len(points),
                "board_ids": ",".join(str(b) for b in sorted({p.board_id for p in points})),
            }
        )
    return matlab_struct_array(records, ("x", "cspond")), frame_metadata


def write_flat_points_csv(path: Path, rows: Sequence[PointRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "split",
                "frame_index",
                "frame_label",
                "board_id",
                "point_id",
                "point_type",
                "observed_x",
                "observed_y",
                "target_x",
                "target_y",
                "target_z",
                "source_file",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "split": row.split,
                "frame_index": row.frame_index,
                "frame_label": row.frame_label,
                "board_id": row.board_id,
                "point_id": row.point_id,
                "point_type": row.point_type,
                "observed_x": row.observed_x,
                "observed_y": row.observed_y,
                "target_x": row.target_x,
                "target_y": row.target_y,
                "target_z": row.target_z,
                "source_file": row.source_file,
            })


def subset_grouped(
    grouped: Dict[Tuple[int, str], List[PointRow]],
    keys: Iterable[Tuple[int, str]],
) -> Dict[Tuple[int, str], List[PointRow]]:
    return {key: grouped[key] for key in keys if key in grouped}


def split_keys(
    grouped_train: Dict[Tuple[int, str], List[PointRow]],
    grouped_holdout: Dict[Tuple[int, str], List[PointRow]],
    use_existing: bool,
    split_ratio: float,
    seed: int,
) -> Tuple[List[Tuple[int, str]], List[Tuple[int, str]], List[Tuple[int, str]]]:
    all_keys = sorted(set(grouped_train).union(grouped_holdout))
    if use_existing:
        return all_keys, sorted(grouped_train), sorted(grouped_holdout)
    rng = random.Random(seed)
    shuffled = list(all_keys)
    rng.shuffle(shuffled)
    test_count = max(1, int(round(len(shuffled) * split_ratio))) if shuffled else 0
    test_keys = sorted(shuffled[:test_count])
    train_keys = sorted(shuffled[test_count:])
    return all_keys, train_keys, test_keys


def rows_for_keys(rows: Sequence[PointRow], keys: Sequence[Tuple[int, str]]) -> List[PointRow]:
    key_set = set(keys)
    return [row for row in rows if (row.frame_index, row.frame_label) in key_set]


def find_backend_rows(
    args: argparse.Namespace,
    run_dir: Path,
    point_types: Sequence[str],
    warnings: List[str],
) -> Tuple[List[PointRow], str]:
    candidates: List[Path] = []
    if args.backend_points_csv is not None:
        candidates.append(args.backend_points_csv)
    candidates.append(run_dir / "backend_training_points.csv")
    for path in candidates:
        if path.exists():
            rows = read_point_rows(path, args.method, point_types, args.max_abs_target_z, warnings)
            if not rows and path.name.startswith("backend_"):
                rows = read_point_rows(
                    path,
                    "backend_committed_state",
                    point_types,
                    args.max_abs_target_z,
                    warnings,
                )
            return (rows, str(path.resolve()))
    return [], ""


def write_jsonl(path: Path, records: Sequence[Dict[str, object]]) -> None:
    with path.open("w") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def write_mat(
    path: Path,
    corners: np.ndarray,
    boards: np.ndarray,
    imgsize: Tuple[int, int],
    metadata: Dict[str, object],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    savemat(
        path,
        {
            "corners": corners,
            "boards": boards,
            "imgsize": np.array([[imgsize[0], imgsize[1]]], dtype=np.float64),
            "export_metadata_json": json.dumps(metadata, ensure_ascii=False, sort_keys=True),
        },
        do_compression=True,
        long_field_names=True,
    )


def summarize_frame_metadata(frame_metadata: Sequence[Dict[str, object]]) -> Dict[str, object]:
    counts = [int(r["point_count"]) for r in frame_metadata]
    if not counts:
        return {"image_count": 0}
    return {
        "image_count": len(counts),
        "point_count_min": min(counts),
        "point_count_median": float(np.median(counts)),
        "point_count_max": max(counts),
    }


def image_stem_sort_key(path: Path) -> Tuple[int, str]:
    nums = re.findall(r"\d+", path.stem)
    return (int(nums[-1]) if nums else -1, path.stem)


def collect_source_image_labels(image_dir: Optional[Path]) -> List[str]:
    if image_dir is None:
        return []
    if not image_dir.exists():
        raise FileNotFoundError(f"missing --image-dir: {image_dir}")
    suffixes = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    return [p.stem for p in sorted(image_dir.iterdir(), key=image_stem_sort_key) if p.suffix.lower() in suffixes]


def write_failed_frames(
    path: Path,
    source_image_labels: Sequence[str],
    exported_frame_metadata: Sequence[Dict[str, object]],
) -> Dict[str, object]:
    exported_labels = {str(record["frame_label"]) for record in exported_frame_metadata}
    failed = [label for label in source_image_labels if label not in exported_labels]
    with path.open("w") as f:
        for label in failed:
            f.write(label + "\n")
    return {
        "source_image_count": len(source_image_labels),
        "exported_image_count": len(exported_labels),
        "failed_image_count": len(failed),
    }


def main() -> int:
    args = parse_args()
    run_dir = args.stage5_run_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    warnings: List[str] = []
    point_types = [p.strip() for p in args.point_types.split(",") if p.strip()]
    if not point_types:
        raise RuntimeError("--point-types cannot be empty")

    all_csv, train_csv, holdout_csv = source_point_csvs(args, run_dir)
    if all_csv is not None:
        all_rows_from_csv = read_point_rows(
            all_csv, args.method, point_types, args.max_abs_target_z, warnings)
        train_rows = all_rows_from_csv
        holdout_rows: List[PointRow] = []
    else:
        assert train_csv is not None and holdout_csv is not None
        train_rows = read_point_rows(train_csv, args.method, point_types, args.max_abs_target_z, warnings)
        holdout_rows = read_point_rows(holdout_csv, args.method, point_types, args.max_abs_target_z, warnings)
    all_rows = train_rows + holdout_rows
    if not all_rows:
        raise RuntimeError("No rows were exported. Check --method and --point-types.")

    (
        board_ids,
        board_index_by_id,
        fiducial_index,
        board_points,
        fiducial_metadata,
    ) = build_board_maps(all_rows, warnings)
    layout, layout_source = load_layout(args, run_dir, board_ids)
    validate_layout(layout, board_ids)
    boards = make_boards(board_ids, board_index_by_id, board_points, layout)
    imgsize = infer_imgsize(run_dir, args.img_width, args.img_height)

    grouped_train = group_rows_by_frame(train_rows)
    grouped_holdout = group_rows_by_frame(holdout_rows)
    all_keys, train_keys, test_keys = split_keys(
        grouped_train,
        grouped_holdout,
        args.use_existing_stage5_split,
        args.split_ratio,
        args.seed,
    )
    grouped_all = {**grouped_train, **grouped_holdout}

    all_corners, all_frame_metadata = make_corners(
        subset_grouped(grouped_all, all_keys),
        board_index_by_id,
        fiducial_index,
        args.min_points_per_image,
        warnings,
    )
    train_corners, train_frame_metadata = make_corners(
        subset_grouped(grouped_all, train_keys),
        board_index_by_id,
        fiducial_index,
        args.min_points_per_image,
        warnings,
    )
    test_corners, test_frame_metadata = make_corners(
        subset_grouped(grouped_all, test_keys),
        board_index_by_id,
        fiducial_index,
        args.min_points_per_image,
        warnings,
    )
    backend_rows, backend_source = find_backend_rows(args, run_dir, point_types, warnings)
    backend_fallback = ""
    if not backend_rows:
        if args.allow_backend_fallback == "train":
            backend_rows = rows_for_keys(all_rows, train_keys)
            backend_source = "fallback_train"
            backend_fallback = "train"
            warnings.append("backend.mat fallback: using train observations because no backend point CSV was found")
        elif args.allow_backend_fallback == "all":
            backend_rows = list(all_rows)
            backend_source = "fallback_all"
            backend_fallback = "all"
            warnings.append("backend.mat fallback: using all observations because no backend point CSV was found")
        else:
            raise RuntimeError(
                "No backend point CSV found. Expected --backend-points-csv or backend_training_points.csv. "
                "Use --allow-backend-fallback train|all only when this degradation is intentional."
            )
    backend_grouped = group_rows_by_frame(backend_rows)
    backend_keys = sorted(backend_grouped)
    backend_corners, backend_frame_metadata = make_corners(
        backend_grouped,
        board_index_by_id,
        fiducial_index,
        args.min_points_per_image,
        warnings,
    )

    base_metadata = {
        "stage5_run_dir": str(run_dir),
        "all_points_csv": str(all_csv.resolve()) if all_csv is not None else "",
        "training_points_csv": str(train_csv.resolve()) if train_csv is not None else "",
        "holdout_points_csv": str(holdout_csv.resolve()) if holdout_csv is not None else "",
        "points_source": args.points_source,
        "method": args.method,
        "point_types": point_types,
        "layout_source": layout_source,
        "imgsize_height_width": [imgsize[0], imgsize[1]],
        "board_ids": board_ids,
        "board_index_by_id": board_index_by_id,
        "split_ratio": args.split_ratio,
        "seed": args.seed,
        "use_existing_stage5_split": bool(args.use_existing_stage5_split),
        "warning_count": len(warnings),
        "backend_points_csv": backend_source,
        "backend_fallback": backend_fallback,
    }

    write_mat(
        output_dir / "all.mat",
        all_corners,
        boards,
        imgsize,
        {**base_metadata, "split": "all", **summarize_frame_metadata(all_frame_metadata)},
    )
    write_mat(
        output_dir / "train.mat",
        train_corners,
        boards,
        imgsize,
        {**base_metadata, "split": "train", **summarize_frame_metadata(train_frame_metadata)},
    )
    write_mat(
        output_dir / "test.mat",
        test_corners,
        boards,
        imgsize,
        {**base_metadata, "split": "test", **summarize_frame_metadata(test_frame_metadata)},
    )
    write_mat(
        output_dir / "backend.mat",
        backend_corners,
        boards,
        imgsize,
        {**base_metadata, "split": "backend", **summarize_frame_metadata(backend_frame_metadata)},
    )

    write_jsonl(output_dir / "fiducial_map.jsonl", fiducial_metadata)
    write_jsonl(output_dir / "frames_all.jsonl", all_frame_metadata)
    write_jsonl(output_dir / "frames_train.jsonl", train_frame_metadata)
    write_jsonl(output_dir / "frames_test.jsonl", test_frame_metadata)
    write_jsonl(output_dir / "frames_backend.jsonl", backend_frame_metadata)
    write_flat_points_csv(output_dir / "points_all.csv", all_rows)
    write_flat_points_csv(output_dir / "points_train.csv", rows_for_keys(all_rows, train_keys))
    write_flat_points_csv(output_dir / "points_test.csv", rows_for_keys(all_rows, test_keys))
    write_flat_points_csv(output_dir / "points_backend.csv", backend_rows)
    split = {
        "seed": args.seed,
        "split_ratio": args.split_ratio,
        "use_existing_stage5_split": bool(args.use_existing_stage5_split),
        "all": [{"frame_index": k[0], "frame_label": k[1]} for k in all_keys],
        "train": [{"frame_index": k[0], "frame_label": k[1]} for k in train_keys],
        "test": [{"frame_index": k[0], "frame_label": k[1]} for k in test_keys],
        "backend": [{"frame_index": k[0], "frame_label": k[1]} for k in backend_keys],
    }
    (output_dir / "split.json").write_text(json.dumps(split, indent=2, sort_keys=True))
    (output_dir / "warnings.txt").write_text("\n".join(warnings) + ("\n" if warnings else ""))
    failed_summary = write_failed_frames(
        output_dir / "failed_frames.txt",
        collect_source_image_labels(args.image_dir),
        all_frame_metadata,
    )

    summary = {
        **base_metadata,
        "image_dir": str(args.image_dir.resolve()) if args.image_dir is not None else "",
        "all": summarize_frame_metadata(all_frame_metadata),
        "train": summarize_frame_metadata(train_frame_metadata),
        "test": summarize_frame_metadata(test_frame_metadata),
        "backend": summarize_frame_metadata(backend_frame_metadata),
        "failed_frames": failed_summary,
        "fiducial_count_by_board": {str(b): len(board_points[b]) for b in board_ids},
        "outputs": ["all.mat", "train.mat", "test.mat", "backend.mat"],
    }
    (output_dir / "export_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    (output_dir / "conversion_report.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
