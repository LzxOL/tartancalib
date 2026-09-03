#!/usr/bin/env python3
"""Validate BabelCalib .mat files exported from TartanCalib."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from scipy.io import loadmat


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check BabelCalib all/train/test .mat files.")
    parser.add_argument("mat_files", nargs="+", type=Path)
    return parser.parse_args()


def as_struct_items(arr: np.ndarray) -> List[Any]:
    arr = np.asarray(arr)
    if arr.dtype.names is None:
        raise RuntimeError("expected MATLAB struct array")
    return [arr.flat[i] for i in range(arr.size)]


def field(item: Any, name: str) -> Any:
    value = item[name]
    while isinstance(value, np.ndarray) and value.dtype == object and value.size == 1:
        value = value.item()
    return value


def check_file(path: Path) -> Dict[str, Any]:
    data = loadmat(path, squeeze_me=False, struct_as_record=True)
    for name in ("corners", "boards", "imgsize"):
        if name not in data:
            raise RuntimeError(f"{path}: missing variable {name}")
    corners = data["corners"]
    boards = data["boards"]
    imgsize = np.asarray(data["imgsize"]).astype(float).ravel()
    if imgsize.size != 2:
        raise RuntimeError(f"{path}: imgsize must have two entries, got {imgsize}")
    board_items = as_struct_items(boards)
    corner_items = as_struct_items(corners)
    board_point_counts: Dict[int, int] = {}
    for board_idx, item in enumerate(board_items, start=1):
        X = np.asarray(field(item, "X"), dtype=float)
        Rt = np.asarray(field(item, "Rt"), dtype=float)
        if X.ndim != 2 or X.shape[0] != 2:
            raise RuntimeError(f"{path}: boards({board_idx}).X must be 2xN, got {X.shape}")
        if Rt.shape != (3, 4):
            raise RuntimeError(f"{path}: boards({board_idx}).Rt must be 3x4, got {Rt.shape}")
        if not np.all(np.isfinite(X)) or not np.all(np.isfinite(Rt)):
            raise RuntimeError(f"{path}: boards({board_idx}) contains non-finite values")
        board_point_counts[board_idx] = X.shape[1]
    point_counts: List[int] = []
    board_hist: Dict[int, int] = {}
    for image_idx, item in enumerate(corner_items, start=1):
        x = np.asarray(field(item, "x"), dtype=float)
        cspond = np.asarray(field(item, "cspond"), dtype=float)
        if x.ndim != 2 or x.shape[0] != 2:
            raise RuntimeError(f"{path}: corners({image_idx}).x must be 2xN, got {x.shape}")
        if cspond.shape != x.shape:
            raise RuntimeError(
                f"{path}: corners({image_idx}).cspond must be 2xN matching x; got {cspond.shape} vs {x.shape}"
            )
        if not np.all(np.isfinite(x)) or not np.all(np.isfinite(cspond)):
            raise RuntimeError(f"{path}: corners({image_idx}) contains non-finite values")
        point_counts.append(x.shape[1])
        for fid, board in cspond.T:
            board_i = int(board)
            fid_i = int(fid)
            if board_i < 1 or board_i > len(board_items):
                raise RuntimeError(f"{path}: invalid board index {board_i} in image {image_idx}")
            if fid_i < 1 or fid_i > board_point_counts[board_i]:
                raise RuntimeError(
                    f"{path}: invalid fiducial index {fid_i} for board {board_i} in image {image_idx}; "
                    f"board has {board_point_counts[board_i]} points"
                )
            board_hist[board_i] = board_hist.get(board_i, 0) + 1
    summary = {
        "path": str(path),
        "image_count": len(corner_items),
        "board_count": len(board_items),
        "imgsize_height_width": [int(imgsize[0]), int(imgsize[1])],
        "point_count_min": int(min(point_counts)) if point_counts else 0,
        "point_count_median": float(np.median(point_counts)) if point_counts else 0.0,
        "point_count_max": int(max(point_counts)) if point_counts else 0,
        "board_point_counts": board_point_counts,
        "observed_point_count_by_board_index": board_hist,
    }
    if "export_metadata_json" in data:
        raw = data["export_metadata_json"]
        text = "".join(str(x) for x in np.asarray(raw).ravel())
        try:
            summary["metadata"] = json.loads(text)
        except json.JSONDecodeError:
            summary["metadata_parse_error"] = text[:200]
    return summary


def main() -> int:
    args = parse_args()
    summaries = [check_file(path.resolve()) for path in args.mat_files]
    print(json.dumps(summaries, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
