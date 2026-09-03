#!/usr/bin/env python3
"""Validate a BabelCalib multi-board export produced by this project."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
from scipy.io import loadmat


def parse_args() -> argparse.Namespace:
    default_export = (
        Path(__file__).resolve().parents[1]
        / "image"
        / "babelcalib_multiboard_export_1444190clear"
    )
    parser = argparse.ArgumentParser(description="Check exported BabelCalib multi-board .mat files.")
    parser.add_argument("--export-dir", type=Path, default=default_export)
    return parser.parse_args()


def field(obj: Any, name: str) -> Any:
    if hasattr(obj, name):
        return getattr(obj, name)
    if isinstance(obj, np.void) and name in obj.dtype.names:
        return obj[name]
    raise KeyError(name)


def scalar_text(value: Any) -> str:
    arr = np.asarray(value)
    if arr.shape == ():
        return str(arr.item())
    if arr.size == 1:
        return str(arr.reshape(-1)[0])
    return str(value)


def as_struct_list(value: Any) -> list:
    arr = np.asarray(value)
    return [arr.reshape(-1)[i] for i in range(arr.size)]


def check_file(path: Path, report: Dict[str, Any]) -> Dict[str, Any]:
    mat = loadmat(path, squeeze_me=True, struct_as_record=False)
    for key in ("corners", "boards", "imgsize"):
        if key not in mat:
            raise AssertionError(f"{path.name}: missing {key}")

    corners = as_struct_list(mat["corners"])
    boards = as_struct_list(mat["boards"])
    imgsize = np.asarray(mat["imgsize"]).reshape(-1)
    if imgsize.size != 2:
        raise AssertionError(f"{path.name}: imgsize must have two elements")

    if len(boards) != report["board_count"]:
        raise AssertionError(f"{path.name}: board count mismatch")

    board_point_counts = []
    for board_idx, board in enumerate(boards, start=1):
        X = np.asarray(field(board, "X"), dtype=float)
        Rt = np.asarray(field(board, "Rt"), dtype=float)
        if X.ndim == 1:
            X = X.reshape(2, -1)
        if X.shape[0] != 2:
            raise AssertionError(f"{path.name}: boards({board_idx}).X must be 2xK, got {X.shape}")
        if Rt.shape != (3, 4):
            raise AssertionError(f"{path.name}: boards({board_idx}).Rt must be 3x4, got {Rt.shape}")
        board_point_counts.append(X.shape[1])
        if board_idx == 1 and not np.allclose(Rt, np.hstack([np.eye(3), np.zeros((3, 1))]), atol=1e-9):
            raise AssertionError(f"{path.name}: reference boards(1).Rt is not [I|0]")

    total_points = 0
    per_board_obs = {str(i): 0 for i in range(1, len(boards) + 1)}
    for corner_idx, corner in enumerate(corners, start=1):
        x = np.asarray(field(corner, "x"), dtype=float)
        cspond = np.asarray(field(corner, "cspond"), dtype=int)
        if x.size == 0 and cspond.size == 0:
            continue
        if x.ndim == 1:
            x = x.reshape(2, -1)
        if cspond.ndim == 1:
            cspond = cspond.reshape(2, -1)
        if x.shape[0] != 2:
            raise AssertionError(f"{path.name}: corners({corner_idx}).x must be 2xN, got {x.shape}")
        if cspond.shape[0] != 2:
            raise AssertionError(f"{path.name}: corners({corner_idx}).cspond must be 2xN, got {cspond.shape}")
        if x.shape[1] != cspond.shape[1]:
            raise AssertionError(f"{path.name}: corners({corner_idx}) x/cspond point count mismatch")
        total_points += x.shape[1]
        for point_idx, board_idx in zip(cspond[0, :], cspond[1, :]):
            if board_idx < 1 or board_idx > len(boards):
                raise AssertionError(f"{path.name}: illegal board index {board_idx}")
            if point_idx < 1 or point_idx > board_point_counts[board_idx - 1]:
                raise AssertionError(
                    f"{path.name}: illegal point index {point_idx} for board {board_idx}"
                )
            per_board_obs[str(board_idx)] += 1

    return {
        "file": path.name,
        "frame_count": len(corners),
        "board_count": len(boards),
        "imgsize": [int(imgsize[0]), int(imgsize[1])],
        "total_observation_count": total_points,
        "board_point_counts": board_point_counts,
        "per_board_observation_count": per_board_obs,
    }


def main() -> None:
    args = parse_args()
    export_dir = args.export_dir.resolve()
    report_path = export_dir / "conversion_report.json"
    if not report_path.exists():
        raise FileNotFoundError(f"Missing conversion_report.json in {export_dir}")
    report = json.loads(report_path.read_text())
    results = []
    for name in ("all.mat", "train.mat", "test.mat", "backend.mat"):
        path = export_dir / name
        if not path.exists():
            raise FileNotFoundError(path)
        results.append(check_file(path, report))

    summary = {
        "export_dir": str(export_dir),
        "success": True,
        "files": results,
    }
    out_path = export_dir / "check_report.json"
    out_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
