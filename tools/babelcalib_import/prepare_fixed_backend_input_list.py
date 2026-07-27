#!/usr/bin/env python3
"""Build a neutral frame-board input list from a BabelCalib MAT dataset."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.io import loadmat

import convert_babelcalib_mat as interchange


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mat", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    source = args.mat.resolve()
    data = loadmat(source, squeeze_me=False, struct_as_record=True)
    corner_items = interchange.struct_items(data["corners"], "corners")
    boards = interchange.load_boards(
        data, interchange.count_correspondences(corner_items)
    )
    board_by_babel_index = {
        index + 1: board for index, board in enumerate(boards)
    }
    metadata = interchange.infer_sidecar(source)
    frames = interchange.load_frame_metadata(metadata, len(corner_items))
    keys: set[tuple[int, int]] = set()
    for view_index, (item, (frame_index, _)) in enumerate(
        zip(corner_items, frames), start=1
    ):
        correspondence = interchange.finite_array(
            interchange.field(item, "cspond"), (2, -1),
            f"corners({view_index}).cspond",
        )
        for babel_board_value in np.unique(correspondence[1]):
            babel_board = int(round(float(babel_board_value)))
            if babel_board not in board_by_babel_index:
                raise RuntimeError(f"invalid board index {babel_board}")
            keys.add((frame_index, board_by_babel_index[babel_board].board_id))
    ordered = sorted(keys)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frame_index", "board_id"])
        writer.writerows(ordered)
    payload = json.dumps(ordered, separators=(",", ":"))
    manifest = {
        "source_mat": str(source),
        "source_frame_metadata": str(metadata.resolve()) if metadata else "",
        "frame_board_count": len(ordered),
        "frame_count": len({frame for frame, _ in ordered}),
        "board_count": len({board for _, board in ordered}),
        "frame_board_set_fingerprint": "sha256:" + hashlib.sha256(
            payload.encode("utf-8")
        ).hexdigest(),
        "selection_source": "all_observed_frame_board_keys_before_residual_ablation",
        "residual_independent": True,
    }
    output.with_suffix(".json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
