#!/usr/bin/env python3
"""Create deterministic nested internal-point density MAT variants."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.io import loadmat, savemat

import convert_babelcalib_mat as interchange


RATIOS = (0.0, 0.25, 0.50, 1.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mat", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=1337)
    return parser.parse_args()


def stable_key(seed: int, identity: tuple[object, ...]) -> str:
    payload = "|".join((str(seed), *(str(value) for value in identity)))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def selected_outer_fiducials(
    board: interchange.Board, babel_board: int, correspondence: np.ndarray
) -> set[int]:
    observed = {
        int(round(float(fiducial)))
        for fiducial, board_value in correspondence.T
        if int(round(float(board_value))) == babel_board
    }
    selected: set[int] = set()
    for corner in range(4):
        candidates = [
            fiducial for fiducial in observed
            if board.outer_index_by_fiducial.get(fiducial) == corner
        ]
        if candidates:
            selected.add(max(
                candidates,
                key=lambda fiducial: (
                    board.observation_count_by_fiducial.get(fiducial, 0),
                    -fiducial,
                ),
            ))
    return selected


def main() -> int:
    args = parse_args()
    source = args.mat.resolve()
    data = loadmat(source, squeeze_me=False, struct_as_record=True)
    corner_items = interchange.struct_items(data["corners"], "corners")
    counts = interchange.count_correspondences(corner_items)
    boards = interchange.load_boards(data, counts)
    board_by_index = {index + 1: board for index, board in enumerate(boards)}
    height, width = (int(value) for value in np.asarray(data["imgsize"]).reshape(-1))
    center = np.asarray([0.5 * (width - 1), 0.5 * (height - 1)])
    radius = min(center[0], center[1], width - 1 - center[0], height - 1 - center[1])

    classifications: list[list[tuple[bool, tuple[int, str], tuple[object, ...]]]] = []
    strata: dict[tuple[int, str], list[tuple[str, int, int]]] = defaultdict(list)
    for view_index, item in enumerate(corner_items):
        pixels = interchange.finite_array(
            interchange.field(item, "x"), (2, -1), f"corners({view_index + 1}).x"
        )
        correspondence = interchange.finite_array(
            interchange.field(item, "cspond"), (2, pixels.shape[1]),
            f"corners({view_index + 1}).cspond",
        )
        outer_by_board = {
            board_index: selected_outer_fiducials(board, board_index, correspondence)
            for board_index, board in board_by_index.items()
        }
        view_classes = []
        for point_index in range(pixels.shape[1]):
            fiducial = int(round(float(correspondence[0, point_index])))
            board_index = int(round(float(correspondence[1, point_index])))
            board_id = board_by_index[board_index].board_id
            is_outer = fiducial in outer_by_board[board_index]
            rho = float(np.linalg.norm(pixels[:, point_index] - center) / radius)
            radial = "center" if rho < 0.4 else "middle" if rho < 0.7 else "peripheral"
            identity = (
                view_index, board_id, fiducial,
                repr(float(pixels[0, point_index])),
                repr(float(pixels[1, point_index])),
            )
            view_classes.append((is_outer, (board_id, radial), identity))
            if not is_outer:
                strata[(board_id, radial)].append(
                    (stable_key(args.seed, identity), view_index, point_index)
                )
        classifications.append(view_classes)

    ranked = {key: sorted(values) for key, values in strata.items()}
    selected_by_ratio: dict[float, set[tuple[int, int]]] = {}
    for ratio in RATIOS:
        selected: set[tuple[int, int]] = set()
        for values in ranked.values():
            count = len(values) if ratio >= 1.0 else int(np.floor(ratio * len(values)))
            selected.update((view, point) for _, view, point in values[:count])
        selected_by_ratio[ratio] = selected
    if not selected_by_ratio[0.25].issubset(selected_by_ratio[0.50]):
        raise RuntimeError("25% internal subset is not nested inside 50%")
    if not selected_by_ratio[0.50].issubset(selected_by_ratio[1.0]):
        raise RuntimeError("50% internal subset is not nested inside 100%")

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    report: dict[str, object] = {
        "source_mat": str(source),
        "seed": args.seed,
        "stratification": "board_id_x_common_image_center_radial_bin",
        "radial_bins": {"center": [0.0, 0.4], "middle": [0.4, 0.7], "peripheral": [0.7, None]},
        "nested": True,
        "variants": {},
    }
    for ratio in RATIOS:
        corners = np.empty(data["corners"].shape, dtype=data["corners"].dtype)
        selected_internal = selected_by_ratio[ratio]
        fingerprint = hashlib.sha256()
        internal_count = 0
        outer_count = 0
        for view_index, item in enumerate(corner_items):
            pixels = np.asarray(interchange.field(item, "x"), dtype=float)
            correspondence = np.asarray(interchange.field(item, "cspond"), dtype=float)
            keep: list[int] = []
            for point_index, (is_outer, _, identity) in enumerate(classifications[view_index]):
                if is_outer:
                    keep.append(point_index)
                    outer_count += 1
                elif (view_index, point_index) in selected_internal:
                    keep.append(point_index)
                    internal_count += 1
                    fingerprint.update(("|".join(str(value) for value in identity) + "\n").encode("utf-8"))
            corners.flat[view_index]["x"] = pixels[:, keep]
            corners.flat[view_index]["cspond"] = correspondence[:, keep]
        label = f"internal_{int(round(100 * ratio)):03d}"
        path = output / f"{label}.mat"
        savemat(path, {
            "corners": corners,
            "boards": data["boards"],
            "imgsize": data["imgsize"],
        }, do_compression=True, oned_as="row")
        report["variants"][label] = {
            "path": str(path),
            "internal_ratio": ratio,
            "outer_observation_count": outer_count,
            "internal_observation_count": internal_count,
            "internal_subset_fingerprint": "sha256:" + fingerprint.hexdigest(),
        }
    (output / "internal_subset_manifest.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
