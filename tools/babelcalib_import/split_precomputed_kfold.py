#!/usr/bin/env python3
"""Create geometry-stratified K-fold Stage5 precomputed observation splits."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def quantile_bins(values: dict[int, float], count: int) -> dict[int, int]:
    ordered = sorted(values, key=lambda key: (values[key], key))
    size = max(1, len(ordered))
    return {
        key: min(count - 1, (rank * count) // size)
        for rank, key in enumerate(ordered)
    }


def frame_features(
    points_by_frame: dict[int, list[dict[str, str]]],
    image_width: float,
    image_height: float,
) -> dict[int, tuple[float, float, float, float]]:
    features: dict[int, tuple[float, float, float, float]] = {}
    for frame_index, rows in points_by_frame.items():
        image = np.array(
            [[float(row["observed_x"]), float(row["observed_y"])] for row in rows],
            dtype=float,
        )
        target = np.array(
            [[float(row["target_x"]), float(row["target_y"]), 1.0] for row in rows],
            dtype=float,
        )
        center = image.mean(axis=0)
        span = np.maximum(0.0, image.max(axis=0) - image.min(axis=0))
        area_ratio = float(span[0] * span[1] / max(1.0, image_width * image_height))
        affine, _, _, _ = np.linalg.lstsq(target, image, rcond=None)
        singular_values = np.linalg.svd(affine[:2, :].T, compute_uv=False)
        tilt_ratio = float(
            singular_values[0] / max(1e-12, singular_values[-1])
        )
        features[frame_index] = (
            float(center[0] / max(1.0, image_width)),
            float(center[1] / max(1.0, image_height)),
            area_ratio,
            math.log(max(1.0, tilt_ratio)),
        )
    return features


def assign_folds(
    features: dict[int, tuple[float, float, float, float]],
    fold_count: int,
    seed: int,
) -> list[list[int]]:
    area_bins = quantile_bins({key: value[2] for key, value in features.items()}, 3)
    tilt_bins = quantile_bins({key: value[3] for key, value in features.items()}, 3)
    strata: dict[tuple[int, int, int, int], list[int]] = defaultdict(list)
    for frame_index, (center_x, center_y, _, _) in features.items():
        stratum = (
            min(1, max(0, int(center_x * 2.0))),
            min(1, max(0, int(center_y * 2.0))),
            area_bins[frame_index],
            tilt_bins[frame_index],
        )
        strata[stratum].append(frame_index)

    rng = random.Random(seed)
    folds: list[list[int]] = [[] for _ in range(fold_count)]
    for stratum in sorted(strata):
        members = strata[stratum]
        rng.shuffle(members)
        for frame_index in members:
            target_fold = min(range(fold_count), key=lambda index: (len(folds[index]), index))
            folds[target_fold].append(frame_index)
    for fold in folds:
        fold.sort()
    return folds


def write_metadata_yaml(path: Path, metadata: dict[str, object]) -> None:
    indices = ", ".join(str(value) for value in metadata["source_view_indices"])
    lines = [
        "%YAML:1.0",
        "---",
        f'schema_version: "{metadata["schema_version"]}"',
        f'source_mat: "{metadata.get("source_mat", "")}"',
        f'source_frame_metadata: "{metadata.get("source_frame_metadata", "")}"',
        f'split_label: "{metadata["split_label"]}"',
        f'image_width: {metadata["image_width"]}',
        f'image_height: {metadata["image_height"]}',
        f'reference_board_id: {metadata["reference_board_id"]}',
        f'frame_count: {metadata["frame_count"]}',
        f'source_view_count: {metadata["source_view_count"]}',
        f"source_view_indices: [{indices}]",
        f'board_count: {metadata["board_count"]}',
        f'board_observation_count: {metadata["board_observation_count"]}',
        f'point_count: {metadata["point_count"]}',
        f'outer_point_definition: "{metadata.get("outer_point_definition", "")}"',
        "allows_partial_outer_observations: "
        + ("True" if metadata.get("allows_partial_outer_observations", False) else "False"),
        "partial_outer_board_observation_count: "
        + str(metadata.get("partial_outer_board_observation_count", 0)),
    ]
    path.write_text("\n".join(lines) + "\n")


def write_split(
    source: Path,
    destination: Path,
    frame_indices: set[int],
    label: str,
    frames: list[dict[str, str]],
    points: list[dict[str, str]],
    metadata: dict[str, object],
) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    selected_points = [
        row for row in points if int(row["frame_index"]) in frame_indices
    ]
    point_count_by_frame: dict[int, int] = defaultdict(int)
    board_keys: set[tuple[int, int]] = set()
    for row in selected_points:
        frame_index = int(row["frame_index"])
        point_count_by_frame[frame_index] += 1
        board_keys.add((frame_index, int(row["board_id"])))
    selected_frames = []
    for row in frames:
        frame_index = int(row["frame_index"])
        if frame_index not in frame_indices:
            continue
        updated = dict(row)
        updated["split_label"] = label
        updated["point_count"] = str(point_count_by_frame[frame_index])
        selected_frames.append(updated)

    shutil.copy2(source / "boards.csv", destination / "boards.csv")
    write_csv(destination / "frames.csv", selected_frames, list(frames[0]))
    write_csv(destination / "points.csv", selected_points, list(points[0]))

    split_metadata = dict(metadata)
    split_metadata.update(
        {
            "split_label": label,
            "frame_count": len(selected_frames),
            "source_view_count": len(selected_frames),
            "source_view_indices": sorted(frame_indices),
            "board_observation_count": len(board_keys),
            "point_count": len(selected_points),
        }
    )
    (destination / "metadata.json").write_text(
        json.dumps(split_metadata, indent=2, sort_keys=True) + "\n"
    )
    write_metadata_yaml(destination / "metadata.yaml", split_metadata)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()
    if args.folds < 2:
        raise ValueError("--folds must be at least 2")

    frames = read_csv(args.input / "frames.csv")
    points = read_csv(args.input / "points.csv")
    metadata = json.loads((args.input / "metadata.json").read_text())
    points_by_frame: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in points:
        points_by_frame[int(row["frame_index"])].append(row)
    features = frame_features(
        points_by_frame,
        float(metadata["image_width"]),
        float(metadata["image_height"]),
    )
    folds = assign_folds(features, args.folds, args.seed)
    all_frames = set(features)
    for fold_index, validation_frames in enumerate(folds):
        fold_root = args.output / f"fold_{fold_index}"
        validation_set = set(validation_frames)
        write_split(
            args.input,
            fold_root / "training",
            all_frames - validation_set,
            "training",
            frames,
            points,
            metadata,
        )
        write_split(
            args.input,
            fold_root / "holdout",
            validation_set,
            "holdout",
            frames,
            points,
            metadata,
        )

    manifest = {
        "schema_version": "stage5_geometry_stratified_kfold_v1",
        "input": str(args.input.resolve()),
        "fold_count": args.folds,
        "seed": args.seed,
        "stratification": ["center_quadrant", "area_tertile", "tilt_tertile"],
        "folds": [
            {"fold": index, "validation_frame_indices": fold}
            for index, fold in enumerate(folds)
        ],
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
