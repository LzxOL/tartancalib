#!/usr/bin/env python3
"""Export frozen Stage5 outer and optional recovered internal points as CSV.

Outer4 corners are read from ``outer_detection_final`` cache records.  When
``--internal-cache-hash`` is supplied, valid recovered internal corners are
read from the matching ``internal_refinement`` records and exported alongside
them.  The resulting bridge directory is consumed by
``export_multiboard_babelcalib.py`` to write MATLAB files.
"""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import cv2


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--cache-hash", required=True)
    parser.add_argument(
        "--internal-cache-hash",
        help=(
            "Optional internal_refinement cache hash. Valid recovered internal "
            "points are added with point_type=internal."
        ),
    )
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--layout-csv", type=Path, required=True)
    parser.add_argument(
        "--holdout-points-csv",
        type=Path,
        help="Optional CSV whose frame labels define an existing frozen holdout split.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tag-size", type=float, required=True)
    parser.add_argument("--module-dimension", type=int, default=10)
    return parser.parse_args()


def read_holdout_labels(path: Path | None) -> set[str]:
    if path is None:
        return set()
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["frame_label"] for row in csv.DictReader(handle) if row["frame_label"]}


def image_corners(cache_file: Path, image_root: Path):
    storage = cv2.FileStorage(str(cache_file), cv2.FILE_STORAGE_READ)
    if not storage.isOpened():
        raise RuntimeError(f"cannot read cache file: {cache_file}")
    image_path = Path(storage.getNode("absolute_image_path").string()).resolve()
    try:
        image_path.relative_to(image_root)
    except ValueError:
        return None

    detections = storage.getNode("detections")
    rows = []
    for index in range(detections.size()):
        detection = detections.at(index)
        if int(detection.getNode("success").real()) != 1:
            continue
        corners = detection.getNode("refined_corners_original_image")
        valid = detection.getNode("refined_valid")
        if corners.size() != 4 or valid.size() != 4:
            continue
        if any(int(valid.at(corner_index).real()) != 1 for corner_index in range(4)):
            continue
        board_id = int(detection.getNode("board_id").real())
        if board_id <= 0:
            continue
        rows.append((board_id, [(corners.at(i).at(0).real(), corners.at(i).at(1).real()) for i in range(4)]))
    return image_path.stem, rows


def internal_corners(cache_file: Path, image_root: Path):
    """Read valid non-outer observations from one frozen internal cache record."""
    storage = cv2.FileStorage(str(cache_file), cv2.FILE_STORAGE_READ)
    if not storage.isOpened():
        raise RuntimeError(f"cannot read cache file: {cache_file}")
    image_path = Path(storage.getNode("absolute_image_path").string()).resolve()
    try:
        image_path.relative_to(image_root)
    except ValueError:
        return None

    frame_result = storage.getNode("frame_result")
    board_measurements = frame_result.getNode("board_measurements")
    rows = []
    for board_index in range(board_measurements.size()):
        measurement = board_measurements.at(board_index)
        corners = measurement.getNode("detection").getNode("corners")
        for corner_index in range(corners.size()):
            corner = corners.at(corner_index)
            if int(corner.getNode("valid").real()) != 1:
                continue
            # corner_type=0 is one of the direct Outer4 detections.  Those are
            # sourced from outer_detection_final so the export has one canonical
            # copy of each outer observation.
            if int(corner.getNode("corner_type").real()) == 0:
                continue
            board_id = int(corner.getNode("board_id").real())
            point_id = int(corner.getNode("point_id").real())
            image_xy = corner.getNode("image_xy")
            target_xyz = corner.getNode("target_xyz")
            if board_id <= 0 or point_id < 0 or image_xy.size() != 2 or target_xyz.size() != 3:
                continue
            rows.append(
                (
                    board_id,
                    point_id,
                    image_xy.at(0).real(),
                    image_xy.at(1).real(),
                    target_xyz.at(0).real(),
                    target_xyz.at(1).real(),
                    target_xyz.at(2).real(),
                    corner.getNode("quality").real(),
                )
            )
    return image_path.stem, rows, frame_result.getNode("state_source_label").string()


def main() -> None:
    args = parse_args()
    image_root = args.image_root.resolve()
    cache_root = args.cache_dir.resolve() / "stage5_cache_layout_v1" / "outer_detection_final" / args.cache_hash
    if not cache_root.is_dir():
        raise FileNotFoundError(f"cache hash directory does not exist: {cache_root}")

    holdout_labels = read_holdout_labels(args.holdout_points_csv)
    cache_files = sorted(cache_root.glob("*.yml"))
    frames = []
    for cache_file in cache_files:
        parsed = image_corners(cache_file, image_root)
        if parsed is not None:
            frames.append(parsed)
    frames.sort(key=lambda item: item[0])
    if len({label for label, _ in frames}) != len(frames):
        raise RuntimeError("duplicate frame labels in selected outer-detection cache")

    internals_by_label = {}
    if args.internal_cache_hash:
        internal_root = (
            args.cache_dir.resolve()
            / "stage5_cache_layout_v1"
            / "internal_refinement"
            / args.internal_cache_hash
        )
        if not internal_root.is_dir():
            raise FileNotFoundError(f"internal cache hash directory does not exist: {internal_root}")
        candidates_by_label = {}
        for cache_file in sorted(internal_root.glob("*.yml")):
            parsed = internal_corners(cache_file, image_root)
            if parsed is None:
                continue
            label, rows, state_source = parsed
            candidates_by_label.setdefault(label, []).append((state_source, rows, cache_file))
        for label, candidates in candidates_by_label.items():
            optimized = [candidate for candidate in candidates if candidate[0] == "optimized_scene"]
            if len(optimized) == 1:
                internals_by_label[label] = optimized[0][1]
            elif len(optimized) > 1:
                files = ", ".join(str(candidate[2]) for candidate in optimized)
                raise RuntimeError(f"multiple optimized internal cache records for {label}: {files}")
            elif len(candidates) == 1:
                internals_by_label[label] = candidates[0][1]
            else:
                files = ", ".join(str(candidate[2]) for candidate in candidates)
                raise RuntimeError(f"ambiguous non-optimized internal cache records for {label}: {files}")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.layout_csv, output_dir / "backend_board_poses.csv")
    header = [
        "method", "split", "frame_index", "frame_label", "board_id", "point_id", "point_type",
        "observed_x", "observed_y", "predicted_x", "predicted_y", "target_x", "target_y", "target_z",
        "residual_x", "residual_y", "residual_norm", "debug_quality", "source_kind", "source_point_index",
    ]
    corner_ids = [0, args.module_dimension, args.module_dimension * (args.module_dimension + 1) + args.module_dimension,
                  args.module_dimension * (args.module_dimension + 1)]
    corner_targets = [(0.0, 0.0), (args.tag_size, 0.0), (args.tag_size, args.tag_size), (0.0, args.tag_size)]
    train_rows, holdout_rows = [], []
    detected_labels = set()
    for frame_index, (label, boards) in enumerate(frames):
        split = "holdout" if label in holdout_labels else "training"
        output_rows = holdout_rows if split == "holdout" else train_rows
        if boards:
            detected_labels.add(label)
        for board_id, corners in boards:
            for point_index, ((x, y), point_id, (target_x, target_y)) in enumerate(
                zip(corners, corner_ids, corner_targets)
            ):
                output_rows.append([
                    "outer_detection_cache", split, frame_index, label, board_id, point_id, "outer",
                    x, y, "", "", target_x, target_y, 0.0, "", "", "", "", "direct_outer_detection_cache", point_index,
                ])
        for board_id, point_id, x, y, target_x, target_y, target_z, quality in internals_by_label.get(label, []):
            output_rows.append([
                "internal_refinement_cache", split, frame_index, label, board_id, point_id, "internal",
                x, y, "", "", target_x, target_y, target_z, "", "", "", quality,
                "stage5_internal_recovery_cache", point_id,
            ])

    for filename, rows in (("backend_training_points.csv", train_rows), ("backend_holdout_points.csv", holdout_rows)):
        with (output_dir / filename).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(header)
            writer.writerows(rows)
    all_labels = {
        path.stem
        for path in image_root.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    }
    cached_labels = {label for label, _ in frames}
    failed_images = sorted(all_labels - cached_labels)
    with (output_dir / "cache_export_summary.txt").open("w", encoding="utf-8") as handle:
        handle.write(f"input_image_count: {len(all_labels)}\n")
        handle.write(f"cached_frame_count: {len(frames)}\n")
        handle.write(f"detected_frame_count: {len(detected_labels)}\n")
        handle.write(f"training_frame_count: {len({row[3] for row in train_rows})}\n")
        handle.write(f"holdout_frame_count: {len({row[3] for row in holdout_rows})}\n")
        handle.write(f"cache_hash: {args.cache_hash}\n")
        if args.internal_cache_hash:
            train_internal_count = sum(row[6] == "internal" for row in train_rows)
            holdout_internal_count = sum(row[6] == "internal" for row in holdout_rows)
            train_outer_count = len(train_rows) - train_internal_count
            holdout_outer_count = len(holdout_rows) - holdout_internal_count
            handle.write(f"internal_cache_hash: {args.internal_cache_hash}\n")
            handle.write(f"internal_cached_frame_count: {len(internals_by_label)}\n")
            handle.write(f"training_outer_point_count: {train_outer_count}\n")
            handle.write(f"holdout_outer_point_count: {holdout_outer_count}\n")
            handle.write(f"training_internal_point_count: {train_internal_count}\n")
            handle.write(f"holdout_internal_point_count: {holdout_internal_count}\n")
            handle.write("point_source: direct_outer_detection_cache_plus_stage5_internal_recovery_cache\n")
        else:
            handle.write("point_source: direct_outer_detection_cache_only\n")
    with (output_dir / "failed_images.txt").open("w", encoding="utf-8") as handle:
        handle.write("\n".join(failed_images))
        if failed_images:
            handle.write("\n")


if __name__ == "__main__":
    main()
