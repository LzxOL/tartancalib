#!/usr/bin/env python3
"""Render only frame-board outer corners actually used by Stage5 initialization."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


BOARD_COLORS = (
    (44, 160, 255),
    (80, 200, 120),
    (230, 140, 60),
    (190, 90, 220),
    (60, 210, 230),
    (220, 180, 70),
)


@dataclass(frozen=True)
class UsedBoardObservation:
    frame_index: int
    frame_label: str
    board_id: int
    pose_fit_outer_rmse: float
    corners: tuple[tuple[float, float], ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap-csv", required=True, type=Path)
    parser.add_argument("--image-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--mode", choices=("frames", "crops", "both"), default="both"
    )
    parser.add_argument("--crop-padding-ratio", type=float, default=0.25)
    parser.add_argument("--crop-min-padding-px", type=int, default=80)
    return parser.parse_args()


def finite_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"non-finite numeric value: {value}")
    return number


def load_used_observations(csv_path: Path) -> list[UsedBoardObservation]:
    grouped: dict[tuple[int, str, int], list[dict[str, str]]] = defaultdict(list)
    with csv_path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        required = {
            "frame_index",
            "frame_label",
            "board_id",
            "used_in_lm",
            "pose_init_success",
            "pose_fit_outer_rmse",
            "corner_index",
            "x",
            "y",
        }
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"bootstrap CSV is missing columns: {sorted(missing)}")
        for row in reader:
            if row["used_in_lm"] != "1" or row["pose_init_success"] != "1":
                continue
            key = (
                int(row["frame_index"]),
                row["frame_label"],
                int(row["board_id"]),
            )
            grouped[key].append(row)

    observations: list[UsedBoardObservation] = []
    for (frame_index, frame_label, board_id), rows in sorted(grouped.items()):
        corners_by_index: dict[int, tuple[float, float]] = {}
        for row in rows:
            corners_by_index[int(row["corner_index"])] = (
                finite_float(row["x"]),
                finite_float(row["y"]),
            )
        if sorted(corners_by_index) != [0, 1, 2, 3]:
            continue
        observations.append(
            UsedBoardObservation(
                frame_index=frame_index,
                frame_label=frame_label,
                board_id=board_id,
                pose_fit_outer_rmse=finite_float(rows[0]["pose_fit_outer_rmse"]),
                corners=tuple(corners_by_index[index] for index in range(4)),
            )
        )
    return observations


def build_image_index(image_dir: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for path in sorted(image_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}:
            index[path.stem] = path
    return index


def color_for_board(board_id: int) -> tuple[int, int, int]:
    return BOARD_COLORS[(max(1, board_id) - 1) % len(BOARD_COLORS)]


def draw_board(
    image: np.ndarray,
    observation: UsedBoardObservation,
    offset: tuple[int, int] = (0, 0),
) -> None:
    color = color_for_board(observation.board_id)
    points = np.array(
        [
            [int(round(x - offset[0])), int(round(y - offset[1]))]
            for x, y in observation.corners
        ],
        dtype=np.int32,
    )
    overlay = image.copy()
    cv2.fillConvexPoly(overlay, points, color, lineType=cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.12, image, 0.88, 0.0, image)
    cv2.polylines(image, [points], True, color, 5, cv2.LINE_AA)
    for corner_index, point in enumerate(points):
        center = (int(point[0]), int(point[1]))
        cv2.circle(image, center, 10, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(image, center, 7, color, -1, cv2.LINE_AA)
        cv2.putText(
            image,
            str(corner_index),
            (center[0] + 13, center[1] - 13),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            str(corner_index),
            (center[0] + 13, center[1] - 13),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            color,
            1,
            cv2.LINE_AA,
        )


def draw_header(image: np.ndarray, lines: Iterable[str]) -> None:
    text_lines = list(lines)
    width = min(image.shape[1] - 24, 1450)
    height = 28 + 38 * len(text_lines)
    overlay = image.copy()
    cv2.rectangle(overlay, (12, 12), (12 + width, 12 + height), (15, 18, 22), -1)
    cv2.addWeighted(overlay, 0.82, image, 0.18, 0.0, image)
    for index, line in enumerate(text_lines):
        cv2.putText(
            image,
            line,
            (30, 48 + 38 * index),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.82,
            (245, 245, 245),
            2,
            cv2.LINE_AA,
        )


def crop_bounds(
    image: np.ndarray,
    observation: UsedBoardObservation,
    padding_ratio: float,
    min_padding: int,
) -> tuple[int, int, int, int]:
    xs = [point[0] for point in observation.corners]
    ys = [point[1] for point in observation.corners]
    width = max(xs) - min(xs)
    height = max(ys) - min(ys)
    padding = max(min_padding, int(round(max(width, height) * padding_ratio)))
    x0 = max(0, int(math.floor(min(xs))) - padding)
    y0 = max(0, int(math.floor(min(ys))) - padding)
    x1 = min(image.shape[1], int(math.ceil(max(xs))) + padding + 1)
    y1 = min(image.shape[0], int(math.ceil(max(ys))) + padding + 1)
    return x0, y0, x1, y1


def main() -> int:
    args = parse_args()
    observations = load_used_observations(args.bootstrap_csv)
    image_index = build_image_index(args.image_dir)
    frames: dict[tuple[int, str], list[UsedBoardObservation]] = defaultdict(list)
    for observation in observations:
        frames[(observation.frame_index, observation.frame_label)].append(observation)

    frame_dir = args.output_dir / "used_frames"
    crop_dir = args.output_dir / "used_frame_board_crops"
    frame_dir.mkdir(parents=True, exist_ok=True)
    crop_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, object]] = []
    missing_images: set[str] = set()

    for (frame_index, frame_label), frame_observations in sorted(frames.items()):
        image_path = image_index.get(frame_label)
        if image_path is None:
            missing_images.add(frame_label)
            continue
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            missing_images.add(frame_label)
            continue

        if args.mode in {"frames", "both"}:
            frame_overlay = image.copy()
            for observation in frame_observations:
                draw_board(frame_overlay, observation)
            board_list = ",".join(str(item.board_id) for item in frame_observations)
            draw_header(
                frame_overlay,
                (
                    f"Initialization LM frame {frame_index}: {frame_label}",
                    f"used boards: {board_list}  (only pose-init-success observations shown)",
                ),
            )
            frame_output = frame_dir / f"frame_{frame_index:06d}__{frame_label}.png"
            if not cv2.imwrite(str(frame_output), frame_overlay):
                raise RuntimeError(f"failed to write {frame_output}")

        for observation in frame_observations:
            crop_output = ""
            if args.mode in {"crops", "both"}:
                x0, y0, x1, y1 = crop_bounds(
                    image,
                    observation,
                    max(0.0, args.crop_padding_ratio),
                    max(0, args.crop_min_padding_px),
                )
                crop = image[y0:y1, x0:x1].copy()
                draw_board(crop, observation, (x0, y0))
                draw_header(
                    crop,
                    (
                        f"frame {frame_index}  board {observation.board_id}",
                        f"outer pose RMSE: {observation.pose_fit_outer_rmse:.4f} px",
                    ),
                )
                output_path = crop_dir / (
                    f"frame_{frame_index:06d}__board_{observation.board_id:02d}__"
                    f"{frame_label}.png"
                )
                if not cv2.imwrite(str(output_path), crop):
                    raise RuntimeError(f"failed to write {output_path}")
                crop_output = str(output_path)
            manifest_rows.append(
                {
                    "frame_index": observation.frame_index,
                    "frame_label": observation.frame_label,
                    "board_id": observation.board_id,
                    "used_in_lm": 1,
                    "pose_init_success": 1,
                    "pose_fit_outer_rmse": observation.pose_fit_outer_rmse,
                    "corner_count": 4,
                    "crop_path": crop_output,
                }
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "used_frame_board_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(manifest_rows[0]) if manifest_rows else ["frame_index"])
        writer.writeheader()
        writer.writerows(manifest_rows)

    summary_path = args.output_dir / "summary.txt"
    summary_path.write_text(
        "\n".join(
            (
                f"source_csv: {args.bootstrap_csv}",
                f"image_dir: {args.image_dir}",
                f"used_frame_count: {len(frames) - len(missing_images)}",
                f"used_frame_board_count: {len(manifest_rows)}",
                f"rendered_only_used_in_lm_and_pose_init_success: 1",
                f"missing_image_count: {len(missing_images)}",
                *(f"missing_image: {label}" for label in sorted(missing_images)),
                "",
            )
        ),
        encoding="utf-8",
    )
    print(summary_path.read_text(encoding="utf-8"), end="")
    return 0 if not missing_images else 2


if __name__ == "__main__":
    raise SystemExit(main())
