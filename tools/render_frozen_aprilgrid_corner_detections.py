#!/usr/bin/env python3
"""Render only the real AprilGrid corners from frozen frontend observations."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observations", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    groups: dict[tuple[int, str, str], list[dict[str, str]]] = defaultdict(list)
    with args.observations.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row.get("point_type") != "outer":
                continue
            groups[(int(row["frame_index"]), row["frame_label"], row["image_path"])].append(row)

    if not groups:
        raise RuntimeError("No outer-corner observations found.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows: list[tuple[int, str, int, int]] = []
    for frame_index, frame_label, image_path in sorted(groups):
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Failed to read image: {image_path}")
        rows = groups[(frame_index, frame_label, image_path)]
        centers: dict[int, list[tuple[int, int]]] = defaultdict(list)
        for row in rows:
            point = (int(round(float(row["observed_x"]))),
                     int(round(float(row["observed_y"])) ))
            board_id = int(row["board_id"])
            centers[board_id].append(point)
            cv2.circle(image, point, 7, (50, 230, 60), 2, cv2.LINE_AA)
            cv2.circle(image, point, 2, (50, 230, 60), -1, cv2.LINE_AA)
        for board_id, points in centers.items():
            center = tuple(int(round(sum(value[index] for value in points) / len(points)))
                           for index in (0, 1))
            cv2.putText(image, str(board_id), center, cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (20, 220, 255), 2, cv2.LINE_AA)
        text = (f"{frame_label} | real detected corners: {len(rows)} | "
                f"detected tags: {len(centers)}")
        cv2.rectangle(image, (16, 16), (min(image.shape[1] - 16, 1700), 78),
                      (0, 0, 0), -1)
        cv2.putText(image, text, (32, 56), cv2.FONT_HERSHEY_SIMPLEX,
                    0.75, (245, 245, 245), 2, cv2.LINE_AA)
        output = args.output_dir / f"frame_{frame_index:04d}_{frame_label}_detected_outer4.png"
        if not cv2.imwrite(str(output), image):
            raise RuntimeError(f"Failed to write image: {output}")
        summary_rows.append((frame_index, frame_label, len(centers), len(rows)))

    with (args.output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["frame_index", "frame_label", "detected_tag_count", "detected_outer_corner_count"])
        writer.writerows(summary_rows)
    print(f"rendered_frames={len(summary_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
