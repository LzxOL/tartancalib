#!/usr/bin/env python3
"""Export detection-only crops for high-internal-residual board observations.

This utility reads completed Stage5 CSV artifacts. It never reruns calibration
or changes detections: it only draws the recorded observed pixel locations.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--threshold-px", type=float, default=1.5)
    parser.add_argument("--scale", type=int, default=3)
    return parser.parse_args()


def rmse(values: list[float]) -> float:
    return math.sqrt(sum(value * value for value in values) / len(values))


def main() -> None:
    args = parse_args()
    points_path = args.result_dir / "two_layer_shared_points.csv"
    output_dir = args.result_dir / "detected_internal_anomaly_visualizations"
    output_dir.mkdir(exist_ok=True)
    rows = list(csv.DictReader(points_path.open(newline="", encoding="utf-8")))
    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["frame_index"], row["frame_label"], row["board_id"])].append(row)

    selected = []
    for key, group in grouped.items():
        internal = [float(row["residual_norm"]) for row in group
                    if row["point_type"] == "internal"]
        if internal and rmse(internal) >= args.threshold_px:
            selected.append((rmse(internal), key, group))
    selected.sort(reverse=True)

    report_rows = []
    thumbnails = []
    for rank, (internal_rmse, (frame_index, label, board_id), group) in enumerate(selected, 1):
        image_path = args.image_dir / f"{label}.png"
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Missing source image: {image_path}")
        xs = [float(row["observed_x"]) for row in group]
        ys = [float(row["observed_y"]) for row in group]
        padding = 110
        x0, x1 = max(0, int(min(xs)) - padding), min(image.shape[1], int(max(xs)) + padding)
        y0, y1 = max(0, int(min(ys)) - padding), min(image.shape[0], int(max(ys)) + padding)
        crop = image[y0:y1, x0:x1].copy()
        crop = cv2.resize(crop, None, fx=args.scale, fy=args.scale, interpolation=cv2.INTER_CUBIC)
        outer_count = internal_count = 0
        for row in group:
            x = int(round((float(row["observed_x"]) - x0) * args.scale))
            y = int(round((float(row["observed_y"]) - y0) * args.scale))
            if row["point_type"] == "outer":
                outer_count += 1
                cv2.circle(crop, (x, y), 10, (60, 220, 80), 2, cv2.LINE_AA)
            else:
                internal_count += 1
                cv2.circle(crop, (x, y), 6, (0, 170, 255), 2, cv2.LINE_AA)
        banner_height = 110
        canvas = cv2.copyMakeBorder(crop, banner_height, 0, 0, 0, cv2.BORDER_CONSTANT,
                                    value=(18, 18, 18))
        title = f"rank {rank}: {label} | Board {board_id} | detection only"
        detail = (f"internal RMSE in completed shared BA: {internal_rmse:.3f}px  |  "
                  f"outer={outer_count}, internal={internal_count}")
        legend = "green ring = detected Outer4; orange ring = detected internal point; no model projection drawn"
        cv2.putText(canvas, title, (20, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.82,
                    (240, 240, 240), 2, cv2.LINE_AA)
        cv2.putText(canvas, detail, (20, 69), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                    (220, 220, 220), 1, cv2.LINE_AA)
        cv2.putText(canvas, legend, (20, 97), cv2.FONT_HERSHEY_SIMPLEX, 0.47,
                    (185, 185, 185), 1, cv2.LINE_AA)
        filename = f"rank{rank:02d}_frame{int(frame_index):03d}_board{board_id}_{label}_detected_points.png"
        cv2.imwrite(str(output_dir / filename), canvas)
        thumbnail = cv2.resize(canvas, (420, int(canvas.shape[0] * 420 / canvas.shape[1])))
        thumbnails.append(thumbnail)
        report_rows.append({
            "rank": rank, "frame_index": frame_index, "frame_label": label,
            "board_id": board_id, "internal_rmse_px": f"{internal_rmse:.6f}",
            "outer_count": outer_count, "internal_count": internal_count, "file": filename,
        })

    with (output_dir / "anomaly_index.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(report_rows[0]))
        writer.writeheader()
        writer.writerows(report_rows)
    (output_dir / "README.txt").write_text(
        "Detection-only diagnostic. Calibration, poses, and detections were not changed.\n"
        "Orange rings are recorded internal observation pixels; green rings are recorded Outer4 pixels.\n"
        "The selection criterion is internal RMSE >= " + str(args.threshold_px) + " px in two_layer_shared_points.csv.\n",
        encoding="utf-8")

    columns = 3
    cell_height = max(image.shape[0] for image in thumbnails)
    rows_needed = math.ceil(len(thumbnails) / columns)
    padded = [cv2.copyMakeBorder(
        thumbnail, 0, cell_height - thumbnail.shape[0], 0, 0,
        cv2.BORDER_CONSTANT, value=(18, 18, 18)) for thumbnail in thumbnails]
    blank = cv2.copyMakeBorder(
        padded[0], 0, 0, 0, 0, cv2.BORDER_CONSTANT, value=(18, 18, 18))
    blank[:] = (18, 18, 18)
    sheet = cv2.vconcat([
        cv2.hconcat(row + [blank] * (columns - len(row)))
        for row in [padded[i:i + columns] for i in range(0, len(padded), columns)]
    ])
    cv2.imwrite(str(output_dir / "contact_sheet_detected_points.png"), sheet)
    print(f"Exported {len(report_rows)} detection-only anomaly crops to {output_dir}")


if __name__ == "__main__":
    main()
