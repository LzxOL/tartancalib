#!/usr/bin/env python3
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Render Stage5 observed/projected corner overlays from backend point CSVs."
    )
    parser.add_argument("--points-csv", required=True)
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--point-radius", type=int, default=4)
    parser.add_argument("--line-thickness", type=int, default=1)
    return parser.parse_args()


def image_path_for_label(image_dir, frame_label):
    candidates = [
        image_dir / frame_label,
        image_dir / f"{frame_label}.png",
        image_dir / f"{frame_label}.jpg",
        image_dir / f"{frame_label}.jpeg",
        image_dir / f"{frame_label}.bmp",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def safe_float(row, key):
    try:
        return float(row[key])
    except Exception:
        return None


def first_float(row, keys):
    for key in keys:
        value = safe_float(row, key)
        if value is not None:
            return value
    return None


def main():
    args = parse_args()
    points_csv = Path(args.points_csv)
    image_dir = Path(args.image_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows_by_frame = defaultdict(list)
    with points_csv.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            frame_index = row.get("frame_index", "")
            frame_label = row.get("frame_label", "")
            if not frame_label:
                continue
            rows_by_frame[(frame_index, frame_label)].append(row)

    summary_rows = []
    rendered = 0
    for (frame_index, frame_label), rows in sorted(
        rows_by_frame.items(), key=lambda item: int(item[0][0])
    ):
        if args.max_frames > 0 and rendered >= args.max_frames:
            break
        image_path = image_path_for_label(image_dir, frame_label)
        if image_path is None:
            summary_rows.append([frame_index, frame_label, 0, "", "missing_image"])
            continue
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            summary_rows.append([frame_index, frame_label, 0, str(image_path), "read_failed"])
            continue

        overlay = image.copy()
        residuals = []
        board_ids = set()
        for row in rows:
            ox = safe_float(row, "observed_x")
            oy = safe_float(row, "observed_y")
            px = first_float(row, ["predicted_x", "backend_predicted_x"])
            py = first_float(row, ["predicted_y", "backend_predicted_y"])
            if ox is None or oy is None or px is None or py is None:
                continue
            obs = (int(round(ox)), int(round(oy)))
            pred = (int(round(px)), int(round(py)))
            board_ids.add(row.get("board_id", ""))
            residuals.append(float(np.hypot(px - ox, py - oy)))
            cv2.line(overlay, obs, pred, (0, 255, 255), args.line_thickness, cv2.LINE_AA)
            cv2.circle(overlay, obs, args.point_radius, (0, 80, 255), -1, cv2.LINE_AA)
            cv2.circle(overlay, pred, args.point_radius, (0, 220, 0), 1, cv2.LINE_AA)

        if residuals:
            rmse = float(np.sqrt(np.mean(np.square(residuals))))
        else:
            rmse = 0.0
        label = (
            f"frame {frame_index} | boards {len(board_ids)} | "
            f"points {len(residuals)} | rmse {rmse:.3f}px"
        )
        cv2.rectangle(overlay, (16, 16), (min(1320, 16 + 18 * len(label)), 74),
                      (0, 0, 0), -1)
        cv2.putText(overlay, label, (28, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(overlay, "observed: orange  projected: green  residual: yellow",
                    (28, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    (255, 255, 255), 2, cv2.LINE_AA)

        safe_label = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in frame_label)
        output_path = output_dir / f"frame_{int(frame_index):06d}_{safe_label}.png"
        cv2.imwrite(str(output_path), overlay)
        summary_rows.append(
            [frame_index, frame_label, len(residuals), str(image_path), f"{rmse:.6f}"]
        )
        rendered += 1

    with (output_dir / "overlay_summary.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frame_index", "frame_label", "point_count", "image_path", "rmse_or_status"])
        writer.writerows(summary_rows)

    print(f"rendered_frames: {rendered}")
    print(f"output_dir: {output_dir}")


if __name__ == "__main__":
    main()
