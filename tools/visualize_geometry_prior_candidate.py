#!/usr/bin/env python3

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Render one rejected Stage5 geometry-prior board candidate.")
    parser.add_argument("--result-dir", required=True, type=Path)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--frame", required=True, type=int)
    parser.add_argument("--board", required=True, type=int)
    parser.add_argument("--source", default="bootstrap_visible_refit_single")
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def point(row, prefix, index):
    return np.array([
        float(row[f"{prefix}_corner_u{index}"]),
        float(row[f"{prefix}_corner_v{index}"]),
    ], dtype=np.float32)


def read_candidate(csv_path, frame, board, source):
    with csv_path.open(newline="") as stream:
        rows = [row for row in csv.DictReader(stream)
                if int(row["frame_index"]) == frame
                and int(row["missing_board_id"]) == board]
    exact = [row for row in rows if row["prediction_source"] == source]
    if exact:
        return exact[0]
    if not rows:
        raise RuntimeError(f"No candidate for frame {frame}, board {board}")
    raise RuntimeError(
        "Source not found. Available sources: "
        + ", ".join(sorted({row["prediction_source"] for row in rows})))


def draw_text(image, text, origin, scale=0.62, color=(255, 255, 255)):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (width, height), baseline = cv2.getTextSize(text, font, scale, 1)
    x, y = origin
    cv2.rectangle(image, (x - 5, y - height - 6),
                  (x + width + 5, y + baseline + 5), (20, 20, 20), -1)
    cv2.putText(image, text, (x, y), font, scale, color, 1, cv2.LINE_AA)


def draw_candidate(image, row, offset=(0, 0), scale=1.0, detailed=True):
    prediction = [point(row, "predicted", index) for index in range(4)]
    refinement = [point(row, "refined", index) for index in range(4)]
    failed = set()
    for item in row["spherical_refine_failure_summary"].split(";"):
        if ":" in item:
            failed.add(int(item.split(":", 1)[0]))

    def transform(value):
        return tuple(np.rint((value - np.asarray(offset)) * scale).astype(int))

    predicted_polygon = np.array([transform(value) for value in prediction])
    refined_polygon = np.array([transform(value) for value in refinement])
    cv2.polylines(image, [predicted_polygon], True, (0, 215, 255), 5,
                  cv2.LINE_AA)
    cv2.polylines(image, [refined_polygon], True, (255, 80, 220), 3,
                  cv2.LINE_AA)

    for index, (predicted, refined) in enumerate(zip(prediction, refinement)):
        p = transform(predicted)
        r = transform(refined)
        cv2.drawMarker(image, p, (0, 215, 255), cv2.MARKER_TILTED_CROSS,
                       24, 3, cv2.LINE_AA)
        cv2.line(image, p, r, (210, 120, 210), 2, cv2.LINE_AA)
        if index in failed:
            cv2.drawMarker(image, r, (40, 40, 255), cv2.MARKER_CROSS,
                           30, 4, cv2.LINE_AA)
            status = "edge-fit failed"
            color = (80, 80, 255)
        else:
            cv2.circle(image, r, 11, (60, 230, 80), 4, cv2.LINE_AA)
            status = "refined"
            color = (80, 240, 100)
        if detailed:
            draw_text(image, f"c{index}: {status}", (r[0] + 12, r[1] - 10),
                      0.48, color)
    return prediction + refinement


def main():
    args = parse_args()
    row = read_candidate(
        args.result_dir / "geometry_prior_outer_seed_candidates.csv",
        args.frame, args.board, args.source)
    original = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if original is None:
        raise RuntimeError(f"Cannot read image: {args.image}")

    full = original.copy()
    points = draw_candidate(full, row)
    draw_text(full,
              f"Board {args.board} rejected candidate (diagnostic only; NOT used by BA)",
              (24, 42), 0.82, (255, 255, 255))
    draw_text(full, "yellow: topology prediction | magenta: refined quad | green: success | red: failure",
              (24, 76), 0.58, (255, 255, 255))

    coordinates = np.vstack(points)
    x0, y0 = np.floor(coordinates.min(axis=0) - 140).astype(int)
    x1, y1 = np.ceil(coordinates.max(axis=0) + 140).astype(int)
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(original.shape[1], x1), min(original.shape[0], y1)
    crop = original[y0:y1, x0:x1].copy()
    zoom_scale = min(1.35, 1500.0 / max(crop.shape[:2]))
    if zoom_scale != 1.0:
        crop = cv2.resize(crop, None, fx=zoom_scale, fy=zoom_scale,
                          interpolation=cv2.INTER_AREA)
    draw_candidate(crop, row, offset=(x0, y0), scale=zoom_scale)
    draw_text(crop, f"frame {args.frame} | board {args.board} | {args.source}",
              (20, 38), 0.66, (255, 255, 255))
    draw_text(crop,
              "spherical refinement: "
              f"{row['spherical_refine_successful_corner_count']}/4 corners | "
              f"edge support={float(row['edge_support_ratio']):.3f}",
              (20, 70), 0.55, (255, 255, 255))
    draw_text(crop, "historical reject: " + row["reject_reason"],
              (20, 102), 0.48, (100, 140, 255))

    args.output.mkdir(parents=True, exist_ok=True)
    full_path = args.output / f"frame_{args.frame}_board_{args.board}_candidate_full.png"
    zoom_path = args.output / f"frame_{args.frame}_board_{args.board}_candidate_zoom.png"
    cv2.imwrite(str(full_path), full)
    cv2.imwrite(str(zoom_path), crop)
    print(full_path)
    print(zoom_path)


if __name__ == "__main__":
    main()
