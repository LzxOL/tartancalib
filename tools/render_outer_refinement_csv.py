#!/usr/bin/env python3
"""Render coarse/refined outer corners from the CSV diagnostic cache."""

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np


def load_rows(path, board_id):
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows = [row for row in rows if int(row["board_id"]) == board_id]
    if len(rows) != 1:
        raise RuntimeError(f"expected one board row in {path}, got {len(rows)}")
    row = rows[0]
    return np.array([[float(row[f"x{i}"]), float(row[f"y{i}"])] for i in range(4)])


def marker(image, point, color, label, scale=1.0):
    p = tuple(np.round(np.asarray(point) * scale).astype(int))
    size = max(12, int(round(28 * scale)))
    thick = max(2, int(round(2.5 * scale)))
    cv2.drawMarker(image, p, color, cv2.MARKER_TILTED_CROSS, size, thick, cv2.LINE_AA)
    cv2.putText(image, label, (p[0] + 10, p[1] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                max(0.5, 0.8 * scale), color, max(1, int(round(2 * scale))), cv2.LINE_AA)


def crop(image, points, extent=260):
    center = np.mean(points, axis=0)
    x, y = np.round(center).astype(int)
    left, top = x - extent, y - extent
    right, bottom = x + extent, y + extent
    out = np.zeros((2 * extent, 2 * extent, 3), dtype=image.dtype)
    sl, st, sr, sb = max(0, left), max(0, top), min(image.shape[1], right), min(image.shape[0], bottom)
    out[st - top:sb - top, sl - left:sr - left] = image[st:sb, sl:sr]
    return out, np.array([left, top])


def draw_pair(image, coarse, refined, scale, label_prefix=""):
    for i, (c, r) in enumerate(zip(coarse, refined)):
        c2, r2 = c * scale, r * scale
        cv2.arrowedLine(image, tuple(np.round(c2).astype(int)), tuple(np.round(r2).astype(int)),
                        (0, 220, 255), max(2, int(round(2 * scale))), cv2.LINE_AA, tipLength=0.16)
        marker(image, c, (0, 220, 255), f"{label_prefix}C{i}", scale)
        marker(image, r, (60, 255, 80), f"{label_prefix}R{i}", scale)


def draw_points(image, points, color, scale, prefix):
    for i, point in enumerate(points):
        marker(image, point, color, f"{prefix}{i}", scale)


def render(frame_label, coarse_csv, refined_csv, output, board_id, window_radii):
    coarse = load_rows(coarse_csv, board_id)
    refined = load_rows(refined_csv, board_id)
    with coarse_csv.open(newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if int(row["board_id"]) == board_id]
    row = rows[0]
    image_path = Path(row["image_path"])
    image = cv2.imread(str(image_path))
    if image is None and not image_path.is_absolute():
        image = cv2.imread(str(Path("tartancalib") / image_path))
    if image is None:
        raise RuntimeError(f"cannot read image: {image_path}")

    scale = 0.32
    overview = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    draw_pair(overview, coarse, refined, scale)
    cv2.rectangle(overview, (0, 0), (overview.shape[1], 78), (24, 24, 24), -1)
    shifts = np.linalg.norm(refined - coarse, axis=1)
    cv2.putText(overview, f"{frame_label}  Board {board_id} | C=before, R=after",
                (18, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(overview, "yellow arrow: coarse -> refined | green: refined corner",
                (18, 59), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (235, 235, 235), 1, cv2.LINE_AA)

    before_scale = scale / 2.0
    before = cv2.resize(image, None, fx=before_scale, fy=before_scale, interpolation=cv2.INTER_AREA)
    after = before.copy()
    after[:] = before
    draw_points(before, coarse, (0, 220, 255), before_scale, "C")
    draw_points(after, refined, (60, 255, 80), before_scale, "R")
    for panel, title in ((before, "Before refinement (coarse detector corners)"),
                         (after, "After refinement (subpixel corners)")):
        cv2.rectangle(panel, (0, 0), (panel.shape[1], 40), (24, 24, 24), -1)
        cv2.putText(panel, title, (14, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 1, cv2.LINE_AA)
    before_after = np.hstack((before, after))

    panels = []
    for i, (c, r, shift) in enumerate(zip(coarse, refined, shifts)):
        panel, origin = crop(image, np.array([c, r]), 280)
        local_c, local_r = c - origin, r - origin
        draw_pair(panel, np.array([local_c]), np.array([local_r]), 1.0, f"{i}:")
        cv2.rectangle(panel, (0, 0), (panel.shape[1], 38), (24, 24, 24), -1)
        cv2.putText(panel, f"corner {i}: window r={window_radii[i]} px | shift={shift:.2f} px",
                    (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 1, cv2.LINE_AA)
        panels.append(cv2.resize(panel, (overview.shape[1] // 2, overview.shape[1] // 2),
                                 interpolation=cv2.INTER_AREA))
    grid = np.vstack((np.hstack((panels[0], panels[1])), np.hstack((panels[2], panels[3]))))
    result = np.vstack((overview, before_after, grid))
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), result)
    print(output)
    print("shifts_px=" + ",".join(f"{x:.3f}" for x in shifts))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coarse", type=Path, required=True)
    parser.add_argument("--refined", type=Path, required=True)
    parser.add_argument("--frame-label", required=True)
    parser.add_argument("--board-id", type=int, default=3)
    parser.add_argument("--window-radii", default="48,48,48,48",
                        help="Four comma-separated subpixel window radii in pixels.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    window_radii = [int(value) for value in args.window_radii.split(",")]
    if len(window_radii) != 4 or any(radius < 1 for radius in window_radii):
        parser.error("--window-radii must contain four positive integer radii")
    render(args.frame_label, args.coarse, args.refined, args.output, args.board_id,
           window_radii)


if __name__ == "__main__":
    main()
