#!/usr/bin/env python3
"""Render coarse-to-subpixel outer-corner refinement diagnostics from a cache."""

import argparse
from pathlib import Path

import cv2
import numpy as np


def read_point(node):
    return (float(node.at(0).real()), float(node.at(1).real()))


def read_board(cache_path, board_id):
    storage = cv2.FileStorage(str(cache_path), cv2.FILE_STORAGE_READ)
    if not storage.isOpened():
        raise RuntimeError(f"Cannot open cache: {cache_path}")
    root = storage.getNode("frame_result")
    image_path = root.getNode("absolute_image_path").string() if not root.empty() else ""
    if not image_path:
        image_path = storage.getNode("absolute_image_path").string()
    boards = (root.getNode("board_measurements") if not root.empty()
              else storage.getNode("detections"))
    for index in range(boards.size()):
        board = boards.at(index)
        if int(board.getNode("board_id").real()) != board_id:
            continue
        detection = board.getNode("detection")
        outer = detection.getNode("outer_detection") if not detection.empty() else board
        coarse = [read_point(outer.getNode("coarse_corners_original_image").at(i))
                  for i in range(4)]
        refined = [read_point(outer.getNode("refined_corners_original_image").at(i))
                   for i in range(4)]
        debug = outer.getNode("corner_verification_debug")
        windows = [int(debug.at(i).getNode("subpix_window_radius").real())
                   for i in range(4)]
        applied = [bool(int(debug.at(i).getNode("subpix_applied").real()))
                   for i in range(4)]
        rollback = [bool(int(debug.at(i).getNode("subpix_unstable_rollback_detected").real()))
                    for i in range(4)]
        disagreement = [bool(int(debug.at(i).getNode("subpix_scale_disagreement_detected").real()))
                        if not debug.at(i).getNode("subpix_scale_disagreement_detected").empty()
                        else False for i in range(4)]
        probe_radii = [int(debug.at(i).getNode("subpix_scale_probe_window_radius").real())
                       if not debug.at(i).getNode("subpix_scale_probe_window_radius").empty()
                       else 0 for i in range(4)]
        probe_deltas = [float(debug.at(i).getNode("subpix_scale_probe_endpoint_delta").real())
                        if not debug.at(i).getNode("subpix_scale_probe_endpoint_delta").empty()
                        else 0.0 for i in range(4)]
        summary = outer.getNode("local_patch_rescue_summary").string()
        storage.release()
        return (image_path, coarse, refined, windows, applied, rollback,
                disagreement, probe_radii, probe_deltas, summary)
    storage.release()
    raise RuntimeError(f"Board {board_id} is not present in {cache_path}")


def draw_marker(image, point, color, label, scale=1.0):
    p = tuple(np.round(np.asarray(point) * scale).astype(int))
    cv2.drawMarker(image, p, color, markerType=cv2.MARKER_TILTED_CROSS,
                   markerSize=max(10, round(22 * scale)), thickness=max(1, round(2 * scale)),
                   line_type=cv2.LINE_AA)
    cv2.putText(image, label, (p[0] + 8, p[1] - 8), cv2.FONT_HERSHEY_SIMPLEX,
                max(0.4, 0.7 * scale), color, max(1, round(2 * scale)), cv2.LINE_AA)


def draw_corner_overlay(image, coarse, refined, window, index, scale=1.0):
    c = tuple(np.round(np.asarray(coarse) * scale).astype(int))
    r = tuple(np.round(np.asarray(refined) * scale).astype(int))
    displacement = float(np.linalg.norm(np.asarray(refined) - np.asarray(coarse)))
    cv2.arrowedLine(image, c, r, (0, 210, 255), max(1, round(2 * scale)),
                    cv2.LINE_AA, tipLength=0.14)
    cv2.circle(image, r, max(1, round(window * scale)), (255, 120, 0),
               max(1, round(2 * scale)), cv2.LINE_AA)
    draw_marker(image, coarse, (0, 210, 255), f"C{index}", scale)
    draw_marker(image, refined, (0, 255, 80), f"R{index}", scale)
    return displacement


def crop_with_padding(image, center, half_extent):
    x, y = (int(round(center[0])), int(round(center[1])))
    left, top = x - half_extent, y - half_extent
    right, bottom = x + half_extent, y + half_extent
    canvas = np.zeros((2 * half_extent, 2 * half_extent, 3), dtype=image.dtype)
    src_l, src_t = max(0, left), max(0, top)
    src_r, src_b = min(image.shape[1], right), min(image.shape[0], bottom)
    dst_l, dst_t = src_l - left, src_t - top
    canvas[dst_t:dst_t + (src_b - src_t), dst_l:dst_l + (src_r - src_l)] = \
        image[src_t:src_b, src_l:src_r]
    return canvas, (left, top)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path,
                        help="Detection cache used for the standard or cache-backed sweep view.")
    parser.add_argument("--board-id", type=int, required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--image", type=Path,
                        help="Image for a targeted sweep when the original cache is unavailable.")
    parser.add_argument("--coarse-seed",
                        help="Targeted-sweep coarse seed as 'x,y'; requires --image.")
    parser.add_argument("--sweep-corner", type=int,
                        help="Render a subpixel-window sweep for this corner index.")
    parser.add_argument("--sweep-radii", default="43,48,68,90",
                        help="Comma-separated cornerSubPix window radii for --sweep-corner.")
    parser.add_argument("--decision-radii",
                        help="Two radii, 'nominal,active', to annotate the close-edge stability decision.")
    args = parser.parse_args()

    if args.cache:
        (image_path, coarse, refined, windows, applied, rollback,
         disagreement, probe_radii, probe_deltas, summary) = read_board(
            args.cache, args.board_id)
    elif args.sweep_corner is not None and args.image and args.coarse_seed:
        values = [float(value) for value in args.coarse_seed.split(",")]
        if len(values) != 2:
            parser.error("--coarse-seed must be 'x,y'")
        image_path = str(args.image)
        coarse = [tuple(values)]
        refined = windows = applied = rollback = disagreement = probe_radii = probe_deltas = summary = None
        if args.sweep_corner != 0:
            parser.error("targeted --image/--coarse-seed sweeps require --sweep-corner 0")
    else:
        parser.error("provide --cache, or provide --image and --coarse-seed for a targeted sweep")
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Cannot read image: {image_path}")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if args.sweep_corner is not None:
        index = args.sweep_corner
        if index < 0 or index >= 4:
            raise ValueError("--sweep-corner must be in [0, 3]")
        radii = [int(value) for value in args.sweep_radii.split(",") if value]
        if not radii or any(radius <= 0 for radius in radii):
            raise ValueError("--sweep-radii must contain positive integer radii")
        seed = np.asarray(coarse[index], dtype=np.float32).reshape(1, 1, 2)
        final_points = []
        for radius in radii:
            point = seed.copy()
            cv2.cornerSubPix(
                gray, point, (radius, radius), (-1, -1),
                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1))
            final_points.append(tuple(point[0, 0]))

        center = np.mean(np.asarray([coarse[index], *final_points]), axis=0)
        half_extent = 175
        overview, origin = crop_with_padding(image, center, half_extent)
        seed_local = (coarse[index][0] - origin[0], coarse[index][1] - origin[1])
        draw_marker(overview, seed_local, (0, 210, 255), "coarse seed")
        colors = [(70, 255, 70), (255, 180, 0), (255, 80, 220),
                  (70, 220, 255), (255, 90, 90), (200, 120, 255)]
        for radius, final, color in zip(radii, final_points, colors):
            final_local = (final[0] - origin[0], final[1] - origin[1])
            cv2.arrowedLine(overview, tuple(map(int, np.round(seed_local))),
                            tuple(map(int, np.round(final_local))), color, 2,
                            cv2.LINE_AA, tipLength=0.12)
            cv2.drawMarker(overview, tuple(map(int, np.round(final_local))), color,
                          cv2.MARKER_TILTED_CROSS, 17, 2, cv2.LINE_AA)
            cv2.putText(overview, f"r={radius}",
                        (int(final_local[0]) + 7, int(final_local[1]) - 7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)

        panels = []
        for radius, final, color in zip(radii, final_points, colors):
            panel, panel_origin = crop_with_padding(image, center, half_extent)
            seed_panel = (coarse[index][0] - panel_origin[0],
                          coarse[index][1] - panel_origin[1])
            final_panel = (final[0] - panel_origin[0], final[1] - panel_origin[1])
            cv2.circle(panel, tuple(map(int, np.round(seed_panel))), radius, color, 2,
                       cv2.LINE_AA)
            cv2.arrowedLine(panel, tuple(map(int, np.round(seed_panel))),
                            tuple(map(int, np.round(final_panel))), color, 2,
                            cv2.LINE_AA, tipLength=0.12)
            draw_marker(panel, seed_panel, (0, 210, 255), "C")
            draw_marker(panel, final_panel, color, "R")
            shift = float(np.linalg.norm(np.asarray(final) - np.asarray(coarse[index])))
            cv2.rectangle(panel, (0, 0), (panel.shape[1], 35), (24, 24, 24), -1)
            cv2.putText(panel, f"window radius={radius}px, shift={shift:.2f}px",
                        (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                        (255, 255, 255), 1, cv2.LINE_AA)
            panels.append(panel)
        # Preserve the two 350 px detail panels side-by-side. Downsampling the
        # grid back to the overview width made window labels unreadable.
        render_width = overview.shape[1] * 2
        overview = cv2.resize(overview, (render_width, render_width),
                              interpolation=cv2.INTER_NEAREST)
        header = np.full((96, render_width, 3), 24, dtype=np.uint8)
        shifts = [float(np.linalg.norm(np.asarray(final) - np.asarray(coarse[index])))
                  for final in final_points]
        endpoint_spread = max(
            float(np.linalg.norm(np.asarray(first) - np.asarray(second)))
            for first in final_points for second in final_points)
        cv2.putText(header, f"Board {args.board_id}, C{index}: same seed, different cornerSubPix windows",
                    (18, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 255, 255), 1,
                    cv2.LINE_AA)
        cv2.putText(header, f"Endpoint spread={endpoint_spread:.2f}px; shifts="
                    + ", ".join(f"{radius}:{shift:.2f}" for radius, shift in zip(radii, shifts)),
                    (18, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (230, 230, 230), 1,
                    cv2.LINE_AA)
        if args.decision_radii:
            values = [int(value) for value in args.decision_radii.split(",") if value]
            if len(values) != 2:
                parser.error("--decision-radii must be 'nominal,active'")
            nominal_radius, active_radius = values
            try:
                nominal_index = radii.index(nominal_radius)
                active_index = radii.index(active_radius)
            except ValueError:
                parser.error("--decision-radii values must occur in --sweep-radii")
            endpoint_delta = float(np.linalg.norm(
                np.asarray(final_points[nominal_index]) -
                np.asarray(final_points[active_index])))
            limit = 0.35 * max(nominal_radius, active_radius)
            verdict = "REJECT unstable corner" if endpoint_delta > limit else "ACCEPT stable corner"
            verdict_color = (70, 80, 255) if endpoint_delta > limit else (70, 255, 70)
            cv2.rectangle(header, (0, 72), (header.shape[1], header.shape[0]), (38, 38, 38), -1)
            cv2.putText(header,
                        f"Decision: nominal r={nominal_radius}px vs active r={active_radius}px: "
                        f"delta={endpoint_delta:.2f}px, allowed={limit:.2f}px -> {verdict}",
                        (18, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.43, verdict_color, 1,
                        cv2.LINE_AA)
        rows = []
        for start in range(0, len(panels), 2):
            row = panels[start:start + 2]
            if len(row) == 1:
                row.append(np.zeros_like(row[0]))
            rows.append(np.hstack(row))
        grid = np.vstack(rows)
        grid = cv2.resize(grid, (render_width, render_width),
                          interpolation=cv2.INTER_AREA)
        result = np.vstack((header, overview, grid))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(args.output), result):
            raise RuntimeError(f"Cannot write {args.output}")
        print(args.output)
        print("sweep_results=" + ", ".join(
            f"r={radius}:({point[0]:.2f},{point[1]:.2f}),shift={shift:.2f}"
            for radius, point, shift in zip(radii, final_points, shifts)))
        return

    overview_scale = 0.42
    overview = cv2.resize(image, None, fx=overview_scale, fy=overview_scale,
                          interpolation=cv2.INTER_AREA)
    displacements = []
    for index in range(4):
        displacements.append(draw_corner_overlay(
            overview, coarse[index], refined[index], windows[index], index, overview_scale))

    top_height = 110
    overview = cv2.copyMakeBorder(overview, top_height, 0, 0, 0,
                                  cv2.BORDER_CONSTANT, value=(24, 24, 24))
    lines = [
        f"Board {args.board_id}: geometry-prior rescue outer-corner refinement",
        "Yellow C = coarse seed, green R = final subpixel corner, orange circle = subpixel window radius",
        " | ".join(
            f"C{i}: shift={displacements[i]:.1f}px, window={windows[i]}px, "
            f"subpix={'yes' if applied[i] else 'no'}, rollback={'yes' if rollback[i] else 'no'}, "
            f"scale_disagreement={'yes' if disagreement[i] else 'no'}"
            for i in range(4)),
    ]
    for index, line in enumerate(lines):
        cv2.putText(overview, line, (26, 35 + index * 29), cv2.FONT_HERSHEY_SIMPLEX,
                    0.64 if index < 2 else 0.54, (245, 245, 245), 1, cv2.LINE_AA)

    panels = []
    half_extent = 260
    for index in range(4):
        center = np.mean(np.array([coarse[index], refined[index]]), axis=0)
        crop, origin = crop_with_padding(image, center, half_extent)
        c_local = (coarse[index][0] - origin[0], coarse[index][1] - origin[1])
        r_local = (refined[index][0] - origin[0], refined[index][1] - origin[1])
        draw_corner_overlay(crop, c_local, r_local, windows[index], index)
        caption = (f"C{index}: {displacements[index]:.1f}px  |  subpix window: {windows[index]}px"
                   f"  |  probe: {probe_radii[index]}px/{probe_deltas[index]:.1f}px"
                   f"  |  unstable: {'yes' if disagreement[index] else 'no'}")
        cv2.rectangle(crop, (0, 0), (crop.shape[1], 34), (24, 24, 24), -1)
        cv2.putText(crop, caption, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                    (255, 255, 255), 1, cv2.LINE_AA)
        panels.append(crop)
    panel_grid = np.vstack((np.hstack((panels[0], panels[1])),
                            np.hstack((panels[2], panels[3]))))
    panel_grid = cv2.resize(panel_grid, (overview.shape[1],
                                         round(panel_grid.shape[0] * overview.shape[1] /
                                               panel_grid.shape[1])),
                            interpolation=cv2.INTER_AREA)

    result = np.vstack((overview, panel_grid))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), result):
        raise RuntimeError(f"Cannot write {args.output}")
    print(args.output)
    print(summary)


if __name__ == "__main__":
    main()
