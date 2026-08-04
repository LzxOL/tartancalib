#!/usr/bin/env python3
"""Render qualitative raw-fisheye epipolar-locus comparisons.

The renderer consumes a completed Peripheral Epipolar Consistency experiment.
It never re-detects features, refits an extrinsic, or selects points from an
error value. A deterministic frame and six frozen matches are selected from
the pre-existing raw-image correspondence set, then each stereo system maps
the same left observation to its predicted locus in the raw right image.
"""

from __future__ import annotations

import argparse
import base64
import csv
import html
import importlib.util
import json
import math
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DOWNSTREAM = Path(__file__).with_name("run_rectification_disparity_visualization.py")
REGIONS = ("central_0_30", "middle_30_60", "peripheral_60_80")
POINTS_PER_REGION = 2
POINT_IDS = tuple(
    f"{prefix}{index}"
    for prefix in ("C", "M", "P")
    for index in range(1, POINTS_PER_REGION + 1)
)
# One bright color per polar-angle region: central, middle, peripheral.
# Labels make individual correspondences identifiable without color alone.
POINT_COLORS = (
    "#0072B2", "#0072B2",
    "#009E73", "#009E73",
    "#D55E00", "#D55E00",
)
LEGACY_POINT_COLORS = tuple(
    (int(color[5:7], 16), int(color[3:5], 16), int(color[1:3], 16))
    for color in POINT_COLORS
)
REGION_LABELS = {
    "central_0_30": "center (0-30 deg)",
    "middle_30_60": "middle (30-60 deg)",
    "peripheral_60_80": "peripheral (60-80 deg)",
    "outside_0_80": "outside",
}
CANVAS_WIDTH = 4320
CANVAS_HEIGHT = 1900
PANEL_MARGIN = 22
PANEL_GAP = 20
PANEL_TITLE_HEIGHT = 88
INSET_GAP = 24
INSET_STRIP_HEIGHT = 318

# Publication-scale drawing parameters.  These values are chosen for the
# final 7.2-inch export rather than for inspection at full raster resolution.
# Markers use thin colored strokes without white halos so the final figure
# keeps a conventional technical-drawing look after page reduction.
SOURCE_MARKER_RADIUS = 12.0
GLOBAL_LOCUS_WIDTH = 4.0
GLOBAL_LOCUS_OPACITY = 0.62
LOCAL_LOCUS_WIDTH = 5.0
RESIDUAL_WIDTH = 3.5
CLOSEST_POINT_RADIUS = 11.0
CLOSEST_POINT_WIDTH = 3.0
OBSERVED_CROSS_HALF_SIZE = 13.0
OBSERVED_CROSS_WIDTH = 3.0
INSET_LOCAL_LOCUS_WIDTH = 7.0
INSET_RESIDUAL_WIDTH = 5.0
INSET_CLOSEST_POINT_RADIUS = 15.0
INSET_CLOSEST_POINT_WIDTH = 4.0
INSET_OBSERVED_CROSS_HALF_SIZE = 22.0
INSET_OBSERVED_CROSS_WIDTH = 5.0


def fail(message: str) -> None:
    raise RuntimeError(message)


def load_downstream() -> Any:
    spec = importlib.util.spec_from_file_location("stereo_downstream_loci", DOWNSTREAM)
    if spec is None or spec.loader is None:
        fail(f"cannot import {DOWNSTREAM}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


D = load_downstream()


def read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        fail(f"missing required frozen artifact: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def normalized(vector: np.ndarray) -> np.ndarray | None:
    norm = float(np.linalg.norm(vector))
    if not math.isfinite(norm) or norm <= 1e-12:
        return None
    return vector / norm


def load_frozen_inputs(experiment_dir: Path) -> tuple[dict[int, Any], list[dict[str, Any]]]:
    pair_rows = read_csv(experiment_dir / "frame_manifest.csv")
    match_rows = read_csv(experiment_dir / "frozen_matches.csv")
    pairs: dict[int, Any] = {}
    for row in pair_rows:
        frame_id = int(row["frame_id"])
        pairs[frame_id] = D.ImagePair(
            frame_id, Path(row["left_image"]), Path(row["right_image"]),
            0.0, 0.0, 0.0, 0.0, int(row["timestamp_delta_ns"]),
        )
    matches: list[dict[str, Any]] = []
    for row in match_rows:
        theta = float(row["polar_proxy_deg"])
        region = (
            "central_0_30" if theta < 30.0 else
            "middle_30_60" if theta < 60.0 else
            "peripheral_60_80" if theta < 80.0 else "outside_0_80"
        )
        matches.append({
            "frame_id": int(row["frame_id"]),
            "match_rank": int(row["match_rank"]),
            "u_left": float(row["u_left"]), "v_left": float(row["v_left"]),
            "u_right": float(row["u_right"]), "v_right": float(row["v_right"]),
            "polar_proxy_deg": theta, "region": region,
        })
    if not pairs or not matches:
        fail("frozen frame manifest or match set is empty")
    return pairs, matches


def choose_display_matches(
    matches: list[dict[str, Any]],
    selection_mode: str = "spatial_max",
    selection_seed: int = 20260804,
) -> tuple[int, list[dict[str, Any]]]:
    """Choose two spatially separated frozen matches per radial region.

    Selection uses only frame IDs, match ranks, and image coordinates. Within
    each polar band, the selected pair maximizes normalized image-plane
    separation, which exposes both sides of the available field of view. It
    never reads either method's locus or epipolar error.
    """
    by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in matches:
        if row["region"] in REGIONS:
            by_frame[int(row["frame_id"])].append(row)
    if not by_frame:
        fail("frozen match set has no center-to-periphery observations")

    def frame_key(item: tuple[int, list[dict[str, Any]]]) -> tuple[int, int, int, int]:
        frame_id, rows = item
        counts = {region: sum(row["region"] == region for row in rows) for region in REGIONS}
        complete = int(all(counts[region] >= POINTS_PER_REGION for region in REGIONS))
        covered_slots = sum(min(counts[region], POINTS_PER_REGION) for region in REGIONS)
        weighted_pool = (3 * counts["peripheral_60_80"]
                         + 2 * counts["middle_30_60"]
                         + counts["central_0_30"])
        return complete, covered_slots, weighted_pool, -frame_id

    frame_id, candidates = max(by_frame.items(), key=frame_key)
    counts = {region: sum(row["region"] == region for row in candidates) for region in REGIONS}
    missing = [region for region in REGIONS if counts[region] < POINTS_PER_REGION]
    if missing:
        fail("no frozen frame contains enough matches in every radial region; "
             f"best frame {frame_id} is missing {missing}")

    if selection_mode not in {"spatial_max", "random_spatial"}:
        fail(f"unsupported qualitative selection mode: {selection_mode}")
    rng = np.random.default_rng(selection_seed + 1000003 * frame_id)
    selected: list[dict[str, Any]] = []
    for region in REGIONS:
        region_rows = sorted(
            (row for row in candidates if row["region"] == region),
            key=lambda row: (row["match_rank"], row["u_left"], row["v_left"]),
        )
        if len(region_rows) < POINTS_PER_REGION:
            fail(f"region {region} has fewer than {POINTS_PER_REGION} frozen matches")

        # The sensor dimensions are fixed by the stored raw coordinates. The
        # pairwise score is normalized so horizontal and vertical coverage have
        # the same interpretation on this non-square image.
        max_u = max(row["u_left"] for row in candidates)
        min_u = min(row["u_left"] for row in candidates)
        max_v = max(row["v_left"] for row in candidates)
        min_v = min(row["v_left"] for row in candidates)
        width = max(max_u - min_u, 1.0)
        height = max(max_v - min_v, 1.0)

        def separation(pair: tuple[dict[str, Any], dict[str, Any]]) -> tuple[float, int, int]:
            first, second = pair
            du = (first["u_left"] - second["u_left"]) / width
            dv = (first["v_left"] - second["v_left"]) / height
            return (math.hypot(du, dv), -first["match_rank"], -second["match_rank"])

        pairs = [
            (first, second)
            for index, first in enumerate(region_rows)
            for second in region_rows[index + 1:]
        ]
        if selection_mode == "spatial_max":
            pair = max(pairs, key=separation)
        else:
            max_distance = max(separation(candidate)[0] for candidate in pairs)
            # Preserve broad image-plane support while randomly drawing the
            # shown correspondences without consulting either method's error.
            eligible = [
                candidate for candidate in pairs
                if separation(candidate)[0] >= 0.55 * max_distance
            ]
            pair = eligible[int(rng.integers(len(eligible)))]
        selected.extend(sorted(pair, key=lambda row: row["match_rank"]))
    return frame_id, selected


def snap_manual_points(
    frame_matches: list[dict[str, Any]],
    raw_points: list[Any],
    radius: float = 150.0,
) -> list[dict[str, Any]]:
    """Snap clicked left-image coordinates to distinct frozen matches."""
    expected = 3 * POINTS_PER_REGION
    if len(raw_points) != expected:
        fail(f"manual selection requires {expected} points, got {len(raw_points)}")
    unused = list(frame_matches)
    snapped: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_points):
        if isinstance(raw, dict):
            u = float(raw.get("u", raw.get("u_left")))
            v = float(raw.get("v", raw.get("v_left")))
        else:
            u, v = float(raw[0]), float(raw[1])
        if not unused:
            fail(f"not enough distinct frozen matches for manual point {index + 1}")
        best = min(
            unused,
            key=lambda row: math.hypot(row["u_left"] - u, row["v_left"] - v),
        )
        distance = math.hypot(best["u_left"] - u, best["v_left"] - v)
        if distance > radius:
            fail(
                f"manual point {index + 1} at ({u:.1f}, {v:.1f}) is "
                f"{distance:.1f}px from the nearest frozen match; "
                f"click closer to a feature or increase --snap-radius"
            )
        snapped.append(best)
        unused.remove(best)
    return snapped


def pick_manual_points(
    left_path: Path,
    frame_matches: list[dict[str, Any]],
    count: int | None = None,
    radius: float = 150.0,
) -> list[dict[str, Any]]:
    """Open an interactive left-image window for manual point selection."""
    expected = count or 3 * POINTS_PER_REGION
    left = cv2.imread(str(left_path), cv2.IMREAD_COLOR)
    if left is None:
        fail(f"cannot load left image for manual selection: {left_path}")
    display, scale = legacy_resize_for_figure(left, 1100)
    window = "Pick points on LEFT image"
    clicks: list[tuple[float, float]] = []
    height, width = display.shape[:2]
    center = (
        int(round((left.shape[1] - 1.0) * scale / 2.0)),
        int(round((left.shape[0] - 1.0) * scale / 2.0)),
    )
    half_px = left.shape[1] / 2.0
    region_radii = {
        "central_0_30": int(round((30.0 / 90.0) * half_px * scale)),
        "middle_30_60": int(round((60.0 / 90.0) * half_px * scale)),
        "peripheral_60_80": int(round((80.0 / 90.0) * half_px * scale)),
    }
    region_bgr = {
        "central_0_30": LEGACY_POINT_COLORS[0],
        "middle_30_60": LEGACY_POINT_COLORS[2],
        "peripheral_60_80": LEGACY_POINT_COLORS[4],
    }
    region_base = display.copy()
    overlay = region_base.copy()
    cv2.circle(overlay, center, region_radii["peripheral_60_80"],
               region_bgr["peripheral_60_80"], -1)
    cv2.circle(overlay, center, region_radii["middle_30_60"],
               region_bgr["middle_30_60"], -1)
    cv2.circle(overlay, center, region_radii["central_0_30"],
               region_bgr["central_0_30"], -1)
    region_base = cv2.addWeighted(region_base, 0.78, overlay, 0.22, 0)
    for angle, color, label in (
        (30, region_bgr["central_0_30"], "center 0-30"),
        (60, region_bgr["middle_30_60"], "middle 30-60"),
        (80, region_bgr["peripheral_60_80"], "peripheral 60-80"),
    ):
        radius_px = int(round((angle / 90.0) * half_px * scale))
        cv2.circle(region_base, center, radius_px, color, 2, cv2.LINE_AA)
        cv2.putText(
            region_base,
            label,
            (center[0] + radius_px + 8, center[1]),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )

    def on_mouse(event: int, x: int, y: int, flags: int, param: Any) -> None:
        del flags, param
        if event == cv2.EVENT_LBUTTONDOWN and len(clicks) < expected:
            clicks.append((float(x) / scale, float(y) / scale))
        elif event == cv2.EVENT_RBUTTONDOWN and clicks:
            clicks.pop()

    def preview_counts(current: list[tuple[float, float]]) -> dict[str, int]:
        preview = snap_manual_points(frame_matches, current, radius=radius)
        counts = {region: 0 for region in REGIONS}
        for row in preview:
            counts[row["region"]] += 1
        return counts

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, on_mouse)
    try:
        while True:
            canvas = region_base.copy()
            status = f"Left: add ({len(clicks)}/{expected})  Right: undo  N: done  ESC: cancel"
            if len(clicks) == expected:
                try:
                    counts = preview_counts(clicks)
                    summary = (
                        f"{counts['central_0_30']}/{counts['middle_30_60']}/"
                        f"{counts['peripheral_60_80']}"
                    )
                    if all(counts[region] >= POINTS_PER_REGION for region in REGIONS):
                        status += f"  |  C/M/P = {summary} OK, press N"
                    else:
                        status += f"  |  C/M/P = {summary}, need 2/2/2"
                except RuntimeError as exc:
                    status += "  |  " + str(exc)
            cv2.putText(
                canvas,
                status,
                (18, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            for index, (u, v) in enumerate(clicks):
                marker_center = (int(round(u * scale)), int(round(v * scale)))
                color = LEGACY_POINT_COLORS[index % len(LEGACY_POINT_COLORS)]
                cv2.circle(canvas, marker_center, 9, color, 2, cv2.LINE_AA)
                cv2.putText(
                    canvas,
                    str(index + 1),
                    (marker_center[0] + 12, marker_center[1] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    color,
                    2,
                    cv2.LINE_AA,
                )
            cv2.imshow(window, canvas)
            key = cv2.waitKey(30) & 0xFF
            if key in (13, ord("n")):
                if len(clicks) == expected:
                    try:
                        counts = preview_counts(clicks)
                        if all(counts[region] >= POINTS_PER_REGION for region in REGIONS):
                            break
                    except RuntimeError:
                        pass
            if key == 27:
                clicks.clear()
                break
    finally:
        cv2.destroyWindow(window)
        cv2.waitKey(1)
    if len(clicks) != expected:
        fail(f"manual selection finished with {len(clicks)} points; expected {expected}")
    return snap_manual_points(frame_matches, clicks, radius=radius)


def epipolar_locus(system: Any, left_pixel: np.ndarray, samples: int = 1080) -> list[np.ndarray]:
    rays_left, valid = D.ds_unproject(system.left, left_pixel.reshape(1, 2))
    if not bool(valid[0]):
        return []
    predicted = system.rotation_cam1_cam0 @ rays_left[0]
    first = normalized(predicted)
    normal = normalized(np.cross(system.translation_cam1_cam0, predicted))
    if first is None or normal is None:
        return []
    second = normalized(np.cross(normal, first))
    if second is None:
        return []
    phase = np.linspace(-math.pi, math.pi, samples, endpoint=True)
    rays = np.cos(phase)[:, None] * first[None, :] + np.sin(phase)[:, None] * second[None, :]
    pixels, valid_project = D.ds_project(system.right, rays)
    pieces: list[np.ndarray] = []
    current: list[np.ndarray] = []
    prior: np.ndarray | None = None
    for pixel, is_valid in zip(pixels, valid_project):
        if not is_valid or not np.isfinite(pixel).all() or (prior is not None and np.linalg.norm(pixel - prior) > 35.0):
            if len(current) >= 2:
                pieces.append(np.asarray(current))
            current, prior = [], None
            continue
        current.append(pixel)
        prior = pixel
    if len(current) >= 2:
        pieces.append(np.asarray(current))
    return pieces


def crop_box(center: tuple[float, float], shape: tuple[int, int], size: int = 760) -> tuple[int, int, int, int]:
    height, width = shape[:2]
    half = size // 2
    x0 = min(max(0, int(round(center[0])) - half), max(0, width - size))
    y0 = min(max(0, int(round(center[1])) - half), max(0, height - size))
    return x0, y0, min(width, x0 + size), min(height, y0 + size)



def crop_box_rect(center: tuple[float, float], shape: tuple[int, int],
                  width: int, height: int) -> tuple[int, int, int, int]:
    """Return a fixed-aspect crop centered on a frozen correspondence."""
    image_height, image_width = shape[:2]
    crop_width = min(width, image_width)
    crop_height = min(height, image_height)
    x0 = min(max(0, int(round(center[0])) - crop_width // 2), image_width - crop_width)
    y0 = min(max(0, int(round(center[1])) - crop_height // 2), image_height - crop_height)
    return x0, y0, x0 + crop_width, y0 + crop_height


def point_label(index: int) -> str:
    if index >= len(POINT_IDS):
        fail(f"expected at most {len(POINT_IDS)} selected matches, got index {index}")
    return POINT_IDS[index]


def prepare_background(image: np.ndarray) -> np.ndarray:
    """Apply one fixed mild gamma adjustment to every source-image panel."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    enhanced = np.clip(0.96 * np.power(gray, 0.82) + 0.025, 0.0, 1.0)
    return cv2.cvtColor(np.round(enhanced * 255.0).astype(np.uint8), cv2.COLOR_GRAY2BGR)


def image_data_uri(image: np.ndarray) -> str:
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 94])
    if not ok:
        fail("cannot JPEG-encode qualitative background")
    return "data:image/jpeg;base64," + base64.b64encode(encoded.tobytes()).decode("ascii")


def export_svg(svg_path: Path, png_path: Path, pdf_path: Path | None) -> None:
    """Export through librsvg when available, with a CairoSVG fallback."""
    converter = shutil.which("rsvg-convert")
    if converter is not None:
        subprocess.run([converter, "--unlimited", "--format", "png",
                        "--width", str(CANVAS_WIDTH),
                        "--height", str(CANVAS_HEIGHT), "--output", str(png_path), str(svg_path)],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        if pdf_path is not None:
            subprocess.run([converter, "--unlimited", "--format", "pdf",
                            "--output", str(pdf_path), str(svg_path)],
                           check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        return
    try:
        import cairosvg
    except ImportError as exc:
        fail("SVG export requires either rsvg-convert or the cairosvg Python package")
        raise AssertionError from exc
    cairosvg.svg2png(url=str(svg_path), write_to=str(png_path),
                     output_width=CANVAS_WIDTH, output_height=CANVAS_HEIGHT)
    if pdf_path is not None:
        cairosvg.svg2pdf(url=str(svg_path), write_to=str(pdf_path))

def svg_polyline(points: np.ndarray, style: str) -> str:
    if len(points) < 2:
        return ""
    encoded = " ".join(f"{point[0]:.2f},{point[1]:.2f}" for point in points)
    return f'<polyline points="{encoded}" {style}/>'


def closest_locus_geometry(pieces: list[np.ndarray], observed: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    if not pieces:
        return None
    best_piece: np.ndarray | None = None
    best_index = -1
    best_distance = math.inf
    for piece in pieces:
        distances = np.linalg.norm(piece - observed[None, :], axis=1)
        index = int(np.argmin(distances))
        distance = float(distances[index])
        if distance < best_distance:
            best_piece, best_index, best_distance = piece, index, distance
    if best_piece is None:
        return None
    start, end = max(0, best_index - 16), min(len(best_piece), best_index + 17)
    return best_piece[best_index], best_piece[start:end]


def load_epipolar_errors(experiment_dir: Path) -> dict[tuple[str, int, int], float]:
    errors: dict[tuple[str, int, int], float] = {}
    for row in read_csv(experiment_dir / "per_match_epipolar_errors.csv"):
        value = float(row["epipolar_angular_error_deg"])
        if math.isfinite(value):
            errors[(row["method"], int(row["frame_id"]), int(row["match_rank"]))] = value
    return errors


def write_selected_epipolar_values(output: Path, frame_id: int,
                                   selected: list[dict[str, Any]],
                                   errors: dict[tuple[str, int, int], float]) -> None:
    rows = []
    for index, row in enumerate(selected):
        values = {
            "point_id": point_label(index),
            "region": row["region"],
            "frame_id": frame_id,
            "match_rank": row["match_rank"],
            "u_left": row["u_left"],
            "v_left": row["v_left"],
            "u_right": row["u_right"],
            "v_right": row["v_right"],
        }
        for method in ("Kalibr", "Ours"):
            key = (method, frame_id, row["match_rank"])
            if key not in errors:
                fail(f"missing frozen epipolar error for {method}, frame {frame_id}, "
                     f"match {row['match_rank']}")
            values[f"{method.lower()}_epi_deg"] = errors[key]
        rows.append(values)
    path = output / "selected_match_epipolar_values.csv"
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    write_json(output / "selected_match_epipolar_values.json", {
        "frame_id": frame_id,
        "values": rows,
        "units": {"kalibr_epi_deg": "deg", "ours_epi_deg": "deg"},
    })


def svg_text(text: str, x: float, y: float, size: float, *, fill: str = "#FFFFFF",
             weight: str = "600", outline: bool = True, anchor: str = "start") -> str:
    stroke = ' stroke="#121212" stroke-width="6" paint-order="stroke"' if outline else ""
    return (f'<text x="{x:.2f}" y="{y:.2f}" font-size="{size:.2f}" font-weight="{weight}" '
            f'fill="{fill}" text-anchor="{anchor}"{stroke}>{html.escape(text)}</text>')


def panel_label(label: str, x: float, panel_y: float, panel_size: float) -> str:
    """Place the panel title outside the image so it never hides image content."""
    return svg_text(label, x + panel_size / 2.0, panel_y - 28.0, 46,
                    fill="#202020", weight="600", outline=False, anchor="middle")


def transform_points(points: np.ndarray, panel_x: float, panel_y: float, scale: float) -> np.ndarray:
    return points * scale + np.asarray([panel_x, panel_y])


def _box_overlap(first: tuple[float, float, float, float],
                 second: tuple[float, float, float, float], padding: float = 0.0) -> float:
    ax0, ay0, ax1, ay1 = first
    bx0, by0, bx1, by1 = second
    width = max(0.0, min(ax1, bx1) - max(ax0, bx0) + padding)
    height = max(0.0, min(ay1, by1) - max(ay0, by0) + padding)
    return width * height


def place_label_boxes(points: np.ndarray, panel_size: float) -> list[tuple[float, float, float, float]]:
    """Greedily place compact labels around markers without covering one another.

    Placement depends only on the frozen image coordinates. It is shared by the
    Kalibr and Ours panels and therefore cannot favor either method.
    """
    label_width = 72.0
    label_height = 48.0
    margin = 12.0
    offsets = (
        (24.0, -label_height - 18.0),
        (24.0, 18.0),
        (-label_width - 24.0, -label_height - 18.0),
        (-label_width - 24.0, 18.0),
        (-label_width / 2.0, -label_height - 30.0),
        (-label_width / 2.0, 30.0),
    )
    placed: list[tuple[float, float, float, float]] = []
    for point in points:
        best_box: tuple[float, float, float, float] | None = None
        best_score = math.inf
        for dx, dy in offsets:
            x0 = float(point[0] + dx)
            y0 = float(point[1] + dy)
            x1, y1 = x0 + label_width, y0 + label_height
            box = (x0, y0, x1, y1)
            score = 0.0
            score += max(0.0, margin - x0) * 2000.0
            score += max(0.0, margin - y0) * 2000.0
            score += max(0.0, x1 - (panel_size - margin)) * 2000.0
            score += max(0.0, y1 - (panel_size - margin)) * 2000.0
            score += sum(_box_overlap(box, prior, padding=8.0) * 100.0 for prior in placed)
            for other in points:
                nearest_x = min(max(float(other[0]), x0), x1)
                nearest_y = min(max(float(other[1]), y0), y1)
                distance = math.hypot(float(other[0]) - nearest_x, float(other[1]) - nearest_y)
                if distance < 24.0:
                    score += (24.0 - distance) * 300.0
            score += math.hypot(dx, dy)
            if score < best_score:
                best_score, best_box = score, box
        if best_box is None:
            fail("could not place qualitative point label")
        placed.append(best_box)
    return placed


def draw_label_chip(label: str, point: np.ndarray,
                    box: tuple[float, float, float, float], color: str,
                    panel_x: float, panel_y: float) -> list[str]:
    x0, y0, x1, y1 = box
    box_center = np.asarray([(x0 + x1) / 2.0, (y0 + y1) / 2.0])
    direction = box_center - point
    norm = float(np.linalg.norm(direction))
    if norm > 1e-9:
        direction /= norm
    start = point + direction * 15.0
    end = box_center - direction * 26.0
    start += np.asarray([panel_x, panel_y])
    end += np.asarray([panel_x, panel_y])
    ax0, ay0, ax1, ay1 = (x0 + panel_x, y0 + panel_y, x1 + panel_x, y1 + panel_y)
    return [
        f'<line x1="{start[0]:.2f}" y1="{start[1]:.2f}" x2="{end[0]:.2f}" y2="{end[1]:.2f}" '
        f'stroke="{color}" stroke-width="3" stroke-linecap="round"/>',
        f'<rect x="{ax0:.2f}" y="{ay0:.2f}" width="{ax1 - ax0:.2f}" height="{ay1 - ay0:.2f}" '
        f'rx="9" fill="#FFFFFF" fill-opacity="0.90" stroke="{color}" stroke-width="3"/>',
        svg_text(label, (ax0 + ax1) / 2.0, ay0 + 35.0, 31, fill="#171717",
                 weight="700", outline=False, anchor="middle"),
    ]


def draw_source_points(selected: list[dict[str, Any]], panel_x: float, panel_y: float,
                       scale: float, label_boxes: list[tuple[float, float, float, float]],
                       *, include_labels: bool = True) -> list[str]:
    elements: list[str] = []
    relative_points = np.asarray([[row["u_left"] * scale, row["v_left"] * scale]
                                  for row in selected], dtype=float)
    for index, point in enumerate(relative_points):
        x, y = point + np.asarray([panel_x, panel_y])
        color = POINT_COLORS[index]
        elements.append(
            f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{SOURCE_MARKER_RADIUS:.1f}" '
            f'fill="{color}"/>'
        )
    if include_labels:
        for index, point in enumerate(relative_points):
            elements.extend(draw_label_chip(point_label(index), point, label_boxes[index],
                                            POINT_COLORS[index], panel_x, panel_y))
    return elements


def draw_right_match(index: int, row: dict[str, Any], pieces: list[np.ndarray],
                     panel_x: float, panel_y: float, scale: float) -> list[str]:
    """Draw one match with a quiet global locus and an emphasized local residual."""
    color = POINT_COLORS[index]
    elements: list[str] = []
    for piece in pieces:
        transformed = transform_points(piece, panel_x, panel_y, scale)
        elements.append(svg_polyline(
            transformed,
            f'fill="none" stroke="{color}" stroke-width="{GLOBAL_LOCUS_WIDTH:.1f}" '
            f'stroke-opacity="{GLOBAL_LOCUS_OPACITY:.2f}" '
            'stroke-linecap="round" stroke-linejoin="round"'))
    observed_original = np.asarray([row["u_right"], row["v_right"]])
    observed = transform_points(observed_original[None, :], panel_x, panel_y, scale)[0]
    geometry = closest_locus_geometry(pieces, observed_original)
    if geometry is not None:
        closest_original, local_piece = geometry
        closest = transform_points(closest_original[None, :], panel_x, panel_y, scale)[0]
        local = transform_points(local_piece, panel_x, panel_y, scale)
        elements.append(svg_polyline(
            local,
            f'fill="none" stroke="{color}" stroke-width="{LOCAL_LOCUS_WIDTH:.1f}" '
            'stroke-opacity="1.0" stroke-linecap="round" stroke-linejoin="round"'))
        elements.extend((
            f'<line x1="{observed[0]:.2f}" y1="{observed[1]:.2f}" x2="{closest[0]:.2f}" y2="{closest[1]:.2f}" '
            f'stroke="{color}" stroke-width="{RESIDUAL_WIDTH:.1f}" stroke-linecap="round"/>',
            f'<circle cx="{closest[0]:.2f}" cy="{closest[1]:.2f}" r="{CLOSEST_POINT_RADIUS:.1f}" '
            f'fill="none" stroke="{color}" stroke-width="{CLOSEST_POINT_WIDTH:.1f}"/>',
        ))
    cross_size = OBSERVED_CROSS_HALF_SIZE
    elements.extend((
        f'<line x1="{observed[0] - cross_size:.2f}" y1="{observed[1] - cross_size:.2f}" '
        f'x2="{observed[0] + cross_size:.2f}" y2="{observed[1] + cross_size:.2f}" '
        f'stroke="{color}" stroke-width="{OBSERVED_CROSS_WIDTH:.1f}" stroke-linecap="round"/>',
        f'<line x1="{observed[0] - cross_size:.2f}" y1="{observed[1] + cross_size:.2f}" '
        f'x2="{observed[0] + cross_size:.2f}" y2="{observed[1] - cross_size:.2f}" '
        f'stroke="{color}" stroke-width="{OBSERVED_CROSS_WIDTH:.1f}" stroke-linecap="round"/>',
    ))
    return elements


def draw_right_labels(selected: list[dict[str, Any]], panel_x: float, panel_y: float,
                      scale: float, label_boxes: list[tuple[float, float, float, float]]) -> list[str]:
    relative_points = np.asarray([[row["u_right"] * scale, row["v_right"] * scale]
                                  for row in selected], dtype=float)
    elements: list[str] = []
    for index, point in enumerate(relative_points):
        elements.extend(draw_label_chip(point_label(index), point, label_boxes[index],
                                        POINT_COLORS[index], panel_x, panel_y))
    return elements


def draw_inset(index: int, row: dict[str, Any], system: Any, error_deg: float,
               background_uri: str, image_width: int, image_height: int,
               crop: tuple[int, int, int, int], box: tuple[float, float, float, float],
               *, include_text: bool = True) -> str:
    """Draw a peripheral crop in the dedicated strip below the main panel."""
    x0, y0, x1, y1 = crop
    box_x, box_y, box_width, box_height = box
    pieces = epipolar_locus(system, np.asarray([row["u_left"], row["v_left"]]))
    content = [
        f'<svg x="{box_x:.2f}" y="{box_y:.2f}" width="{box_width:.2f}" height="{box_height:.2f}" '
        f'viewBox="{x0} {y0} {x1 - x0} {y1 - y0}" preserveAspectRatio="xMidYMid meet">',
        f'<image href="{background_uri}" x="0" y="0" width="{image_width}" height="{image_height}"/>',
    ]
    observed = np.asarray([row["u_right"], row["v_right"]])
    color = POINT_COLORS[index]
    geometry = closest_locus_geometry(pieces, observed)
    if geometry is not None:
        closest, local = geometry
        content.extend((
            svg_polyline(local, f'fill="none" stroke="{color}" '
                                f'stroke-width="{INSET_LOCAL_LOCUS_WIDTH:.1f}" '
                                'stroke-linecap="round" stroke-linejoin="round"'),
            f'<line x1="{observed[0]:.2f}" y1="{observed[1]:.2f}" x2="{closest[0]:.2f}" y2="{closest[1]:.2f}" '
            f'stroke="{color}" stroke-width="{INSET_RESIDUAL_WIDTH:.1f}" stroke-linecap="round"/>',
            f'<circle cx="{closest[0]:.2f}" cy="{closest[1]:.2f}" r="{INSET_CLOSEST_POINT_RADIUS:.1f}" '
            f'fill="none" stroke="{color}" stroke-width="{INSET_CLOSEST_POINT_WIDTH:.1f}"/>',
        ))
    cross_size = INSET_OBSERVED_CROSS_HALF_SIZE
    content.extend((
        f'<line x1="{observed[0] - cross_size:.2f}" y1="{observed[1] - cross_size:.2f}" '
        f'x2="{observed[0] + cross_size:.2f}" y2="{observed[1] + cross_size:.2f}" '
        f'stroke="{color}" stroke-width="{INSET_OBSERVED_CROSS_WIDTH:.1f}" stroke-linecap="round"/>',
        f'<line x1="{observed[0] - cross_size:.2f}" y1="{observed[1] + cross_size:.2f}" '
        f'x2="{observed[0] + cross_size:.2f}" y2="{observed[1] - cross_size:.2f}" '
        f'stroke="{color}" stroke-width="{INSET_OBSERVED_CROSS_WIDTH:.1f}" stroke-linecap="round"/>',
    ))
    if include_text:
        label = f"{point_label(index)}   e_epi = {error_deg:.2f} deg"
        content.append(
            f'<rect x="{x0 + 18}" y="{y0 + 18}" width="430" height="68" rx="10" '
            'fill="#111111" fill-opacity="0.74"/>'
        )
        content.append(
            svg_text(label, x0 + 38, y0 + 66, 43, fill="#FFFFFF", weight="600", outline=False)
        )
    content.append(
        f'<rect x="{x0 + 2}" y="{y0 + 2}" width="{x1 - x0 - 4}" height="{y1 - y0 - 4}" '
        'fill="none" stroke="#D8D8D8" stroke-width="5"/>'
    )
    content.append('</svg>')
    return "".join(content)


def draw_match_key(panel_x: float, strip_y: float, panel_size: float) -> list[str]:
    groups = (
        ("Center", range(0, POINTS_PER_REGION)),
        ("Middle", range(POINTS_PER_REGION, 2 * POINTS_PER_REGION)),
        ("Peripheral", range(2 * POINTS_PER_REGION, 3 * POINTS_PER_REGION)),
    )
    elements = [svg_text("Fixed correspondences", panel_x + panel_size / 2.0,
                         strip_y + 42, 34, fill="#282828", weight="600",
                         outline=False, anchor="middle")]
    group_width = panel_size / 3.0
    for group_index, (name, indices) in enumerate(groups):
        center_x = panel_x + (group_index + 0.5) * group_width
        elements.append(svg_text(name, center_x, strip_y + 92, 27, fill="#5B5B5B",
                                 weight="500", outline=False, anchor="middle"))
        offsets = tuple(
            (index - (POINTS_PER_REGION - 1) / 2.0) * 48.0
            for index in range(POINTS_PER_REGION)
        )
        for offset, point_index in zip(offsets, indices):
            x = center_x + offset
            y = strip_y + 146
            color = POINT_COLORS[point_index]
            elements.extend((
                f'<circle cx="{x:.2f}" cy="{y:.2f}" r="9" fill="{color}"/>',
                svg_text(point_label(point_index), x, y + 42, 25, fill="#303030",
                         weight="600", outline=False, anchor="middle"),
            ))
    return elements

def legacy_resize_for_figure(image: np.ndarray, max_width: int) -> tuple[np.ndarray, float]:
    scale = min(1.0, float(max_width) / image.shape[1])
    return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA), scale


def legacy_draw_cross(image: np.ndarray, point: tuple[int, int], color: tuple[int, int, int], size: int,
                      thickness: int = 2) -> None:
    x, y = point
    cv2.line(image, (x - size, y - size), (x + size, y + size), color, thickness, cv2.LINE_AA)
    cv2.line(image, (x - size, y + size), (x + size, y - size), color, thickness, cv2.LINE_AA)


def legacy_draw_right_panel(image: np.ndarray, system: Any, selected: list[dict[str, Any]],
                            roi: tuple[int, int, int, int], scale: float) -> np.ndarray:
    canvas = image.copy()
    for index, row in enumerate(selected):
        color = LEGACY_POINT_COLORS[index]
        left_pixel = np.asarray([row["u_left"], row["v_left"]])
        right_pixel = np.asarray([row["u_right"], row["v_right"]])
        pieces = epipolar_locus(system, left_pixel)
        for piece in pieces:
            scaled = np.round(piece * scale).astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(canvas, [scaled], False, color, 2, cv2.LINE_AA)
        right = (int(round(right_pixel[0] * scale)), int(round(right_pixel[1] * scale)))
        geometry = closest_locus_geometry(pieces, right_pixel)
        if geometry is not None:
            closest, _ = geometry
            predicted = (int(round(closest[0] * scale)), int(round(closest[1] * scale)))
            cv2.line(canvas, right, predicted, color, 2, cv2.LINE_AA)
        # Reuse the project's ordinary OpenCV point convention: a plain tilted
        # cross for the observation and a plain open circle for the locus.
        cv2.drawMarker(canvas, right, color, cv2.MARKER_TILTED_CROSS, 16, 2, cv2.LINE_AA)
        if geometry is not None:
            cv2.circle(canvas, predicted, 9, color, 2, cv2.LINE_AA)
    del roi
    return canvas


def legacy_draw_left_panel(image: np.ndarray, selected: list[dict[str, Any]], scale: float) -> np.ndarray:
    canvas = image.copy()
    for index, row in enumerate(selected):
        color = LEGACY_POINT_COLORS[index]
        point = (int(round(row["u_left"] * scale)), int(round(row["v_left"] * scale)))
        cv2.circle(canvas, point, 6, color, cv2.FILLED, cv2.LINE_AA)
    return canvas


def legacy_labeled_panel(image: np.ndarray, title: str) -> np.ndarray:
    header = np.full((64, image.shape[1], 3), 255, dtype=np.uint8)
    cv2.putText(header, title, (18, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.82, (25, 25, 25), 2, cv2.LINE_AA)
    return np.vstack((header, image))


def legacy_footer(width: int, selected: list[dict[str, Any]]) -> np.ndarray:
    footer = np.full((250, width, 3), 255, dtype=np.uint8)
    cv2.putText(footer, "Same frozen correspondences; curves: predicted epipolar loci; x: observed match; dot/segment: closest locus point and residual.",
                (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (35, 35, 35), 1, cv2.LINE_AA)
    for index, (color, row) in enumerate(zip(LEGACY_POINT_COLORS, selected)):
        x = 18 + (index % 3) * 360
        y = 76 + (index // 3) * 38
        cv2.line(footer, (x, y), (x + 30, y), color, 3, cv2.LINE_AA)
        label = REGION_LABELS.get(row["region"], row["region"])
        cv2.putText(footer, f"{index + 1}: {label}", (x + 40, y + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.54, (35, 35, 35), 1, cv2.LINE_AA)
    return footer


def render_loci(output: Path, experiment_dir: Path, pair: Any, systems: Iterable[Any],
                selected: list[dict[str, Any]]) -> list[Path]:
    """Render the original two-panel-per-method layout without annotations."""
    left = cv2.imread(str(pair.left_path), cv2.IMREAD_COLOR)
    right = cv2.imread(str(pair.right_path), cv2.IMREAD_COLOR)
    if left is None or right is None:
        fail(f"cannot load selected image pair {pair.frame_id}")
    left, scale = legacy_resize_for_figure(left, 1400)
    right, right_scale = legacy_resize_for_figure(right, 1400)
    if abs(scale - right_scale) > 1e-9:
        fail("left/right display scale differs; cannot preserve frozen correspondence geometry")
    errors = load_epipolar_errors(experiment_dir)
    write_selected_epipolar_values(output, pair.frame_id, selected, errors)
    rows: list[np.ndarray] = []
    paths: list[Path] = []
    for system in systems:
        left_panel = legacy_draw_left_panel(left, selected, scale)
        right_panel = legacy_draw_right_panel(right, system, selected, (0, 0, 0, 0), scale)
        row = np.hstack((left_panel, np.full((left_panel.shape[0], 10, 3), 235, dtype=np.uint8), right_panel))
        rows.append(row)
        path = output / f"{system.name.lower()}_epipolar_loci_qualitative.png"
        cv2.imwrite(str(path), row)
        paths.append(path)
    if len(rows) == 2:
        combined = np.vstack((rows[0], np.full((10, rows[0].shape[1], 3), 235, dtype=np.uint8), rows[1]))
        combined_path = output / "peripheral_epipolar_loci_qualitative.png"
        cv2.imwrite(str(combined_path), combined)
        paths.append(combined_path)
    return paths


def render_three_column_loci(output: Path, experiment_dir: Path, pair: Any,
                              systems: Iterable[Any], selected: list[dict[str, Any]],
                              *, skip_pdf: bool = False, include_text: bool = True,
                              file_suffix: str = "", panel_only: bool = False,
                              ours_suffix: str | None = None) -> list[Path]:
    left_raw = cv2.imread(str(pair.left_path), cv2.IMREAD_COLOR)
    right_raw = cv2.imread(str(pair.right_path), cv2.IMREAD_COLOR)
    if left_raw is None or right_raw is None:
        fail(f"cannot load selected image pair {pair.frame_id}")
    if left_raw.shape[:2] != right_raw.shape[:2]:
        fail("left/right image dimensions differ; cannot enforce identical display scales")
    expected_count = 3 * POINTS_PER_REGION
    if len(selected) != expected_count:
        fail(f"publication layout requires exactly {expected_count} frozen matches, got {len(selected)}")
    systems = list(systems)
    if [system.name for system in systems] != ["Kalibr", "Ours"]:
        fail("qualitative figure expects exactly Kalibr and Ours in that order")

    panel_size = (CANVAS_WIDTH - 2 * PANEL_MARGIN - 2 * PANEL_GAP) / 3.0
    panel_y = PANEL_TITLE_HEIGHT
    strip_y = panel_y + panel_size + INSET_GAP
    panel_xs = [PANEL_MARGIN + index * (panel_size + PANEL_GAP) for index in range(3)]
    scale = panel_size / float(left_raw.shape[1])
    left_uri = image_data_uri(prepare_background(left_raw))
    right_uri = image_data_uri(prepare_background(right_raw))
    errors = load_epipolar_errors(experiment_dir)
    write_selected_epipolar_values(output, pair.frame_id, selected, errors)

    peripheral_indices = [index for index, row in enumerate(selected)
                          if row["region"] == "peripheral_60_80"]
    expected_peripheral = list(range(2 * POINTS_PER_REGION, 3 * POINTS_PER_REGION))
    if peripheral_indices != expected_peripheral:
        fail(f"expected fixed peripheral indices {expected_peripheral}, got {peripheral_indices}")
    crops = {
        index: crop_box_rect((selected[index]["u_right"], selected[index]["v_right"]),
                             right_raw.shape, width=820, height=420)
        for index in peripheral_indices
    }
    inset_gap = 12.0
    inset_width = (panel_size - (len(peripheral_indices) - 1) * inset_gap) / len(peripheral_indices)
    inset_boxes = {
        index: (float(offset) * (inset_width + inset_gap), 0.0, inset_width, INSET_STRIP_HEIGHT)
        for offset, index in enumerate(peripheral_indices)
    }

    source_relative = np.asarray([[row["u_left"] * scale, row["v_left"] * scale]
                                  for row in selected], dtype=float)
    right_relative = np.asarray([[row["u_right"] * scale, row["v_right"] * scale]
                                 for row in selected], dtype=float)
    source_label_boxes = place_label_boxes(source_relative, panel_size)
    right_label_boxes = place_label_boxes(right_relative, panel_size)

    physical_height = 7.2 * CANVAS_HEIGHT / CANVAS_WIDTH
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="7.2in" height="{physical_height:.3f}in" '
        f'viewBox="0 0 {CANVAS_WIDTH} {CANVAS_HEIGHT}">',
        '<style>text { font-family: Helvetica, Arial, sans-serif; }</style>',
        '<rect width="100%" height="100%" fill="#FFFFFF"/>',
        '<defs>',
    ]
    for index, x in enumerate(panel_xs):
        svg.append(f'<clipPath id="panel-{index}"><rect x="{x:.2f}" y="{panel_y:.2f}" '
                   f'width="{panel_size:.2f}" height="{panel_size:.2f}"/></clipPath>')
    svg.append('</defs>')

    for index, (x, uri) in enumerate(zip(panel_xs, (left_uri, right_uri, right_uri))):
        svg.append(f'<g clip-path="url(#panel-{index})"><image href="{uri}" x="{x:.2f}" y="{panel_y:.2f}" '
                   f'width="{panel_size:.2f}" height="{panel_size:.2f}"/></g>')
        svg.append(f'<rect x="{x:.2f}" y="{panel_y:.2f}" width="{panel_size:.2f}" height="{panel_size:.2f}" '
                   'fill="none" stroke="#D7D7D7" stroke-width="2"/>')

    svg.append('<g clip-path="url(#panel-0)">')
    svg.extend(draw_source_points(selected, panel_xs[0], panel_y, scale, source_label_boxes,
                                  include_labels=include_text))
    svg.append('</g>')

    for panel_index, system in enumerate(systems, start=1):
        svg.append(f'<g clip-path="url(#panel-{panel_index})">')
        for index, row in enumerate(selected):
            pieces = epipolar_locus(system, np.asarray([row["u_left"], row["v_left"]]))
            svg.extend(draw_right_match(index, row, pieces,
                                        panel_xs[panel_index], panel_y, scale))
        if include_text:
            svg.extend(draw_right_labels(selected, panel_xs[panel_index], panel_y,
                                         scale, right_label_boxes))
        svg.append('</g>')

        for index in peripheral_indices:
            error = errors.get((system.name, pair.frame_id, selected[index]["match_rank"]))
            if error is None:
                fail(f"missing frozen epipolar error for {system.name}, frame {pair.frame_id}, "
                     f"match {selected[index]['match_rank']}")
            relative = inset_boxes[index]
            box = (panel_xs[panel_index] + relative[0], strip_y + relative[1],
                   relative[2], relative[3])
            svg.append(draw_inset(index, selected[index], system, error, right_uri,
                                  right_raw.shape[1], right_raw.shape[0], crops[index], box,
                                  include_text=include_text))

    if include_text:
        svg.extend(draw_match_key(panel_xs[0], strip_y, panel_size))
        for label, x in zip(("(a) Source points", "(b) Kalibr", "(c) Ours"), panel_xs):
            svg.append(panel_label(label, x, panel_y, panel_size))
    svg.append('</svg>')

    svg_path = output / f"peripheral_epipolar_loci_qualitative{file_suffix}.svg"
    svg_path.write_text("\n".join(svg), encoding="utf-8")
    png_path = svg_path.with_suffix(".png")
    pdf_path = svg_path.with_suffix(".pdf")
    export_svg(svg_path, png_path, None if skip_pdf else pdf_path)
    paths = [svg_path, png_path]
    if not skip_pdf:
        paths.append(pdf_path)

    # Export the Ours panel as a standalone preview while preserving the exact
    # pixels, labels, main panel, and inset strip from the three-column figure.
    rendered = cv2.imread(str(png_path), cv2.IMREAD_COLOR)
    if rendered is None:
        fail(f"cannot reload rendered PNG for standalone Ours export: {png_path}")
    ours_x0 = int(round(panel_xs[2]))
    ours_x1 = int(round(panel_xs[2] + panel_size))
    ours_name_suffix = ours_suffix if ours_suffix is not None else file_suffix
    ours_png = output / f"ours_epipolar_loci_qualitative{ours_name_suffix}.png"
    if panel_only:
        ours_y0 = int(round(panel_y))
        ours_y1 = int(round(panel_y + panel_size))
        cv2.imwrite(str(ours_png), rendered[ours_y0:ours_y1, ours_x0:ours_x1])
    else:
        cv2.imwrite(str(ours_png), rendered[:, ours_x0:ours_x1])
    paths.append(ours_png)

    exports = {"svg": str(svg_path), "png": str(png_path), "ours_png": str(ours_png)}
    ours_png_crop = "main panel only (no peripheral insets)" if panel_only else "full Ours column with peripheral insets"
    if not skip_pdf:
        exports["pdf"] = str(pdf_path)
    write_json(output / f"peripheral_epipolar_loci_render_manifest{file_suffix}.json", {
        "layout": "three-column main panels with non-overlapping peripheral inset strip",
        "text_annotations": "included" if include_text else "none",
        "ours_png_crop": ours_png_crop,
        "canvas_px": [CANVAS_WIDTH, CANVAS_HEIGHT],
        "physical_size_in": [7.2, physical_height],
        "font_family": "Helvetica, Arial, sans-serif",
        "source_background": {
            "grayscale": True,
            "gamma": 0.82,
            "gain": 0.96,
            "offset": 0.025,
            "shared_for_kalibr_and_ours": True,
        },
        "visual_hierarchy": {
            "complete_loci": "thin, low-opacity colored curves sized for final-page reduction",
            "local_locus": "thin colored segment near the closest point",
            "labels": ("greedy non-overlapping chips with leader lines" if include_text else "none"),
            "peripheral_insets": "dedicated strip below the main panels",
        },
        "comparison_invariants": {
            "right_background_image": str(pair.right_path),
            "right_display_scale": scale,
            "fixed_matches": [point_label(index) for index in range(len(selected))],
            "right_image_shared_by_kalibr_and_ours": True,
            "right_label_positions_shared_by_kalibr_and_ours": True,
            "inset_crops_shared_by_kalibr_and_ours": True,
            "inset_boxes_shared_by_kalibr_and_ours": True,
        },
        "peripheral_insets": {
            point_label(index): {
                "crop_xyxy": list(crops[index]),
                "relative_strip_box_xywh": list(inset_boxes[index]),
            }
            for index in peripheral_indices
        },
        "exports": exports,
    })
    return paths

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--ours-bundle", type=Path, required=True)
    parser.add_argument("--kalibr-camchain", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--frame-id", type=int,
        help="use one fixed frozen frame for the qualitative rendering",
    )
    parser.add_argument(
        "--selection-mode",
        choices=("spatial_max", "random_spatial"),
        default="spatial_max",
        help="select maximum-coverage pairs or a seeded random spatially separated pair",
    )
    parser.add_argument(
        "--selection-seed", type=int, default=20260804,
        help="fixed seed used only with --selection-mode random_spatial",
    )
    parser.add_argument("--skip-pdf", action="store_true")
    parser.add_argument(
        "--publication-layout", action="store_true",
        help="opt in to the annotated three-column layout; the clean original layout is the default",
    )
    parser.add_argument(
        "--no-text", action="store_true",
        help="export the three-column layout and standalone Ours image without text annotations",
    )
    parser.add_argument(
        "--panel-only", action="store_true",
        help="crop the standalone Ours export to the main panel without peripheral insets",
    )
    parser.add_argument(
        "--manual-pick", action="store_true",
        help="interactively pick points on the left image and snap them to frozen matches",
    )
    parser.add_argument(
        "--manual-points", type=Path,
        help="JSON file with manually selected left-image points [[u, v], ...]",
    )
    parser.add_argument(
        "--snap-radius", type=float, default=150.0,
        help="maximum click-to-frozen-match distance in original pixels",
    )
    parser.add_argument("--legacy-layout", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    experiment_dir = args.experiment_dir.resolve()
    output = (args.output or experiment_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    bundle = args.ours_bundle.resolve()
    ours_left, ours_right = bundle / "left_intrinsics.yaml", bundle / "right_intrinsics.yaml"
    ours_extrinsic, manifest = bundle / "stereo_extrinsic.yaml", bundle / "stereo_bundle_manifest.json"
    D.verify_ours_bundle_manifest(manifest, ours_left, ours_right, ours_extrinsic)
    systems = [D.load_kalibr_system(args.kalibr_camchain.resolve()),
               D.load_ours_system(ours_left, ours_right, ours_extrinsic)]
    pairs, matches = load_frozen_inputs(experiment_dir)
    auto_frame_id, selected = choose_display_matches(
        matches, args.selection_mode, args.selection_seed
    )
    frame_id = auto_frame_id
    if args.frame_id is not None:
        if args.frame_id not in pairs:
            fail(f"requested frame {args.frame_id} is absent from the frozen frame manifest")
        frame_matches = [row for row in matches if row["frame_id"] == args.frame_id]
        if not frame_matches:
            fail(f"requested frame {args.frame_id} has no frozen matches")
        _, selected = choose_display_matches(
            frame_matches, args.selection_mode, args.selection_seed
        )
        frame_id = args.frame_id
    manual_selection = args.manual_pick or args.manual_points is not None
    if manual_selection:
        frame_matches = [row for row in matches if row["frame_id"] == frame_id]
        if args.manual_points is not None:
            data = json.loads(args.manual_points.read_text(encoding="utf-8"))
            raw_points = data.get("points", data) if isinstance(data, dict) else data
            selected = snap_manual_points(frame_matches, raw_points, args.snap_radius)
        else:
            selected = pick_manual_points(
                pairs[frame_id].left_path, frame_matches, radius=args.snap_radius,
            )
        region_order = {name: index for index, name in enumerate(REGIONS)}
        selected.sort(key=lambda row: (region_order[row["region"]], row["match_rank"]))
        region_counts = defaultdict(int)
        for row in selected:
            region_counts[row["region"]] += 1
        missing = [
            region for region in REGIONS
            if region_counts[region] < POINTS_PER_REGION
        ]
        if missing:
            fail("manual points did not cover every radial region with enough matches: "
                 f"missing {missing}")
    if frame_id not in pairs:
        fail(f"selected frozen frame {frame_id} is missing from the manifest")
    if len(selected) < 4:
        fail("not enough frozen matches for a qualitative locus figure")
    if args.no_text and not args.publication_layout:
        fail("--no-text requires --publication-layout")
    if args.panel_only and not args.publication_layout:
        fail("--panel-only requires --publication-layout")
    if args.publication_layout and not args.legacy_layout:
        file_suffix = "_notext" if args.no_text else ""
        ours_suffix = file_suffix + "_panel" if args.panel_only else (file_suffix or None)
        figures = render_three_column_loci(
            output, experiment_dir, pairs[frame_id], systems, selected,
            skip_pdf=args.skip_pdf or args.no_text,
            include_text=not args.no_text,
            file_suffix=file_suffix,
            panel_only=args.panel_only,
            ours_suffix=ours_suffix,
        )
    else:
        if args.no_text:
            fail("--no-text requires --publication-layout")
        figures = render_loci(output, experiment_dir, pairs[frame_id], systems, selected)
    write_json(output / "peripheral_epipolar_loci_selection.json", {
        "selection_protocol": (
            "manual left-image point selection snapped to frozen matches"
            if manual_selection else
            (
                "frozen-coordinate-only: two maximally separated image-plane samples per polar band"
                if args.selection_mode == "spatial_max" else
                "frozen-coordinate-only: seeded random samples from spatially separated pairs per polar band"
            )
        ),
        "point_refinement": None,
        "selection_mode": "manual" if manual_selection else args.selection_mode,
        "selection_seed": None if manual_selection else args.selection_seed,
        "frame_id": frame_id,
        "left_image": str(pairs[frame_id].left_path),
        "right_image": str(pairs[frame_id].right_path),
        "matches": selected,
        "systems": [system.name for system in systems],
    })
    for figure in figures:
        print(f"qualitative_figure={figure}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
