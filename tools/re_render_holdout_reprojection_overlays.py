#!/usr/bin/env python3
"""Re-render Stage5 holdout overlays and audit ours-vs-Kalibr point deltas.

The script consumes a completed Stage5 result directory. It never reruns
calibration or pose fitting: observed and projected pixels are loaded from the
frozen benchmark_holdout_points.csv artifact. Marker semantics are fixed:
outer observation = thin red cross, outer projection = thin open circle;
internal observations and projections = compact filled dots.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import cv2


METHODS = ("ours", "kalibr")
PROJECTED_COLORS = {
    "outer": (60, 220, 80),
    "internal": (40, 180, 255),
}
OBSERVED_COLOR = (0, 0, 255)
RESIDUAL_COLOR = (210, 210, 210)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, required=True,
                        help="Completed Stage5 result directory.")
    parser.add_argument("--image-dir", type=Path, required=True,
                        help="Directory containing the external holdout images.")
    parser.add_argument("--output-dir", type=Path,
                        help="Defaults to RESULT_DIR/holdout_reprojection_visualizations_cross_circle.")
    parser.add_argument("--top-k", type=int, default=30,
                        help="Number of worst ours frames and board observations to render.")
    parser.add_argument("--top-point-count", type=int, default=20,
                        help="Number of largest ours-vs-Kalibr prediction deltas to export separately.")
    parser.add_argument("--worst-frame-count", type=int, default=20,
                        help="Number of high-RMSE validation frames for focused point-residual audits.")
    parser.add_argument("--points-per-type", type=int, default=20,
                        help="Maximum outer and internal points to show per focused validation frame.")
    parser.add_argument("--frame-label", action="append", default=[],
                        help="Exact image stem to audit by board; may be repeated.")
    parser.add_argument("--specified-frame-points-per-type", type=int, default=20,
                        help="Maximum ranked outer/internal point crops per board for --frame-label; 0 exports all.")
    return parser.parse_args()


def point_key(row: dict[str, str]) -> tuple[str, ...]:
    return tuple(row[key] for key in (
        "frame_index", "board_id", "point_id", "point_type", "source_point_index"
    ))


def number(row: dict[str, str], key: str) -> float:
    return float(row[key])


def load_points(path: Path) -> dict[str, dict[tuple[str, ...], dict[str, str]]]:
    by_method: dict[str, dict[tuple[str, ...], dict[str, str]]] = {
        method: {} for method in METHODS
    }
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            method = row["method"]
            if method not in by_method:
                continue
            key = point_key(row)
            if key in by_method[method]:
                raise RuntimeError(f"Duplicate {method} point key: {key}")
            by_method[method][key] = row
    for method, rows in by_method.items():
        if not rows:
            raise RuntimeError(f"No {method} points found in {path}")
    return by_method


def image_by_label(image_dir: Path) -> dict[str, Path]:
    image_map: dict[str, Path] = {}
    for suffix in ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff"):
        for path in image_dir.glob(suffix):
            image_map[path.stem] = path
    if not image_map:
        raise RuntimeError(f"No images found in {image_dir}")
    return image_map


def as_point(row: dict[str, str], prefix: str) -> tuple[int, int]:
    return (int(round(number(row, f"{prefix}_x"))),
            int(round(number(row, f"{prefix}_y"))))


def draw_points(image, rows: list[dict[str, str]],
                projected_color_override: tuple[int, int, int] | None = None,
                residual_color_override: tuple[int, int, int] | None = None) -> tuple[int, int, int, float]:
    outer = 0
    internal = 0
    worst = 0.0
    for row in rows:
        point_type = row["point_type"]
        observed = as_point(row, "observed")
        predicted = as_point(row, "predicted")
        if point_type == "outer":
            outer += 1
        else:
            internal += 1
        worst = max(worst, number(row, "residual_norm"))
        cv2.line(image, observed, predicted,
                 residual_color_override or RESIDUAL_COLOR, 1, cv2.LINE_AA)
        projected_color = (projected_color_override or
                           PROJECTED_COLORS.get(point_type, (255, 255, 255)))
        # Keep model estimates in the foreground. Filled internal dots avoid
        # turning dense internal grids into overlapping cross/circle markers.
        if point_type == "outer":
            cv2.drawMarker(image, observed, OBSERVED_COLOR, cv2.MARKER_CROSS,
                           13, 1, cv2.LINE_AA)
            cv2.circle(image, predicted, 8, projected_color, 2, cv2.LINE_AA)
        else:
            cv2.circle(image, observed, 2, OBSERVED_COLOR, cv2.FILLED,
                       cv2.LINE_AA)
            cv2.circle(image, predicted, 3, projected_color, cv2.FILLED,
                       cv2.LINE_AA)
    return len(rows), outer, internal, worst


def add_banner(image, title: str, subtitle: str) -> None:
    height = 78
    cv2.rectangle(image, (0, 0), (image.shape[1], height), (18, 18, 18), cv2.FILLED)
    cv2.putText(image, title, (18, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                (238, 238, 238), 1, cv2.LINE_AA)
    cv2.putText(image, subtitle, (18, 55), cv2.FONT_HERSHEY_PLAIN, 1.1,
                (200, 200, 200), 1, cv2.LINE_AA)


def render_overlay(image_path: Path, rows: list[dict[str, str]], title: str,
                   rmse: float,
                   projected_color_override: tuple[int, int, int] | None = None,
                   projection_label: str = "green/orange") -> object:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        raise RuntimeError(f"Failed to load image: {image_path}")
    point_count, outer, internal, worst = draw_points(
        image, rows, projected_color_override, projected_color_override)
    add_banner(image, f"{title}  rmse={rmse:.6f}",
               f"outer: red cross/{projection_label} circle  |  internal: red/{projection_label} dots  |  "
               f"points={point_count} outer={outer} internal={internal} worst={worst:.6f}")
    return image


def labeled_compare(left, left_label: str, right, right_label: str,
                    show_labels: bool = True):
    height = max(left.shape[0], right.shape[0])
    width = left.shape[1] + right.shape[1] + 12
    canvas = cv2.copyMakeBorder(left, 0, height - left.shape[0], 0, 0,
                                cv2.BORDER_CONSTANT, value=(18, 18, 18))
    right_pad = cv2.copyMakeBorder(right, 0, height - right.shape[0], 0, 0,
                                   cv2.BORDER_CONSTANT, value=(18, 18, 18))
    comparison = cv2.copyMakeBorder(canvas, 0, 0, 0, 12, cv2.BORDER_CONSTANT,
                                    value=(18, 18, 18))
    comparison = cv2.hconcat([comparison, right_pad])
    if show_labels:
        cv2.putText(comparison, left_label, (18, 76), cv2.FONT_HERSHEY_PLAIN, 1.1,
                    (230, 230, 230), 1, cv2.LINE_AA)
        cv2.putText(comparison, right_label, (left.shape[1] + 30, 76),
                    cv2.FONT_HERSHEY_PLAIN, 1.1, (230, 230, 230), 1, cv2.LINE_AA)
    return comparison


def rms(rows: list[dict[str, str]]) -> float:
    if not rows:
        return float("nan")
    return math.sqrt(sum(number(row, "residual_norm") ** 2 for row in rows) / len(rows))


def clamp_crop(rows: list[dict[str, str]], image_shape: tuple[int, int, int], padding: int):
    xs = []
    ys = []
    for row in rows:
        xs.extend((number(row, "observed_x"), number(row, "predicted_x")))
        ys.extend((number(row, "observed_y"), number(row, "predicted_y")))
    x0 = max(0, int(math.floor(min(xs))) - padding)
    y0 = max(0, int(math.floor(min(ys))) - padding)
    x1 = min(image_shape[1], int(math.ceil(max(xs))) + padding)
    y1 = min(image_shape[0], int(math.ceil(max(ys))) + padding)
    return x0, y0, x1, y1


def point_context_crop_bounds(rows: list[dict[str, str]], image_shape: tuple[int, int, int],
                              width: int = 300, height: int = 200):
    """Return a fixed-size crop centered on a paired observation/projection."""
    xs = []
    ys = []
    for row in rows:
        xs.extend((number(row, "observed_x"), number(row, "predicted_x")))
        ys.extend((number(row, "observed_y"), number(row, "predicted_y")))
    center_x = int(round(sum(xs) / len(xs)))
    center_y = int(round(sum(ys) / len(ys)))
    image_height, image_width = image_shape[:2]
    crop_width = min(width, image_width)
    crop_height = min(height, image_height)
    x0 = min(max(0, center_x - crop_width // 2), image_width - crop_width)
    y0 = min(max(0, center_y - crop_height // 2), image_height - crop_height)
    return x0, y0, x0 + crop_width, y0 + crop_height


def add_top_banner(image, title: str, subtitle: str):
    """Add a banner above a crop instead of hiding its pixels."""
    banner_height = 62
    canvas = cv2.copyMakeBorder(image, banner_height, 0, 0, 0,
                                cv2.BORDER_CONSTANT, value=(18, 18, 18))
    cv2.putText(canvas, title, (14, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                (238, 238, 238), 1, cv2.LINE_AA)
    cv2.putText(canvas, subtitle, (14, 48), cv2.FONT_HERSHEY_PLAIN, 1.05,
                (205, 205, 205), 1, cv2.LINE_AA)
    return canvas


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise RuntimeError(f"Refusing to write empty CSV: {path}")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def comparison_rows(points: dict[str, dict[tuple[str, ...], dict[str, str]]]) -> list[dict[str, object]]:
    ours = points["ours"]
    kalibr = points["kalibr"]
    if ours.keys() != kalibr.keys():
        raise RuntimeError(
            f"Point-key mismatch: ours_only={len(ours.keys() - kalibr.keys())}, "
            f"kalibr_only={len(kalibr.keys() - ours.keys())}"
        )
    output: list[dict[str, object]] = []
    for key in ours:
        ours_row = ours[key]
        kalibr_row = kalibr[key]
        dx = number(ours_row, "predicted_x") - number(kalibr_row, "predicted_x")
        dy = number(ours_row, "predicted_y") - number(kalibr_row, "predicted_y")
        output.append({
            "frame_index": int(ours_row["frame_index"]),
            "frame_label": ours_row["frame_label"],
            "board_id": int(ours_row["board_id"]),
            "point_id": int(ours_row["point_id"]),
            "point_type": ours_row["point_type"],
            "source_point_index": int(ours_row["source_point_index"]),
            "observed_x": number(ours_row, "observed_x"),
            "observed_y": number(ours_row, "observed_y"),
            "ours_predicted_x": number(ours_row, "predicted_x"),
            "ours_predicted_y": number(ours_row, "predicted_y"),
            "kalibr_predicted_x": number(kalibr_row, "predicted_x"),
            "kalibr_predicted_y": number(kalibr_row, "predicted_y"),
            "prediction_delta_x_px": dx,
            "prediction_delta_y_px": dy,
            "prediction_delta_norm_px": math.hypot(dx, dy),
            "ours_residual_norm_px": number(ours_row, "residual_norm"),
            "kalibr_residual_norm_px": number(kalibr_row, "residual_norm"),
            "ours_minus_kalibr_residual_norm_px": (
                number(ours_row, "residual_norm") - number(kalibr_row, "residual_norm")
            ),
        })
    return sorted(output, key=lambda row: float(row["prediction_delta_norm_px"]), reverse=True)


def delta_point_key(delta: dict[str, object]) -> tuple[str, ...]:
    return (
        str(delta["frame_index"]),
        str(delta["board_id"]),
        str(delta["point_id"]),
        str(delta["point_type"]),
        str(delta["source_point_index"]),
    )


def export_prediction_delta_crops(
        output_dir: Path,
        selected: list[dict[str, object]],
        points: dict[str, dict[tuple[str, ...], dict[str, str]]],
        image_map: dict[str, Path],
        directory: Path,
        csv_name: str,
        selection_label: str) -> None:
    """Write paired local views for a selected set of prediction differences."""
    top_dir = output_dir / directory
    top_dir.mkdir(parents=True, exist_ok=True)
    write_csv(top_dir / csv_name, selected)
    overview = [
        "Each PNG shows one identical detected point rendered with the two models.",
        "left: Ours; right: Kalibr.",
        "red cross: detected observation; both methods use the same point-type colors.",
        "ranking: Euclidean distance between the Ours and Kalibr projected pixels.",
        f"selection: {selection_label}",
        f"count: {len(selected)}",
        "",
        "rank,frame_label,board_id,point_id,point_type,prediction_delta_px,ours_residual_px,kalibr_residual_px",
    ]
    for rank, delta in enumerate(selected, start=1):
        key = delta_point_key(delta)
        ours_row = points["ours"][key]
        kalibr_row = points["kalibr"][key]
        image_path = image_map.get(ours_row["frame_label"])
        if image_path is None:
            raise RuntimeError(
                f"No holdout image matching frame label: {ours_row['frame_label']}"
            )
        ours_image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        kalibr_image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if ours_image is None or kalibr_image is None:
            raise RuntimeError(f"Failed to load image: {image_path}")
        draw_points(ours_image, [ours_row])
        draw_points(kalibr_image, [kalibr_row])
        x0, y0, x1, y1 = point_context_crop_bounds(
            [ours_row, kalibr_row], ours_image.shape)
        ours_crop = ours_image[y0:y1, x0:x1].copy()
        kalibr_crop = kalibr_image[y0:y1, x0:x1].copy()
        delta_px = float(delta["prediction_delta_norm_px"])
        ours_residual = float(delta["ours_residual_norm_px"])
        kalibr_residual = float(delta["kalibr_residual_norm_px"])
        filename = (
            f"rank{rank:02d}_delta_{delta_px:.6f}px_"
            f"frame_{delta['frame_index']}_board_{delta['board_id']}_"
            f"point_{delta['point_id']}_{delta['point_type']}.png"
        )
        cv2.imwrite(str(top_dir / filename),
                    labeled_compare(ours_crop, "Ours", kalibr_crop, "Kalibr",
                                    show_labels=False))
        overview.append(
            f"{rank},{delta['frame_label']},{delta['board_id']},{delta['point_id']},"
            f"{delta['point_type']},{delta_px:.9f},{ours_residual:.9f},"
            f"{kalibr_residual:.9f}"
        )
    (top_dir / "README.txt").write_text("\n".join(overview) + "\n", encoding="utf-8")


def export_ranked_prediction_delta_by_point_type(
        output_dir: Path,
        deltas: list[dict[str, object]],
        points: dict[str, dict[tuple[str, ...], dict[str, str]]],
        image_map: dict[str, Path],
        top_count: int) -> None:
    root = output_dir / "ranked_ours_vs_kalibr_prediction_differences_by_point_type"
    root.mkdir(parents=True, exist_ok=True)
    summary: list[dict[str, object]] = []
    for point_type in ("outer", "internal"):
        matching = [delta for delta in deltas if delta["point_type"] == point_type]
        if not matching:
            raise RuntimeError(f"No {point_type} points available for difference audit")
        selected = matching[:top_count]
        export_prediction_delta_crops(
            output_dir, selected, points, image_map,
            Path("ranked_ours_vs_kalibr_prediction_differences_by_point_type") / point_type,
            "ranked_prediction_differences.csv",
            f"largest {len(selected)} {point_type} prediction differences, descending order")
        largest = selected[0]
        summary.append({
            "point_type": point_type,
            "ranked_count": len(selected),
            "rank_1_frame_index": largest["frame_index"],
            "rank_1_frame_label": largest["frame_label"],
            "rank_1_board_id": largest["board_id"],
            "rank_1_point_id": largest["point_id"],
            "rank_1_prediction_delta_norm_px": largest["prediction_delta_norm_px"],
            "rank_1_ours_residual_norm_px": largest["ours_residual_norm_px"],
            "rank_1_kalibr_residual_norm_px": largest["kalibr_residual_norm_px"],
        })
    write_csv(root / "ranked_prediction_differences_by_point_type.csv", summary)
    (root / "README.txt").write_text(
        "Each point type is independently sorted by descending Ours-vs-Kalibr "
        "projected-pixel distance. Every paired PNG uses the same crop coordinates "
        "and pixel scale on both sides, with identical point-type colors. "
        "PNG files deliberately contain no text; see filenames and CSV files for IDs.\n",
        encoding="utf-8")


def export_worst_validation_frame_point_residual_comparison(
        output_dir: Path,
        deltas: list[dict[str, object]],
        points: dict[str, dict[tuple[str, ...], dict[str, str]]],
        image_map: dict[str, Path],
        worst_frame_count: int,
        points_per_type: int) -> None:
    """Audit high-RMSE validation frames with paired worst-point comparisons."""
    root = output_dir / "worst_validation_frames_top_point_residual_comparison"
    root.mkdir(parents=True, exist_ok=True)
    thumbnail_roots = {
        "outer": root / "outer_top_point_thumbnails",
        "internal": root / "internal_top_point_thumbnails",
    }
    for thumbnail_root in thumbnail_roots.values():
        thumbnail_root.mkdir(parents=True, exist_ok=True)
    deltas_by_frame: dict[int, list[dict[str, object]]] = defaultdict(list)
    for delta in deltas:
        deltas_by_frame[int(delta["frame_index"])].append(delta)

    ranked_frames: list[tuple[float, int, float, float, list[dict[str, object]]]] = []
    for frame_index, frame_deltas in deltas_by_frame.items():
        ours_rows = [points["ours"][delta_point_key(delta)] for delta in frame_deltas]
        kalibr_rows = [points["kalibr"][delta_point_key(delta)] for delta in frame_deltas]
        ours_rmse = rms(ours_rows)
        kalibr_rmse = rms(kalibr_rows)
        ranked_frames.append((max(ours_rmse, kalibr_rmse), frame_index,
                              ours_rmse, kalibr_rmse, frame_deltas))
    ranked_frames.sort(reverse=True, key=lambda item: item[0])

    frame_summary: list[dict[str, object]] = []
    for frame_rank, (ranking_rmse, frame_index, ours_rmse, kalibr_rmse, frame_deltas) in enumerate(
            ranked_frames[:worst_frame_count], start=1):
        frame_label = str(frame_deltas[0]["frame_label"])
        image_path = image_map.get(frame_label)
        if image_path is None:
            raise RuntimeError(f"No holdout image matching frame label: {frame_label}")
        frame_dir = root / f"rank{frame_rank:02d}_frame_{frame_index}_{frame_label}"
        frame_dir.mkdir(parents=True, exist_ok=True)
        counts: dict[str, int] = {}
        selected_board_ids: set[int] = set()
        for point_type in ("outer", "internal"):
            typed = [delta for delta in frame_deltas if delta["point_type"] == point_type]
            typed.sort(key=lambda delta: max(
                float(delta["ours_residual_norm_px"]),
                float(delta["kalibr_residual_norm_px"])), reverse=True)
            selected = typed[:points_per_type]
            counts[point_type] = len(selected)
            if not selected:
                continue
            csv_rows: list[dict[str, object]] = []
            ours_selected = []
            kalibr_selected = []
            for point_rank, delta in enumerate(selected, start=1):
                row = dict(delta)
                row["point_rank"] = point_rank
                row["selection_residual_score_px"] = max(
                    float(delta["ours_residual_norm_px"]),
                    float(delta["kalibr_residual_norm_px"]))
                csv_rows.append(row)
                key = delta_point_key(delta)
                ours_selected.append(points["ours"][key])
                kalibr_selected.append(points["kalibr"][key])
                selected_board_ids.add(int(delta["board_id"]))
            write_csv(frame_dir / f"{point_type}_top_point_residual_comparison.csv", csv_rows)
            thumbnail_dir = thumbnail_roots[point_type] / (
                f"rank{frame_rank:02d}_frame_{frame_index}_{frame_label}"
            )
            thumbnail_dir.mkdir(parents=True, exist_ok=True)
            for point_rank, delta in enumerate(selected, start=1):
                key = delta_point_key(delta)
                ours_thumbnail = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                kalibr_thumbnail = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                if ours_thumbnail is None or kalibr_thumbnail is None:
                    raise RuntimeError(f"Failed to load image: {image_path}")
                ours_point = points["ours"][key]
                kalibr_point = points["kalibr"][key]
                x0, y0, x1, y1 = point_context_crop_bounds(
                    [ours_point, kalibr_point], ours_thumbnail.shape,
                    width=180, height=120)
                draw_points(ours_thumbnail, [ours_point])
                draw_points(kalibr_thumbnail, [kalibr_point])
                residual_score = max(
                    float(delta["ours_residual_norm_px"]),
                    float(delta["kalibr_residual_norm_px"]))
                filename = (
                    f"frame_rank{frame_rank:02d}_frame_{frame_index}_"
                    f"point_rank{point_rank:02d}_score_{residual_score:.6f}px_"
                    f"board_{delta['board_id']}_point_{delta['point_id']}_{point_type}.png"
                )
                cv2.imwrite(
                    str(thumbnail_dir / filename),
                    labeled_compare(ours_thumbnail[y0:y1, x0:x1], "Ours",
                                    kalibr_thumbnail[y0:y1, x0:x1], "Kalibr",
                                    show_labels=False))
            ours_image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            kalibr_image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if ours_image is None or kalibr_image is None:
                raise RuntimeError(f"Failed to load image: {image_path}")
            draw_points(ours_image, ours_selected)
            draw_points(kalibr_image, kalibr_selected)
            x0, y0, x1, y1 = clamp_crop(
                ours_selected + kalibr_selected, ours_image.shape, padding=90)
            ours_crop = ours_image[y0:y1, x0:x1].copy()
            kalibr_crop = kalibr_image[y0:y1, x0:x1].copy()
            cv2.imwrite(
                str(frame_dir / f"{point_type}_top{len(selected):02d}_point_residuals_ours_vs_kalibr.png"),
                labeled_compare(ours_crop, "Ours", kalibr_crop, "Kalibr",
                                show_labels=False))
        board_summary: list[dict[str, object]] = []
        for board_id in sorted(selected_board_ids):
            board_deltas = [
                delta for delta in frame_deltas if int(delta["board_id"]) == board_id
            ]
            board_deltas.sort(key=lambda delta: max(
                float(delta["ours_residual_norm_px"]),
                float(delta["kalibr_residual_norm_px"])), reverse=True)
            ours_board = [points["ours"][delta_point_key(delta)] for delta in board_deltas]
            kalibr_board = [points["kalibr"][delta_point_key(delta)] for delta in board_deltas]
            ours_image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            kalibr_image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if ours_image is None or kalibr_image is None:
                raise RuntimeError(f"Failed to load image: {image_path}")
            draw_points(ours_image, ours_board)
            draw_points(kalibr_image, kalibr_board)
            x0, y0, x1, y1 = clamp_crop(
                ours_board + kalibr_board, ours_image.shape, padding=80)
            ours_crop = ours_image[y0:y1, x0:x1].copy()
            kalibr_crop = kalibr_image[y0:y1, x0:x1].copy()
            cv2.imwrite(
                str(frame_dir / f"board_{board_id}_all_outer_internal_ours_vs_kalibr.png"),
                labeled_compare(ours_crop, "Ours", kalibr_crop, "Kalibr",
                                show_labels=False))
            board_rows: list[dict[str, object]] = []
            for point_rank, delta in enumerate(board_deltas, start=1):
                row = dict(delta)
                row["point_rank"] = point_rank
                row["selection_residual_score_px"] = max(
                    float(delta["ours_residual_norm_px"]),
                    float(delta["kalibr_residual_norm_px"]))
                board_rows.append(row)
            write_csv(frame_dir / f"board_{board_id}_all_point_residuals.csv", board_rows)
            board_summary.append({
                "board_id": board_id,
                "point_count": len(board_deltas),
                "outer_point_count": sum(
                    delta["point_type"] == "outer" for delta in board_deltas),
                "internal_point_count": sum(
                    delta["point_type"] == "internal" for delta in board_deltas),
                "ours_board_rmse_px": rms(ours_board),
                "kalibr_board_rmse_px": rms(kalibr_board),
            })
        if board_summary:
            write_csv(frame_dir / "board_rmse_comparison.csv", board_summary)
        frame_summary.append({
            "frame_rank": frame_rank,
            "frame_index": frame_index,
            "frame_label": frame_label,
            "frame_ranking_rmse_px": ranking_rmse,
            "ours_frame_rmse_px": ours_rmse,
            "kalibr_frame_rmse_px": kalibr_rmse,
            "outer_point_count": counts.get("outer", 0),
            "internal_point_count": counts.get("internal", 0),
        })
    write_csv(root / "worst_validation_frames.csv", frame_summary)
    (root / "README.txt").write_text(
        "Frames are sorted by max(Ours frame RMSE, Kalibr frame RMSE), descending. "
        "For each selected frame and point type, points are sorted by "
        "max(Ours point residual, Kalibr point residual), descending. Point values "
        "in the CSV are reprojection residual norms in pixels, not per-point RMSE. "
        "Paired PNGs share identical crop coordinates, scale, and point-type colors; "
        "the red cross is the detected observation. "
        "PNG files contain no text. Every board represented by a selected top point "
        "also has a board_<id>_all_outer_internal_ours_vs_kalibr.png crop containing "
        "all of that board's outer and internal points, plus board_rmse_comparison.csv. "
        "The root-level outer_top_point_thumbnails/ and internal_top_point_thumbnails/ "
        "directories group small paired crops by point type, then by ranked frame.\n",
        encoding="utf-8")


def export_specified_frame_board_comparison(
        output_dir: Path,
        deltas: list[dict[str, object]],
        points: dict[str, dict[tuple[str, ...], dict[str, str]]],
        image_map: dict[str, Path],
        frame_labels: list[str],
        points_per_type: int) -> None:
    """Export every detected board of explicitly requested holdout frames."""
    root = output_dir / "specified_frame_board_comparison"
    root.mkdir(parents=True, exist_ok=True)
    deltas_by_label: dict[str, list[dict[str, object]]] = defaultdict(list)
    for delta in deltas:
        deltas_by_label[str(delta["frame_label"])].append(delta)

    summary: list[dict[str, object]] = []
    for requested_label in frame_labels:
        frame_label = Path(requested_label).stem
        frame_deltas = deltas_by_label.get(frame_label, [])
        if not frame_deltas:
            raise RuntimeError(
                f"No paired Ours/Kalibr holdout points found for frame label: {frame_label}")
        image_path = image_map.get(frame_label)
        if image_path is None:
            raise RuntimeError(f"No image matching requested frame label: {frame_label}")
        frame_index = int(frame_deltas[0]["frame_index"])
        frame_dir = root / f"frame_{frame_index}_{frame_label}"
        frame_dir.mkdir(parents=True, exist_ok=True)
        board_summary: list[dict[str, object]] = []

        for board_id in sorted({int(delta["board_id"]) for delta in frame_deltas}):
            board_deltas = [
                delta for delta in frame_deltas if int(delta["board_id"]) == board_id
            ]
            board_dir = frame_dir / f"board_{board_id}"
            board_dir.mkdir(parents=True, exist_ok=True)
            ours_board = [points["ours"][delta_point_key(delta)] for delta in board_deltas]
            kalibr_board = [points["kalibr"][delta_point_key(delta)] for delta in board_deltas]

            ours_image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            kalibr_image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if ours_image is None or kalibr_image is None:
                raise RuntimeError(f"Failed to load image: {image_path}")
            draw_points(ours_image, ours_board)
            draw_points(kalibr_image, kalibr_board)
            x0, y0, x1, y1 = clamp_crop(
                ours_board + kalibr_board, ours_image.shape, padding=80)
            cv2.imwrite(
                str(board_dir / "all_outer_internal_ours_vs_kalibr.png"),
                labeled_compare(ours_image[y0:y1, x0:x1], "Ours",
                                kalibr_image[y0:y1, x0:x1], "Kalibr",
                                show_labels=False))

            type_counts: dict[str, int] = {}
            for point_type in ("outer", "internal"):
                typed = [
                    delta for delta in board_deltas
                    if delta["point_type"] == point_type
                ]
                typed.sort(key=lambda delta: max(
                    float(delta["ours_residual_norm_px"]),
                    float(delta["kalibr_residual_norm_px"])), reverse=True)
                ranked_rows: list[dict[str, object]] = []
                for point_rank, delta in enumerate(typed, start=1):
                    row = dict(delta)
                    row["point_rank"] = point_rank
                    row["selection_residual_score_px"] = max(
                        float(delta["ours_residual_norm_px"]),
                        float(delta["kalibr_residual_norm_px"]))
                    ranked_rows.append(row)
                type_counts[point_type] = len(ranked_rows)
                if not ranked_rows:
                    continue
                write_csv(board_dir / f"{point_type}_ranked_point_residuals.csv",
                          ranked_rows)

                selected = typed if points_per_type == 0 else typed[:points_per_type]
                ours_type = [points["ours"][delta_point_key(delta)] for delta in typed]
                kalibr_type = [points["kalibr"][delta_point_key(delta)] for delta in typed]
                ours_type_image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                kalibr_type_image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                if ours_type_image is None or kalibr_type_image is None:
                    raise RuntimeError(f"Failed to load image: {image_path}")
                draw_points(ours_type_image, ours_type)
                draw_points(kalibr_type_image, kalibr_type)
                tx0, ty0, tx1, ty1 = clamp_crop(
                    ours_type + kalibr_type, ours_type_image.shape, padding=60)
                cv2.imwrite(
                    str(board_dir / f"{point_type}_all_points_ours_vs_kalibr.png"),
                    labeled_compare(ours_type_image[ty0:ty1, tx0:tx1], "Ours",
                                    kalibr_type_image[ty0:ty1, tx0:tx1], "Kalibr",
                                    show_labels=False))

                crops_dir = board_dir / f"{point_type}_ranked_point_crops"
                crops_dir.mkdir(parents=True, exist_ok=True)
                for point_rank, delta in enumerate(selected, start=1):
                    key = delta_point_key(delta)
                    ours_point = points["ours"][key]
                    kalibr_point = points["kalibr"][key]
                    ours_crop_image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                    kalibr_crop_image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                    if ours_crop_image is None or kalibr_crop_image is None:
                        raise RuntimeError(f"Failed to load image: {image_path}")
                    draw_points(ours_crop_image, [ours_point])
                    draw_points(kalibr_crop_image, [kalibr_point])
                    px0, py0, px1, py1 = point_context_crop_bounds(
                        [ours_point, kalibr_point], ours_crop_image.shape,
                        width=180, height=120)
                    score = max(float(delta["ours_residual_norm_px"]),
                                float(delta["kalibr_residual_norm_px"]))
                    filename = (
                        f"rank{point_rank:02d}_score_{score:.6f}px_"
                        f"point_{delta['point_id']}_{point_type}.png")
                    cv2.imwrite(
                        str(crops_dir / filename),
                        labeled_compare(ours_crop_image[py0:py1, px0:px1], "Ours",
                                        kalibr_crop_image[py0:py1, px0:px1], "Kalibr",
                                        show_labels=False))

            board_summary.append({
                "frame_index": frame_index,
                "frame_label": frame_label,
                "board_id": board_id,
                "point_count": len(board_deltas),
                "outer_point_count": type_counts.get("outer", 0),
                "internal_point_count": type_counts.get("internal", 0),
                "ours_board_rmse_px": rms(ours_board),
                "kalibr_board_rmse_px": rms(kalibr_board),
            })
        write_csv(frame_dir / "board_rmse_comparison.csv", board_summary)
        summary.extend(board_summary)

    write_csv(root / "specified_frame_board_summary.csv", summary)
    (root / "README.txt").write_text(
        "Each requested frame is split by board. Each board has all-point Ours/Kalibr "
        "comparisons, separate outer/internal ranked CSV files, and rank-named local crops. "
        "Ranks use descending max(Ours residual, Kalibr residual) in pixels.\n",
        encoding="utf-8")


def main() -> int:
    args = parse_args()
    result_dir = args.result_dir.resolve()
    image_dir = args.image_dir.resolve()
    output_dir = (args.output_dir or
                  result_dir / "holdout_reprojection_visualizations_cross_observed_circle_projected").resolve()
    frames_dir = output_dir / "frames"
    boards_dir = output_dir / "boards"
    frames_dir.mkdir(parents=True, exist_ok=True)
    boards_dir.mkdir(parents=True, exist_ok=True)
    if (args.top_k <= 0 or args.top_point_count <= 0 or
            args.worst_frame_count <= 0 or args.points_per_type <= 0 or
            args.specified_frame_points_per_type < 0):
        raise ValueError("All requested audit counts must be positive")

    points = load_points(result_dir / "benchmark_holdout_points.csv")
    image_map = image_by_label(image_dir)
    deltas = comparison_rows(points)
    write_csv(output_dir / "ours_vs_kalibr_reprojection_point_delta.csv", deltas)
    write_csv(output_dir / "ours_vs_kalibr_reprojection_point_delta_top.csv",
              deltas[:args.top_point_count])
    export_prediction_delta_crops(
        output_dir, deltas[:args.top_point_count], points, image_map,
        Path("top_ours_vs_kalibr_prediction_differences"),
        "top_prediction_differences.csv",
        f"global top {args.top_point_count} prediction differences")
    export_ranked_prediction_delta_by_point_type(
        output_dir, deltas, points, image_map, args.top_point_count)
    export_worst_validation_frame_point_residual_comparison(
        output_dir, deltas, points, image_map,
        args.worst_frame_count, args.points_per_type)
    if args.frame_label:
        export_specified_frame_board_comparison(
            output_dir, deltas, points, image_map, args.frame_label,
            args.specified_frame_points_per_type)

    rows_by_frame: dict[str, dict[int, list[dict[str, str]]]] = {
        method: defaultdict(list) for method in METHODS
    }
    rows_by_board: dict[str, dict[tuple[int, int], list[dict[str, str]]]] = {
        method: defaultdict(list) for method in METHODS
    }
    for method in METHODS:
        for row in points[method].values():
            frame = int(row["frame_index"])
            board = int(row["board_id"])
            rows_by_frame[method][frame].append(row)
            rows_by_board[method][(frame, board)].append(row)

    ours_frames = sorted(rows_by_frame["ours"],
                         key=lambda frame: rms(rows_by_frame["ours"][frame]), reverse=True)
    ours_boards = sorted(rows_by_board["ours"],
                         key=lambda key: rms(rows_by_board["ours"][key]), reverse=True)
    summary = [
        "purpose: re-render frozen Stage5 holdout reprojections without rerunning calibration.",
        "marker_semantics: outer=red cross observation plus green/orange open-circle projection; internal=red/projected-color filled dots; gray line=residual.",
        f"point_key_count: {len(points['ours'])}",
        f"largest_prediction_delta_px: {float(deltas[0]['prediction_delta_norm_px']):.9f}",
        f"frame_top_k: {args.top_k}",
        f"board_top_k: {args.top_k}",
        "",
        "[frames]",
    ]

    for rank, frame in enumerate(ours_frames[:args.top_k], start=1):
        ours_rows = rows_by_frame["ours"][frame]
        kalibr_rows = rows_by_frame["kalibr"][frame]
        label = ours_rows[0]["frame_label"]
        image_path = image_map.get(label)
        if image_path is None:
            raise RuntimeError(f"No holdout image matching frame label: {label}")
        ours_rmse = rms(ours_rows)
        kalibr_rmse = rms(kalibr_rows)
        ours_image = render_overlay(image_path, ours_rows, "Ours", ours_rmse)
        kalibr_image = render_overlay(
            image_path, kalibr_rows, "Kalibr", kalibr_rmse,
            None, "green/orange")
        stem = f"rank{rank}_frame_{frame}_{label}"
        cv2.imwrite(str(frames_dir / f"{stem}_ours_cross_observed_circle_projected.png"), ours_image)
        cv2.imwrite(str(frames_dir / f"{stem}_ours_vs_kalibr_cross_circle.png"),
                    labeled_compare(ours_image, f"Ours RMSE={ours_rmse:.6f}",
                                    kalibr_image, f"Kalibr RMSE={kalibr_rmse:.6f}"))
        summary.append(
            f"rank{rank},frame_index={frame},frame_label={label},"
            f"ours_rmse={ours_rmse:.9f},kalibr_rmse={kalibr_rmse:.9f}"
        )

    summary.append("")
    summary.append("[boards]")
    for rank, (frame, board) in enumerate(ours_boards[:args.top_k], start=1):
        ours_rows = rows_by_board["ours"][(frame, board)]
        kalibr_rows = rows_by_board["kalibr"][(frame, board)]
        label = ours_rows[0]["frame_label"]
        image_path = image_map.get(label)
        if image_path is None:
            raise RuntimeError(f"No holdout image matching frame label: {label}")
        ours_rmse = rms(ours_rows)
        kalibr_rmse = rms(kalibr_rows)
        ours_image = render_overlay(image_path, ours_rows, f"Ours board={board}", ours_rmse)
        kalibr_image = render_overlay(
            image_path, kalibr_rows, f"Kalibr board={board}", kalibr_rmse,
            None, "green/orange")
        x0, y0, x1, y1 = clamp_crop(ours_rows + kalibr_rows, ours_image.shape, padding=80)
        ours_crop = ours_image[y0:y1, x0:x1].copy()
        kalibr_crop = kalibr_image[y0:y1, x0:x1].copy()
        add_banner(ours_crop, f"Ours frame={frame} board={board} rmse={ours_rmse:.6f}",
                   "outer: red cross/green circle | internal: red/orange dots")
        add_banner(kalibr_crop, f"Kalibr frame={frame} board={board} rmse={kalibr_rmse:.6f}",
                   "outer: red cross/green circle | internal: red/orange dots")
        stem = f"rank{rank}_frame_{frame}_board_{board}_{label}"
        cv2.imwrite(str(boards_dir / f"{stem}_ours_cross_observed_circle_projected.png"), ours_crop)
        cv2.imwrite(str(boards_dir / f"{stem}_ours_vs_kalibr_cross_circle.png"),
                    labeled_compare(ours_crop, f"Ours RMSE={ours_rmse:.6f}",
                                    kalibr_crop, f"Kalibr RMSE={kalibr_rmse:.6f}"))
        summary.append(
            f"rank{rank},frame_index={frame},frame_label={label},board_id={board},"
            f"ours_rmse={ours_rmse:.9f},kalibr_rmse={kalibr_rmse:.9f}"
        )

    (output_dir / "reprojection_overlay_summary.txt").write_text(
        "\n".join(summary) + "\n", encoding="utf-8"
    )
    print(f"Wrote overlays: {output_dir}")
    print(f"Wrote point deltas: {output_dir / 'ours_vs_kalibr_reprojection_point_delta_top.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
