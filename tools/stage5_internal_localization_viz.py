#!/usr/bin/env python3
"""Visualize large Stage5 internal-point localization residuals.

The input CSV is produced by run_stage5_backend as backend_training_points.csv
or benchmark_training_points.csv. For internal points, observed_x/y is the
regenerated/localized internal point, while predicted_x/y is the projection from
the optimized camera and refit board pose used by the Stage5 evaluator.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path


def _import_cv2():
    try:
        import cv2  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "OpenCV Python module is required for visualization: import cv2 failed"
        ) from exc
    return cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find and visualize worst Stage5 internal localization residuals."
    )
    parser.add_argument("--points-csv", required=True, help="Path to backend_training_points.csv")
    parser.add_argument("--image-dir", required=True, help="Directory containing input images")
    parser.add_argument("--output-dir", required=True, help="Directory for reports and overlays")
    parser.add_argument("--method", default="", help="Optional method label filter, e.g. backend")
    parser.add_argument("--split", default="training", help="Split label filter")
    parser.add_argument("--top-frames", type=int, default=20, help="Number of worst frames to draw")
    parser.add_argument(
        "--top-points-per-frame",
        type=int,
        default=30,
        help="Maximum residual vectors to draw per frame",
    )
    parser.add_argument(
        "--min-residual",
        type=float,
        default=0.0,
        help="Only draw/report internal points at or above this residual in pixels",
    )
    parser.add_argument(
        "--rank-frames-by",
        choices=("max_residual", "p95_residual", "mean_residual", "x_std", "x_p95_abs", "x_max_abs"),
        default="max_residual",
        help="Frame ranking metric for choosing overlays",
    )
    parser.add_argument(
        "--rank-points-by",
        choices=("residual_norm", "abs_residual_x"),
        default="residual_norm",
        help="Point ranking metric inside each overlay",
    )
    return parser.parse_args()


def to_float(row: dict[str, str], key: str) -> float:
    try:
        return float(row[key])
    except (KeyError, ValueError) as exc:
        raise ValueError(f"Invalid numeric value for {key}: {row.get(key)!r}") from exc


def to_int(row: dict[str, str], key: str) -> int:
    try:
        return int(row[key])
    except (KeyError, ValueError) as exc:
        raise ValueError(f"Invalid integer value for {key}: {row.get(key)!r}") from exc


def find_image_path(image_dir: Path, frame_label: str) -> Path | None:
    candidates = [image_dir / frame_label]
    for suffix in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"):
        candidates.append(image_dir / f"{frame_label}{suffix}")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    matches = sorted(image_dir.glob(f"{frame_label}*"))
    return matches[0] if matches else None


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = int(round((len(ordered) - 1) * q))
    return ordered[max(0, min(idx, len(ordered) - 1))]


def load_internal_points(args: argparse.Namespace) -> list[dict[str, object]]:
    points: list[dict[str, object]] = []
    with open(args.points_csv, newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("point_type") != "internal":
                continue
            if args.method and row.get("method") != args.method:
                continue
            if args.split and row.get("split") != args.split:
                continue
            residual_norm = to_float(row, "residual_norm")
            if residual_norm < args.min_residual:
                continue
            points.append(
                {
                    "method": row.get("method", ""),
                    "split": row.get("split", ""),
                    "frame_index": to_int(row, "frame_index"),
                    "frame_label": row.get("frame_label", ""),
                    "board_id": to_int(row, "board_id"),
                    "point_id": row.get("point_id", ""),
                    "observed_x": to_float(row, "observed_x"),
                    "observed_y": to_float(row, "observed_y"),
                    "predicted_x": to_float(row, "predicted_x"),
                    "predicted_y": to_float(row, "predicted_y"),
                    "residual_x": to_float(row, "residual_x"),
                    "residual_y": to_float(row, "residual_y"),
                    "residual_norm": residual_norm,
                    "source_kind": row.get("source_kind", ""),
                    "source_point_index": row.get("source_point_index", ""),
                }
            )
    return points


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def frame_rows(points: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[int, str], list[dict[str, object]]] = defaultdict(list)
    for point in points:
        grouped[(int(point["frame_index"]), str(point["frame_label"]))].append(point)

    rows: list[dict[str, object]] = []
    for (frame_index, frame_label), frame_points in grouped.items():
        norms = [float(point["residual_norm"]) for point in frame_points]
        residual_x = [float(point["residual_x"]) for point in frame_points]
        residual_y = [float(point["residual_y"]) for point in frame_points]
        abs_x = [abs(value) for value in residual_x]
        mean_x = sum(residual_x) / len(residual_x)
        mean_y = sum(residual_y) / len(residual_y)
        std_x = math.sqrt(sum((value - mean_x) ** 2 for value in residual_x) / len(residual_x))
        std_y = math.sqrt(sum((value - mean_y) ** 2 for value in residual_y) / len(residual_y))
        board_to_max: dict[int, float] = defaultdict(float)
        for point in frame_points:
            board_id = int(point["board_id"])
            board_to_max[board_id] = max(board_to_max[board_id], float(point["residual_norm"]))
        worst_board = max(board_to_max.items(), key=lambda item: item[1])[0]
        rows.append(
            {
                "frame_index": frame_index,
                "frame_label": frame_label,
                "internal_point_count": len(frame_points),
                "mean_internal_residual": sum(norms) / len(norms),
                "p95_internal_residual": percentile(norms, 0.95),
                "max_internal_residual": max(norms),
                "mean_residual_x": mean_x,
                "mean_residual_y": mean_y,
                "std_residual_x": std_x,
                "std_residual_y": std_y,
                "p95_abs_residual_x": percentile(abs_x, 0.95),
                "max_abs_residual_x": max(abs_x),
                "worst_board_id": worst_board,
                "worst_board_max_residual": board_to_max[worst_board],
            }
        )
    return rows


def sort_frame_rows(rows: list[dict[str, object]], rank_by: str) -> list[dict[str, object]]:
    metric_map = {
        "max_residual": "max_internal_residual",
        "p95_residual": "p95_internal_residual",
        "mean_residual": "mean_internal_residual",
        "x_std": "std_residual_x",
        "x_p95_abs": "p95_abs_residual_x",
        "x_max_abs": "max_abs_residual_x",
    }
    primary = metric_map[rank_by]
    rows.sort(
        key=lambda row: (
            float(row[primary]),
            float(row["std_residual_x"]),
            float(row["p95_abs_residual_x"]),
            float(row["max_internal_residual"]),
        ),
        reverse=True,
    )
    return rows


def draw_cross(cv2, image, x: float, y: float, color: tuple[int, int, int], radius: int) -> None:
    center = (int(round(x)), int(round(y)))
    cv2.line(image, (center[0] - radius, center[1]), (center[0] + radius, center[1]), color, 2)
    cv2.line(image, (center[0], center[1] - radius), (center[0], center[1] + radius), color, 2)


def draw_overlays(
    args: argparse.Namespace,
    points: list[dict[str, object]],
    frames: list[dict[str, object]],
    output_dir: Path,
) -> list[dict[str, object]]:
    cv2 = _import_cv2()
    image_dir = Path(args.image_dir)
    overlay_dir = output_dir / "overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)

    by_frame: dict[tuple[int, str], list[dict[str, object]]] = defaultdict(list)
    for point in points:
        by_frame[(int(point["frame_index"]), str(point["frame_label"]))].append(point)

    overlay_rows: list[dict[str, object]] = []
    for frame in frames[: args.top_frames]:
        frame_index = int(frame["frame_index"])
        frame_label = str(frame["frame_label"])
        image_path = find_image_path(image_dir, frame_label)
        if image_path is None:
            overlay_rows.append(
                {
                    "frame_index": frame_index,
                    "frame_label": frame_label,
                    "overlay_path": "",
                    "status": "missing_image",
                }
            )
            continue
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            overlay_rows.append(
                {
                    "frame_index": frame_index,
                    "frame_label": frame_label,
                    "overlay_path": "",
                    "status": "failed_to_read_image",
                }
            )
            continue

        if args.rank_points_by == "abs_residual_x":
            frame_points = sorted(
                by_frame[(frame_index, frame_label)],
                key=lambda point: abs(float(point["residual_x"])),
                reverse=True,
            )[: args.top_points_per_frame]
        else:
            frame_points = sorted(
                by_frame[(frame_index, frame_label)],
                key=lambda point: float(point["residual_norm"]),
                reverse=True,
            )[: args.top_points_per_frame]

        for rank, point in enumerate(frame_points, start=1):
            ox = float(point["observed_x"])
            oy = float(point["observed_y"])
            px = float(point["predicted_x"])
            py = float(point["predicted_y"])
            residual = float(point["residual_norm"])
            board_id = int(point["board_id"])
            cv2.line(
                image,
                (int(round(px)), int(round(py))),
                (int(round(ox)), int(round(oy))),
                (0, 220, 255),
                2,
            )
            draw_cross(cv2, image, px, py, (255, 0, 255), 6)
            cv2.circle(image, (int(round(ox)), int(round(oy))), 5, (255, 255, 0), -1)
            if rank <= 8:
                label = f"b{board_id}:{residual:.1f}px"
                cv2.putText(
                    image,
                    label,
                    (int(round(ox)) + 8, int(round(oy)) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )

        title = (
            f"frame {frame_index}  xstd={float(frame['std_residual_x']):.2f}px  "
            f"x95={float(frame['p95_abs_residual_x']):.2f}px  "
            f"max={float(frame['max_internal_residual']):.2f}px"
        )
        cv2.rectangle(image, (20, 20), (min(image.shape[1] - 20, 1100), 86), (0, 0, 0), -1)
        cv2.putText(
            image,
            title,
            (35, 58),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            "cyan dot=localized internal point, magenta cross=model projection, yellow line=residual",
            (35, 82),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        output_path = overlay_dir / f"frame_{frame_index:06d}_internal_localization.png"
        cv2.imwrite(str(output_path), image)
        overlay_rows.append(
            {
                "frame_index": frame_index,
                "frame_label": frame_label,
                "overlay_path": str(output_path),
                "status": "ok",
            }
        )
    return overlay_rows


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    points = load_internal_points(args)
    points.sort(key=lambda point: float(point["residual_norm"]), reverse=True)
    frames = sort_frame_rows(frame_rows(points), args.rank_frames_by)

    point_fields = [
        "method",
        "split",
        "frame_index",
        "frame_label",
        "board_id",
        "point_id",
        "observed_x",
        "observed_y",
        "predicted_x",
        "predicted_y",
        "residual_x",
        "residual_y",
        "residual_norm",
        "source_kind",
        "source_point_index",
    ]
    frame_fields = [
        "frame_index",
        "frame_label",
        "internal_point_count",
        "mean_internal_residual",
        "p95_internal_residual",
        "max_internal_residual",
        "mean_residual_x",
        "mean_residual_y",
        "std_residual_x",
        "std_residual_y",
        "p95_abs_residual_x",
        "max_abs_residual_x",
        "worst_board_id",
        "worst_board_max_residual",
    ]
    write_csv(output_dir / "internal_localization_worst_points.csv", point_fields, points)
    write_csv(output_dir / "internal_localization_worst_frames.csv", frame_fields, frames)
    overlay_rows = draw_overlays(args, points, frames, output_dir)
    write_csv(
        output_dir / "internal_localization_overlay_index.csv",
        ["frame_index", "frame_label", "overlay_path", "status"],
        overlay_rows,
    )

    with open(output_dir / "README.txt", "w") as handle:
        handle.write("Stage5 internal point localization visualization\n")
        handle.write("cyan dot: regenerated/localized internal point used as observation\n")
        handle.write("magenta cross: optimized model projection from Stage5 evaluator\n")
        handle.write("yellow line: reprojection residual vector\n")
        handle.write("Note: model projection is an evaluator reference, not physical ground truth.\n")
        handle.write(f"points_csv: {args.points_csv}\n")
        handle.write(f"image_dir: {args.image_dir}\n")
        handle.write(f"top_frames: {args.top_frames}\n")
        handle.write(f"top_points_per_frame: {args.top_points_per_frame}\n")
        handle.write(f"rank_frames_by: {args.rank_frames_by}\n")
        handle.write(f"rank_points_by: {args.rank_points_by}\n")

    print(f"internal points: {len(points)}")
    print(f"frames: {len(frames)}")
    print(f"output_dir: {output_dir}")
    if frames:
        worst = frames[0]
        print(
            "worst_frame: "
            f"{worst['frame_index']} xstd={float(worst['std_residual_x']):.3f}px "
            f"x95={float(worst['p95_abs_residual_x']):.3f}px "
            f"mean={float(worst['mean_internal_residual']):.3f}px"
        )


if __name__ == "__main__":
    main()
