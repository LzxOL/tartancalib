#!/usr/bin/env python3
"""Visualize Stage6 polar-angle buckets on a camera frame canvas.

The input is the Stage6 angular diagnostic corner trace produced by
`--stage6-export-angular-fixedk-diagnostic`.  The script draws observed corner
locations for each camera, colored by polar-angle bucket, and writes residual
bucket summaries.
"""

import argparse
import csv
import math
import os
from collections import defaultdict
from statistics import median


DEFAULT_BUCKETS = [
    ("0-30 deg", 0.0, 30.0, "#2563eb"),
    ("30-45 deg", 30.0, 45.0, "#0891b2"),
    ("45-60 deg", 45.0, 60.0, "#16a34a"),
    ("60-75 deg", 60.0, 75.0, "#f59e0b"),
    ("75+ deg", 75.0, float("inf"), "#dc2626"),
]


def finite_float(value, default=float("nan")):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def percentile(values, q):
    if not values:
        return float("nan")
    ordered = sorted(values)
    index = (len(ordered) - 1) * q
    lo = int(math.floor(index))
    hi = int(math.ceil(index))
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - index) + ordered[hi] * (index - lo)


def rms(values):
    if not values:
        return float("nan")
    return math.sqrt(sum(v * v for v in values) / len(values))


def bucket_for_angle(angle_deg):
    for name, lower, upper, color in DEFAULT_BUCKETS:
        if angle_deg >= lower and angle_deg < upper:
            return name, lower, upper, color
    return None


def load_trace(path):
    rows = []
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if str(row.get("valid", "")).strip() not in ("1", "true", "True"):
                continue
            x = finite_float(row.get("u_obs_x"))
            y = finite_float(row.get("u_obs_y"))
            polar = finite_float(row.get("polar_angle_deg"))
            pixel = finite_float(row.get("pixel_error_px"))
            angular = finite_float(row.get("angular_error_deg"))
            cam_id = int(float(row.get("cam_id", "0")))
            if not all(math.isfinite(v) for v in (x, y, polar, pixel, angular)):
                continue
            bucket = bucket_for_angle(polar)
            if bucket is None:
                continue
            rows.append(
                {
                    "cam_id": cam_id,
                    "x": x,
                    "y": y,
                    "polar": polar,
                    "pixel": pixel,
                    "angular": angular,
                    "bucket": bucket[0],
                    "color": bucket[3],
                    "frame_id": row.get("frame_id", ""),
                    "board_id": row.get("board_id", ""),
                    "corner_id": row.get("corner_id", ""),
                }
            )
    return rows


def aggregate(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["bucket"]].append(row)
    summaries = []
    for name, lower, upper, color in DEFAULT_BUCKETS:
        items = grouped.get(name, [])
        pixels = [r["pixel"] for r in items]
        angulars = [r["angular"] for r in items]
        xs = [r["x"] for r in items]
        ys = [r["y"] for r in items]
        summaries.append(
            {
                "bucket_name": name,
                "polar_min_deg": lower,
                "polar_max_deg": upper,
                "corner_count": len(items),
                "pixel_rmse_px": rms(pixels),
                "pixel_median_px": median(pixels) if pixels else float("nan"),
                "pixel_p90_px": percentile(pixels, 0.90),
                "angular_rmse_deg": rms(angulars),
                "angular_median_deg": median(angulars) if angulars else float("nan"),
                "angular_p90_deg": percentile(angulars, 0.90),
                "bbox_min_x": min(xs) if xs else float("nan"),
                "bbox_min_y": min(ys) if ys else float("nan"),
                "bbox_max_x": max(xs) if xs else float("nan"),
                "bbox_max_y": max(ys) if ys else float("nan"),
                "color": color,
            }
        )
    return summaries


def infer_canvas(rows, width_arg, height_arg):
    if width_arg > 0 and height_arg > 0:
        return width_arg, height_arg
    max_x = max((r["x"] for r in rows), default=0.0)
    max_y = max((r["y"] for r in rows), default=0.0)
    width = width_arg if width_arg > 0 else int(math.ceil(max_x / 100.0) * 100)
    height = height_arg if height_arg > 0 else int(math.ceil(max_y / 100.0) * 100)
    return max(width, 1), max(height, 1)


def svg_escape(text):
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def write_svg(rows, summaries, output_path, frame_width, frame_height, max_points):
    scale = min(1.0, 720.0 / max(frame_width, frame_height))
    panel_w = frame_width * scale
    panel_h = frame_height * scale
    gutter = 70
    margin = 30
    legend_h = 170
    svg_w = margin * 2 + panel_w * 2 + gutter
    svg_h = margin * 2 + panel_h + legend_h

    by_cam = defaultdict(list)
    for row in rows:
        by_cam[row["cam_id"]].append(row)

    def sample(items):
        if len(items) <= max_points:
            return items
        step = max(1, int(math.ceil(len(items) / max_points)))
        return items[::step]

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{svg_w:.0f}" '
        f'height="{svg_h:.0f}" viewBox="0 0 {svg_w:.3f} {svg_h:.3f}">',
        "<style>",
        "text{font-family:Arial,Helvetica,sans-serif;fill:#111827}",
        ".small{font-size:13px}.label{font-size:15px;font-weight:700}",
        ".panel{fill:#f8fafc;stroke:#94a3b8;stroke-width:1.5}",
        ".bbox{fill-opacity:0.055;stroke-width:3}",
        "</style>",
        '<rect x="0" y="0" width="100%" height="100%" fill="#ffffff"/>',
    ]

    for panel_index, cam_id in enumerate([0, 1]):
        ox = margin + panel_index * (panel_w + gutter)
        oy = margin + 28
        parts.append(
            f'<text class="label" x="{ox:.3f}" y="{margin:.3f}">cam{cam_id}</text>'
        )
        parts.append(
            f'<rect class="panel" x="{ox:.3f}" y="{oy:.3f}" '
            f'width="{panel_w:.3f}" height="{panel_h:.3f}"/>'
        )
        cam_rows = by_cam.get(cam_id, [])
        for summary in summaries:
            bucket_rows = [r for r in cam_rows if r["bucket"] == summary["bucket_name"]]
            if not bucket_rows:
                continue
            min_x = min(r["x"] for r in bucket_rows) * scale + ox
            min_y = min(r["y"] for r in bucket_rows) * scale + oy
            max_x = max(r["x"] for r in bucket_rows) * scale + ox
            max_y = max(r["y"] for r in bucket_rows) * scale + oy
            color = summary["color"]
            parts.append(
                f'<rect class="bbox" x="{min_x:.3f}" y="{min_y:.3f}" '
                f'width="{max(1.0, max_x - min_x):.3f}" '
                f'height="{max(1.0, max_y - min_y):.3f}" '
                f'fill="{color}" stroke="{color}"/>'
            )
        for row in sample(cam_rows):
            x = ox + row["x"] * scale
            y = oy + row["y"] * scale
            parts.append(
                f'<circle cx="{x:.3f}" cy="{y:.3f}" r="2.1" '
                f'fill="{row["color"]}" fill-opacity="0.72"/>'
            )
        parts.append(
            f'<text class="small" x="{ox:.3f}" y="{oy + panel_h + 22:.3f}">'
            f'points shown: {min(len(cam_rows), max_points)} / {len(cam_rows)}</text>'
        )

    legend_x = margin
    legend_y = margin + panel_h + 85
    parts.append(f'<text class="label" x="{legend_x:.3f}" y="{legend_y:.3f}">'
                 "Polar bucket residual summary</text>")
    y = legend_y + 26
    header = "bucket | count | pixel RMSE px | angular RMSE deg | frame bbox"
    parts.append(f'<text class="small" x="{legend_x:.3f}" y="{y:.3f}">'
                 f'{svg_escape(header)}</text>')
    for summary in summaries:
        y += 21
        color = summary["color"]
        bbox = (
            f'[{summary["bbox_min_x"]:.0f},{summary["bbox_min_y"]:.0f}]'
            f'-[{summary["bbox_max_x"]:.0f},{summary["bbox_max_y"]:.0f}]'
            if summary["corner_count"]
            else "n/a"
        )
        line = (
            f'{summary["bucket_name"]} | {summary["corner_count"]} | '
            f'{summary["pixel_rmse_px"]:.4f} | '
            f'{summary["angular_rmse_deg"]:.5f} | {bbox}'
        )
        parts.append(
            f'<rect x="{legend_x:.3f}" y="{y - 12:.3f}" width="12" height="12" '
            f'fill="{color}"/>'
        )
        parts.append(
            f'<text class="small" x="{legend_x + 18:.3f}" y="{y:.3f}">'
            f'{svg_escape(line)}</text>'
        )
    parts.append("</svg>\n")
    with open(output_path, "w") as handle:
        handle.write("\n".join(parts))


def write_csv(path, summaries):
    fields = [
        "bucket_name",
        "polar_min_deg",
        "polar_max_deg",
        "corner_count",
        "pixel_rmse_px",
        "pixel_median_px",
        "pixel_p90_px",
        "angular_rmse_deg",
        "angular_median_deg",
        "angular_p90_deg",
        "bbox_min_x",
        "bbox_min_y",
        "bbox_max_x",
        "bbox_max_y",
    ]
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for summary in summaries:
            writer.writerow({field: summary[field] for field in fields})


def write_markdown(path, summaries, svg_name, trace_csv):
    with open(path, "w") as handle:
        handle.write("# Stage6 Polar Bucket Frame Map\n\n")
        handle.write(f"trace_csv: `{trace_csv}`\n\n")
        handle.write(f"visualization: `{svg_name}`\n\n")
        handle.write("| polar bucket | corners | pixel RMSE px | pixel median px | "
                     "pixel p90 px | angular RMSE deg | angular median deg | "
                     "angular p90 deg | approximate frame bbox |\n")
        handle.write("|---|---:|---:|---:|---:|---:|---:|---:|---|\n")
        for summary in summaries:
            if summary["corner_count"]:
                bbox = (
                    f'[{summary["bbox_min_x"]:.0f}, {summary["bbox_min_y"]:.0f}]'
                    f' - [{summary["bbox_max_x"]:.0f}, {summary["bbox_max_y"]:.0f}]'
                )
            else:
                bbox = "n/a"
            handle.write(
                f'| {summary["bucket_name"]} | {summary["corner_count"]} | '
                f'{summary["pixel_rmse_px"]:.4f} | '
                f'{summary["pixel_median_px"]:.4f} | '
                f'{summary["pixel_p90_px"]:.4f} | '
                f'{summary["angular_rmse_deg"]:.5f} | '
                f'{summary["angular_median_deg"]:.5f} | '
                f'{summary["angular_p90_deg"]:.5f} | {bbox} |\n'
            )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--frame-width", type=int, default=0)
    parser.add_argument("--frame-height", type=int, default=0)
    parser.add_argument("--max-points-per-camera", type=int, default=9000)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    rows = load_trace(args.trace_csv)
    if not rows:
        raise SystemExit("No valid rows found in trace CSV.")
    frame_width, frame_height = infer_canvas(
        rows, args.frame_width, args.frame_height
    )
    summaries = aggregate(rows)
    svg_name = "polar_bucket_frame_map.svg"
    write_svg(
        rows,
        summaries,
        os.path.join(args.output_dir, svg_name),
        frame_width,
        frame_height,
        max(1, args.max_points_per_camera),
    )
    write_csv(os.path.join(args.output_dir, "polar_bucket_residual_summary.csv"),
              summaries)
    write_markdown(
        os.path.join(args.output_dir, "polar_bucket_report.md"),
        summaries,
        svg_name,
        args.trace_csv,
    )
    print(f"rows: {len(rows)}")
    print(f"frame_size: {frame_width}x{frame_height}")
    print(f"wrote: {os.path.join(args.output_dir, svg_name)}")
    print(f"wrote: {os.path.join(args.output_dir, 'polar_bucket_report.md')}")


if __name__ == "__main__":
    main()
