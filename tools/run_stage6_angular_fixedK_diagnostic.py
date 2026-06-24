#!/usr/bin/env python3
import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ModuleNotFoundError:
    HAS_MATPLOTLIB = False
    from PIL import Image, ImageDraw, ImageFont


BUCKETS = [
    ("0-30 deg", 0.0, 30.0),
    ("30-45 deg", 30.0, 45.0),
    ("45-60 deg", 45.0, 60.0),
    ("60-75 deg", 60.0, 75.0),
    ("75+ deg", 75.0, math.inf),
]


def as_float(value):
    try:
        out = float(value)
    except (TypeError, ValueError):
        return math.nan
    return out


def finite(value):
    return isinstance(value, float) and math.isfinite(value)


def percentile(values, q):
    clean = sorted(v for v in values if finite(v))
    if not clean:
        return math.nan
    if len(clean) == 1:
        return clean[0]
    pos = (len(clean) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return clean[lo]
    return clean[lo] * (hi - pos) + clean[hi] * (pos - lo)


def rmse(values):
    clean = [v for v in values if finite(v)]
    if not clean:
        return math.nan
    return math.sqrt(sum(v * v for v in clean) / len(clean))


def fmt(value):
    if isinstance(value, bool):
        return "1" if value else "0"
    if value is None:
        return ""
    if isinstance(value, float):
        if not math.isfinite(value):
            return "nan"
        return f"{value:.10g}"
    return str(value)


def read_trace(path):
    rows = []
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            parsed = dict(row)
            for key in [
                "pixel_error_px",
                "angular_error_rad",
                "angular_error_deg",
                "chordal_error",
                "polar_angle_deg",
                "edge_distance_px",
            ]:
                parsed[key] = as_float(row.get(key))
            for key in ["cam_id", "frame_id", "board_id", "valid", "is_rescued"]:
                try:
                    parsed[key] = int(row.get(key, "0") or 0)
                except ValueError:
                    parsed[key] = 0
            rows.append(parsed)
    return rows


def subset(rows, cam_id=None):
    out = [r for r in rows if r.get("valid") == 1]
    if cam_id is not None:
        out = [r for r in out if r.get("cam_id") == cam_id]
    return out


def metric_block(rows, prefix):
    return {
        f"pixel_rmse_px_{prefix}": rmse([r["pixel_error_px"] for r in rows]),
        f"angular_rmse_deg_{prefix}": rmse([r["angular_error_deg"] for r in rows]),
        f"angular_median_deg_{prefix}": percentile(
            [r["angular_error_deg"] for r in rows], 0.5
        ),
        f"angular_p90_deg_{prefix}": percentile(
            [r["angular_error_deg"] for r in rows], 0.9
        ),
        f"chordal_rmse_{prefix}": rmse([r["chordal_error"] for r in rows]),
    }


def write_summary(rows, path):
    valid_rows = subset(rows)
    cam0 = subset(rows, 0)
    cam1 = subset(rows, 1)
    summary = {
        "total_corner_count": len(valid_rows),
        "invalid_corner_count": len([r for r in rows if r.get("valid") != 1]),
        "cam0_corner_count": len(cam0),
        "cam1_corner_count": len(cam1),
    }
    summary.update(metric_block(valid_rows, "total"))
    summary.update(metric_block(cam0, "cam0"))
    summary.update(metric_block(cam1, "cam1"))
    fields = [
        "total_corner_count",
        "invalid_corner_count",
        "cam0_corner_count",
        "cam1_corner_count",
        "pixel_rmse_px_total",
        "pixel_rmse_px_cam0",
        "pixel_rmse_px_cam1",
        "angular_rmse_deg_total",
        "angular_rmse_deg_cam0",
        "angular_rmse_deg_cam1",
        "angular_median_deg_total",
        "angular_median_deg_cam0",
        "angular_median_deg_cam1",
        "angular_p90_deg_total",
        "angular_p90_deg_cam0",
        "angular_p90_deg_cam1",
        "chordal_rmse_total",
        "chordal_rmse_cam0",
        "chordal_rmse_cam1",
    ]
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow({k: fmt(summary.get(k)) for k in fields})
    return summary


def bucket_for(polar):
    for name, lo, hi in BUCKETS:
        if finite(polar) and polar >= lo and (polar < hi or math.isinf(hi)):
            return name, lo, hi
    return None


def bucket_stats(rows, path, low_count_threshold=30):
    valid_rows = subset(rows)
    global_pixel_median = percentile([r["pixel_error_px"] for r in valid_rows], 0.5)
    global_angular_p90 = percentile([r["angular_error_deg"] for r in valid_rows], 0.9)
    rows_by_bucket = {name: [] for name, _, _ in BUCKETS}
    for row in valid_rows:
        bucket = bucket_for(row["polar_angle_deg"])
        if bucket:
            rows_by_bucket[bucket[0]].append(row)

    output_rows = []
    for name, lo, hi in BUCKETS:
        items = rows_by_bucket[name]
        cam0 = [r for r in items if r["cam_id"] == 0]
        cam1 = [r for r in items if r["cam_id"] == 1]
        mismatch = [
            r
            for r in items
            if r["pixel_error_px"] < global_pixel_median
            and r["angular_error_deg"] > global_angular_p90
        ]
        regular_count = sum(1 for r in items if r.get("detection_source") == "regular")
        unknown_count = sum(
            1 for r in items if r.get("detection_source", "unknown") == "unknown"
        )
        output_rows.append(
            {
                "bucket_name": name,
                "polar_min_deg": lo,
                "polar_max_deg": hi if finite(hi) else "inf",
                "corner_count": len(items),
                "low_count": len(items) < low_count_threshold,
                "pixel_rmse_px": rmse([r["pixel_error_px"] for r in items]),
                "pixel_median_px": percentile([r["pixel_error_px"] for r in items], 0.5),
                "pixel_p90_px": percentile([r["pixel_error_px"] for r in items], 0.9),
                "angular_rmse_deg": rmse([r["angular_error_deg"] for r in items]),
                "angular_median_deg": percentile(
                    [r["angular_error_deg"] for r in items], 0.5
                ),
                "angular_p90_deg": percentile(
                    [r["angular_error_deg"] for r in items], 0.9
                ),
                "chordal_rmse": rmse([r["chordal_error"] for r in items]),
                "cam0_count": len(cam0),
                "cam1_count": len(cam1),
                "cam0_pixel_rmse_px": rmse([r["pixel_error_px"] for r in cam0]),
                "cam1_pixel_rmse_px": rmse([r["pixel_error_px"] for r in cam1]),
                "cam0_angular_rmse_deg": rmse([r["angular_error_deg"] for r in cam0]),
                "cam1_angular_rmse_deg": rmse([r["angular_error_deg"] for r in cam1]),
                "cam0_angular_median_deg": percentile(
                    [r["angular_error_deg"] for r in cam0], 0.5
                ),
                "cam1_angular_median_deg": percentile(
                    [r["angular_error_deg"] for r in cam1], 0.5
                ),
                "regular_count": regular_count,
                "rescued_count": sum(1 for r in items if r.get("is_rescued") == 1),
                "unknown_detection_count": unknown_count,
                "pixel_angular_mismatch_count": len(mismatch),
                "pixel_angular_mismatch_ratio": len(mismatch) / len(items)
                if items
                else 0.0,
            }
        )

    fields = list(output_rows[0].keys())
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in output_rows:
            writer.writerow({k: fmt(v) for k, v in row.items()})
    return output_rows


def scatter_by_cam(rows, x_key, y_key, xlabel, ylabel, path):
    valid_rows = subset(rows)
    plt.figure(figsize=(7, 5))
    for cam_id, label, color in [(0, "cam0", "#2764b8"), (1, "cam1", "#c9552d")]:
        pts = [r for r in valid_rows if r["cam_id"] == cam_id]
        plt.scatter(
            [r[x_key] for r in pts],
            [r[y_key] for r in pts],
            s=6,
            alpha=0.35,
            label=label,
            color=color,
            linewidths=0,
        )
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def write_figures(rows, bucket_rows, figures_dir):
    figures_dir.mkdir(parents=True, exist_ok=True)
    if not HAS_MATPLOTLIB:
        write_figures_pil(rows, bucket_rows, figures_dir)
        return
    valid_rows = subset(rows)
    scatter_by_cam(
        rows,
        "pixel_error_px",
        "angular_error_deg",
        "pixel_error_px",
        "angular_error_deg",
        figures_dir / "fig_01_pixel_vs_angular_scatter.png",
    )
    scatter_by_cam(
        rows,
        "polar_angle_deg",
        "angular_error_deg",
        "polar_angle_deg",
        "angular_error_deg",
        figures_dir / "fig_02_polar_vs_angular_scatter.png",
    )

    plt.figure(figsize=(7, 5))
    for cam_id, label, color in [(0, "cam0", "#2764b8"), (1, "cam1", "#c9552d")]:
        vals = [r["angular_error_deg"] for r in valid_rows if r["cam_id"] == cam_id]
        plt.hist(vals, bins=40, alpha=0.55, label=label, color=color)
    plt.xlabel("angular_error_deg")
    plt.ylabel("count")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "fig_03_cam0_cam1_angular_hist.png", dpi=180)
    plt.close()

    for group_key, filename, xlabel in [
        ("frame_id", "fig_04_frame_mean_angular_error_rank.png", "frame_id"),
        ("board_id", "fig_05_board_mean_angular_error_rank.png", "board_id"),
    ]:
        grouped = defaultdict(list)
        for row in valid_rows:
            grouped[row[group_key]].append(row["angular_error_deg"])
        ranked = sorted(
            ((key, sum(vals) / len(vals), len(vals)) for key, vals in grouped.items()),
            key=lambda item: item[1],
            reverse=True,
        )[:25]
        plt.figure(figsize=(9, 5))
        plt.bar([str(item[0]) for item in ranked], [item[1] for item in ranked])
        plt.xlabel(xlabel)
        plt.ylabel("mean angular_error_deg")
        plt.xticks(rotation=60, ha="right")
        plt.grid(axis="y", alpha=0.25)
        plt.tight_layout()
        plt.savefig(figures_dir / filename, dpi=180)
        plt.close()

    labels = [r["bucket_name"] for r in bucket_rows]
    for key, filename, ylabel in [
        ("pixel_rmse_px", "fig_06_bucket_pixel_rmse.png", "pixel_RMSE_px"),
        ("angular_rmse_deg", "fig_07_bucket_angular_rmse.png", "angular_RMSE_deg"),
        ("angular_p90_deg", "fig_08_bucket_angular_p90.png", "angular_p90_deg"),
        (
            "pixel_angular_mismatch_ratio",
            "fig_09_bucket_mismatch_ratio.png",
            "mismatch_ratio",
        ),
    ]:
        plt.figure(figsize=(7, 5))
        plt.bar(labels, [as_float(r[key]) for r in bucket_rows])
        plt.ylabel(ylabel)
        plt.xticks(rotation=30, ha="right")
        plt.grid(axis="y", alpha=0.25)
        plt.tight_layout()
        plt.savefig(figures_dir / filename, dpi=180)
        plt.close()

    x = range(len(labels))
    width = 0.38
    plt.figure(figsize=(7, 5))
    plt.bar(
        [i - width / 2 for i in x],
        [as_float(r["cam0_angular_rmse_deg"]) for r in bucket_rows],
        width,
        label="cam0",
    )
    plt.bar(
        [i + width / 2 for i in x],
        [as_float(r["cam1_angular_rmse_deg"]) for r in bucket_rows],
        width,
        label="cam1",
    )
    plt.ylabel("angular_RMSE_deg")
    plt.xticks(list(x), labels, rotation=30, ha="right")
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "fig_10_bucket_cam0_cam1_angular_rmse.png", dpi=180)
    plt.close()


def pil_font(size=16):
    try:
        return ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", size)
    except Exception:
        return ImageFont.load_default()


def draw_axes(draw, width, height, title, xlabel, ylabel):
    margin_l, margin_r, margin_t, margin_b = 78, 28, 42, 66
    x0, y0 = margin_l, height - margin_b
    x1, y1 = width - margin_r, margin_t
    draw.rectangle([x0, y1, x1, y0], outline=(70, 70, 70), width=1)
    font = pil_font(14)
    title_font = pil_font(18)
    draw.text((width / 2, 12), title, fill=(20, 20, 20), font=title_font, anchor="ma")
    draw.text((width / 2, height - 32), xlabel, fill=(20, 20, 20), font=font, anchor="ma")
    draw.text((18, height / 2), ylabel, fill=(20, 20, 20), font=font, anchor="mm")
    return x0, y0, x1, y1


def finite_range(values):
    clean = [v for v in values if finite(v)]
    if not clean:
        return 0.0, 1.0
    lo, hi = min(clean), max(clean)
    if lo == hi:
        return lo - 0.5, hi + 0.5
    pad = 0.04 * (hi - lo)
    return lo - pad, hi + pad


def map_point(x, y, x_min, x_max, y_min, y_max, box):
    x0, y0, x1, y1 = box
    px = x0 + (x - x_min) / (x_max - x_min) * (x1 - x0)
    py = y0 - (y - y_min) / (y_max - y_min) * (y0 - y1)
    return px, py


def pil_scatter(rows, x_key, y_key, xlabel, ylabel, path, title):
    width, height = 900, 620
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image, "RGBA")
    box = draw_axes(draw, width, height, title, xlabel, ylabel)
    pts = [r for r in subset(rows) if finite(r[x_key]) and finite(r[y_key])]
    x_min, x_max = finite_range([r[x_key] for r in pts])
    y_min, y_max = finite_range([r[y_key] for r in pts])
    colors = {0: (39, 100, 184, 90), 1: (201, 85, 45, 90)}
    for r in pts:
        x, y = map_point(r[x_key], r[y_key], x_min, x_max, y_min, y_max, box)
        c = colors.get(r["cam_id"], (80, 80, 80, 90))
        draw.ellipse([x - 2, y - 2, x + 2, y + 2], fill=c)
    image.save(path)


def pil_bar(labels, values, ylabel, path, title):
    width, height = 900, 620
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image, "RGBA")
    box = draw_axes(draw, width, height, title, "bucket", ylabel)
    x0, y0, x1, y1 = box
    clean = [v for v in values if finite(v)]
    max_v = max(clean) if clean else 1.0
    max_v = max(max_v, 1e-12)
    n = max(1, len(labels))
    gap = 12
    bar_w = max(8, ((x1 - x0) - gap * (n + 1)) / n)
    font = pil_font(12)
    for i, (label, value) in enumerate(zip(labels, values)):
        value = value if finite(value) else 0.0
        bx0 = x0 + gap + i * (bar_w + gap)
        bx1 = bx0 + bar_w
        by1 = y0
        by0 = y0 - value / max_v * (y0 - y1)
        draw.rectangle([bx0, by0, bx1, by1], fill=(50, 110, 170, 220))
        draw.text((bx0 + bar_w / 2, y0 + 10), label, fill=(20, 20, 20), font=font, anchor="ma")
    image.save(path)


def pil_hist(rows, path):
    vals_by_cam = {
        cam: [r["angular_error_deg"] for r in subset(rows) if r["cam_id"] == cam]
        for cam in [0, 1]
    }
    all_vals = vals_by_cam[0] + vals_by_cam[1]
    lo, hi = finite_range(all_vals)
    bins = 40
    width, height = 900, 620
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image, "RGBA")
    box = draw_axes(draw, width, height, "cam0/cam1 angular histogram", "angular_error_deg", "count")
    counts_by_cam = {}
    max_count = 1
    for cam, vals in vals_by_cam.items():
        counts = [0] * bins
        for v in vals:
            if finite(v):
                idx = min(bins - 1, max(0, int((v - lo) / (hi - lo) * bins)))
                counts[idx] += 1
        counts_by_cam[cam] = counts
        max_count = max(max_count, max(counts) if counts else 1)
    x0, y0, x1, y1 = box
    bin_w = (x1 - x0) / bins
    colors = {0: (39, 100, 184, 120), 1: (201, 85, 45, 120)}
    for cam, counts in counts_by_cam.items():
        for i, count in enumerate(counts):
            bx0 = x0 + i * bin_w
            bx1 = bx0 + bin_w * 0.9
            by0 = y0 - count / max_count * (y0 - y1)
            draw.rectangle([bx0, by0, bx1, y0], fill=colors[cam])
    image.save(path)


def pil_group_rank(rows, group_key, path, title):
    grouped = defaultdict(list)
    for row in subset(rows):
        grouped[row[group_key]].append(row["angular_error_deg"])
    ranked = sorted(
        ((str(key), sum(vals) / len(vals)) for key, vals in grouped.items()),
        key=lambda item: item[1],
        reverse=True,
    )[:25]
    pil_bar([r[0] for r in ranked], [r[1] for r in ranked], "mean angular_error_deg", path, title)


def pil_grouped_bar(labels, values0, values1, path):
    width, height = 900, 620
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image, "RGBA")
    box = draw_axes(draw, width, height, "bucket cam0/cam1 angular RMSE", "bucket", "angular_RMSE_deg")
    x0, y0, x1, y1 = box
    clean = [v for v in values0 + values1 if finite(v)]
    max_v = max(clean) if clean else 1.0
    n = max(1, len(labels))
    gap = 12
    group_w = max(16, ((x1 - x0) - gap * (n + 1)) / n)
    bar_w = group_w / 2.2
    font = pil_font(12)
    for i, label in enumerate(labels):
        base = x0 + gap + i * (group_w + gap)
        for j, (value, color) in enumerate(
            [(values0[i], (39, 100, 184, 220)), (values1[i], (201, 85, 45, 220))]
        ):
            value = value if finite(value) else 0.0
            bx0 = base + j * bar_w
            bx1 = bx0 + bar_w
            by0 = y0 - value / max_v * (y0 - y1)
            draw.rectangle([bx0, by0, bx1, y0], fill=color)
        draw.text((base + group_w / 2, y0 + 10), label, fill=(20, 20, 20), font=font, anchor="ma")
    image.save(path)


def write_figures_pil(rows, bucket_rows, figures_dir):
    pil_scatter(
        rows,
        "pixel_error_px",
        "angular_error_deg",
        "pixel_error_px",
        "angular_error_deg",
        figures_dir / "fig_01_pixel_vs_angular_scatter.png",
        "pixel vs angular error",
    )
    pil_scatter(
        rows,
        "polar_angle_deg",
        "angular_error_deg",
        "polar_angle_deg",
        "angular_error_deg",
        figures_dir / "fig_02_polar_vs_angular_scatter.png",
        "polar angle vs angular error",
    )
    pil_hist(rows, figures_dir / "fig_03_cam0_cam1_angular_hist.png")
    pil_group_rank(
        rows,
        "frame_id",
        figures_dir / "fig_04_frame_mean_angular_error_rank.png",
        "frame mean angular error rank",
    )
    pil_group_rank(
        rows,
        "board_id",
        figures_dir / "fig_05_board_mean_angular_error_rank.png",
        "board mean angular error rank",
    )
    labels = [r["bucket_name"] for r in bucket_rows]
    for key, filename, ylabel, title in [
        ("pixel_rmse_px", "fig_06_bucket_pixel_rmse.png", "pixel_RMSE_px", "bucket pixel RMSE"),
        ("angular_rmse_deg", "fig_07_bucket_angular_rmse.png", "angular_RMSE_deg", "bucket angular RMSE"),
        ("angular_p90_deg", "fig_08_bucket_angular_p90.png", "angular_p90_deg", "bucket angular p90"),
        (
            "pixel_angular_mismatch_ratio",
            "fig_09_bucket_mismatch_ratio.png",
            "mismatch_ratio",
            "bucket mismatch ratio",
        ),
    ]:
        pil_bar(labels, [as_float(r[key]) for r in bucket_rows], ylabel, figures_dir / filename, title)
    pil_grouped_bar(
        labels,
        [as_float(r["cam0_angular_rmse_deg"]) for r in bucket_rows],
        [as_float(r["cam1_angular_rmse_deg"]) for r in bucket_rows],
        figures_dir / "fig_10_bucket_cam0_cam1_angular_rmse.png",
    )


def write_readme(path, trace_csv, train_label, test_label):
    text = f"""# Stage6 Angular Fixed-K Diagnostic

Dataset: `{train_label} -> {test_label}`

This diagnostic uses the currently calibrated Double Sphere intrinsics as fixed intrinsics for cam0 and cam1.

It does not re-optimize K0/K1.

It does not modify Stage5 or Stage6 selection, rescue, or BA input observations.

It does not change T_1_0 or board poses; it only evaluates ray-space diagnostics from the current BA result.

`q_obs` comes from `unproject_DS(u_obs; K_fixed)` in the C++ `DoubleSphereCameraModel`.

`q_pred` comes from the current board pose / stereo extrinsic camera-frame point normalized to a unit ray.

The purpose is to check whether pixel residuals and angular residuals disagree in high-polar-angle / fisheye-edge regions.

If the high-polar-angle bucket shows clearly higher angular_RMSE or mismatch_ratio, the next experiment should be fixed-K final-only hybrid angular BA.

Corner trace: `{trace_csv.name}`
"""
    path.write_text(text)


def top_groups(rows, group_key, n=5):
    grouped = defaultdict(list)
    for row in subset(rows):
        grouped[row[group_key]].append(row["angular_error_deg"])
    ranked = sorted(
        ((key, sum(vals) / len(vals), len(vals)) for key, vals in grouped.items()),
        key=lambda item: item[1],
        reverse=True,
    )
    return ranked[:n]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-label", default="unknown_train")
    parser.add_argument("--test-label", default="unknown_test")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_csv = Path(args.trace_csv)
    rows = read_trace(trace_csv)
    summary = write_summary(rows, output_dir / "angular_diagnostic_summary.csv")
    bucket_rows = bucket_stats(rows, output_dir / "angular_bucket_summary.csv")
    write_figures(rows, bucket_rows, output_dir / "figures")
    write_readme(output_dir / "README.md", trace_csv, args.train_label, args.test_label)

    top_frames = top_groups(rows, "frame_id")
    top_boards = top_groups(rows, "board_id")
    highest_bucket = max(
        bucket_rows,
        key=lambda r: as_float(r["pixel_angular_mismatch_ratio"]),
    )
    print(f"output_dir: {output_dir}")
    print(f"total valid corners: {summary['total_corner_count']}")
    print(f"total invalid corners: {summary['invalid_corner_count']}")
    print(f"total pixel RMSE: {fmt(summary['pixel_rmse_px_total'])}")
    print(f"total angular RMSE: {fmt(summary['angular_rmse_deg_total'])} deg")
    print(f"cam0 angular RMSE: {fmt(summary['angular_rmse_deg_cam0'])} deg")
    print(f"cam1 angular RMSE: {fmt(summary['angular_rmse_deg_cam1'])} deg")
    print("highest angular-error frame top 5:")
    for frame_id, mean_value, count in top_frames:
        print(f"  frame {frame_id}: mean={fmt(mean_value)} deg, count={count}")
    print("highest angular-error board top 5:")
    for board_id, mean_value, count in top_boards:
        print(f"  board {board_id}: mean={fmt(mean_value)} deg, count={count}")
    print(
        "bucket with highest mismatch_ratio: "
        f"{highest_bucket['bucket_name']} "
        f"ratio={fmt(highest_bucket['pixel_angular_mismatch_ratio'])}"
    )


if __name__ == "__main__":
    main()
