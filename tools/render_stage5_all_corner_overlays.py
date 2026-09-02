#!/usr/bin/env python3
"""Render complete Stage5 point overlays without requiring OpenCV."""

import argparse
import csv
import html
import math
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw


def parse_args():
    parser = argparse.ArgumentParser(
        description="Render all Stage5 training and holdout corner overlays."
    )
    parser.add_argument("--training-points", required=True, type=Path)
    parser.add_argument("--holdout-points", required=True, type=Path)
    parser.add_argument("--image-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--thumbnail-width", type=int, default=480)
    return parser.parse_args()


def read_rows(path):
    rows = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                row["observed_x"] = float(row["observed_x"])
                row["observed_y"] = float(row["observed_y"])
                row["predicted_x"] = float(row["predicted_x"])
                row["predicted_y"] = float(row["predicted_y"])
                row["residual_norm"] = float(row["residual_norm"])
            except (KeyError, TypeError, ValueError):
                continue
            rows.append(row)
    return rows


def image_for_label(image_dir, frame_label):
    for extension in (".png", ".jpg", ".jpeg", ".bmp"):
        candidate = image_dir / f"{frame_label}{extension}"
        if candidate.exists():
            return candidate
    return None


def included(row):
    return row.get("evaluation_included", "1") != "0"


def render_frame(rows, image_path, output_path):
    image = Image.open(image_path).convert("RGB")
    overlay = ImageDraw.Draw(image, "RGBA")
    included_residuals = []
    included_count = 0
    excluded_count = 0

    for row in rows:
        observed = (row["observed_x"], row["observed_y"])
        predicted = (row["predicted_x"], row["predicted_y"])
        is_included = included(row)
        is_outer = row.get("point_type") == "outer"
        radius = 6 if is_outer else 3
        line_color = (255, 218, 0, 210) if is_included else (255, 40, 40, 140)
        observed_color = (255, 145, 0, 255) if is_included else (255, 48, 48, 210)
        predicted_color = (0, 240, 85, 255) if is_included else (150, 150, 150, 200)
        overlay.line([observed, predicted], fill=line_color, width=2)
        overlay.ellipse(
            (observed[0] - radius, observed[1] - radius,
             observed[0] + radius, observed[1] + radius),
            fill=observed_color,
        )
        overlay.ellipse(
            (predicted[0] - radius, predicted[1] - radius,
             predicted[0] + radius, predicted[1] + radius),
            outline=predicted_color,
            width=2,
        )
        if is_included:
            included_residuals.append(row["residual_norm"])
            included_count += 1
        else:
            excluded_count += 1

    rmse = math.sqrt(sum(value * value for value in included_residuals) /
                     len(included_residuals)) if included_residuals else 0.0
    first = rows[0]
    header = (
        f"{first['split']} | frame {first['frame_index']} | {first['frame_label']} | "
        f"RMSE {rmse:.3f}px | included {included_count} | excluded {excluded_count}"
    )
    legend = "observed orange | projected green | residual yellow | excluded red/gray"
    overlay.rectangle((14, 14, min(image.width - 14, 2080), 92), fill=(0, 0, 0, 185))
    overlay.text((28, 28), header, fill=(255, 255, 255, 255), stroke_width=1)
    overlay.text((28, 58), legend, fill=(255, 255, 255, 255), stroke_width=1)
    image.save(output_path)
    return rmse, included_count, excluded_count


def main():
    args = parse_args()
    output_dir = args.output_dir
    full_dir = output_dir / "frames"
    thumb_dir = output_dir / "thumbnails"
    full_dir.mkdir(parents=True, exist_ok=True)
    thumb_dir.mkdir(parents=True, exist_ok=True)

    groups = defaultdict(list)
    for row in read_rows(args.training_points) + read_rows(args.holdout_points):
        key = (row["split"], row["frame_index"], row["frame_label"])
        groups[key].append(row)

    summary = []
    for (split, frame_index, frame_label), rows in sorted(
        groups.items(), key=lambda item: (item[0][0], int(item[0][1]))
    ):
        image_path = image_for_label(args.image_dir, frame_label)
        if image_path is None:
            summary.append((split, frame_index, frame_label, "", 0, 0, 0, "missing_image"))
            continue
        stem = f"{split}_frame_{int(frame_index):04d}_{frame_label}"
        rendered_path = full_dir / f"{stem}.png"
        rmse, included_count, excluded_count = render_frame(rows, image_path, rendered_path)
        thumbnail = Image.open(rendered_path)
        ratio = args.thumbnail_width / thumbnail.width
        thumbnail = thumbnail.resize((args.thumbnail_width, round(thumbnail.height * ratio)))
        thumbnail.save(thumb_dir / f"{stem}.jpg", quality=88)
        summary.append((split, frame_index, frame_label, stem, rmse, included_count, excluded_count, "ok"))

    summary.sort(key=lambda row: (row[0], -row[4]))
    with (output_dir / "overlay_summary.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "split", "frame_index", "frame_label", "artifact_stem", "included_rmse",
            "included_point_count", "excluded_point_count", "status",
        ])
        writer.writerows(summary)

    cards = []
    for split, frame_index, frame_label, stem, rmse, include_count, exclude_count, status in summary:
        if status != "ok":
            continue
        cards.append(
            "<article><a href='frames/{0}.png'><img src='thumbnails/{0}.jpg'></a>"
            "<p><b>{1}</b> frame {2}: {3:.3f}px</p>"
            "<p>{4} included, {5} excluded</p></article>".format(
                html.escape(stem), html.escape(split), html.escape(str(frame_index)),
                rmse, include_count, exclude_count
            )
        )
    page = """<!doctype html><meta charset='utf-8'><title>Stage5 Corner Overlays</title>
<style>body{font-family:Arial,sans-serif;margin:24px;background:#f4f6f8}h1{font-size:22px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(320px,1fr));gap:16px}
article{background:white;border:1px solid #ccd3d9;padding:8px}img{width:100%;display:block}
p{margin:6px 2px;font-size:14px}</style><h1>Stage5 Corner Overlays</h1>
<p>Observed: orange; projected: green; residual: yellow; excluded: red/gray. Frames are ordered by split then descending RMSE.</p>
<div class='grid'>""" + "\n".join(cards) + "</div>"
    (output_dir / "index.html").write_text(page)
    print(f"rendered_frames={sum(row[-1] == 'ok' for row in summary)}")
    print(f"output_dir={output_dir}")


if __name__ == "__main__":
    main()
