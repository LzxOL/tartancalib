#!/usr/bin/env python3
"""Render training-side Stage5 internal point filter decisions with Pillow."""

import argparse
import csv
import html
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--filter-csv", required=True, type=Path)
    parser.add_argument("--outer-points-csv", required=True, type=Path)
    parser.add_argument("--image-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--thumbnail-width", type=int, default=480)
    return parser.parse_args()


def to_float(row, key):
    return float(row[key])


def image_for_label(image_dir, label):
    for extension in (".png", ".jpg", ".jpeg", ".bmp"):
        candidate = image_dir / f"{label}{extension}"
        if candidate.exists():
            return candidate
    return None


def filter_rows(path):
    rows = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                for key in ("observed_x", "observed_y", "predicted_x", "predicted_y", "residual_norm"):
                    row[key] = to_float(row, key)
                row["filtered"] = row["filtered"] == "1"
            except (KeyError, TypeError, ValueError):
                continue
            rows.append(row)
    return rows


def outer_rows(path):
    rows_by_frame = defaultdict(list)
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("point_type") != "outer":
                continue
            try:
                rows_by_frame[(row["frame_index"], row["frame_label"])].append(
                    (float(row["observed_x"]), float(row["observed_y"]))
                )
            except (KeyError, TypeError, ValueError):
                continue
    return rows_by_frame


def render(rows, outer, image_path, output_path):
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image, "RGBA")
    kept = 0
    rejected = 0
    for x, y in outer:
        draw.ellipse((x - 6, y - 6, x + 6, y + 6), outline=(0, 235, 255, 255), width=2)
    for row in rows:
        observed = (row["observed_x"], row["observed_y"])
        predicted = (row["predicted_x"], row["predicted_y"])
        if row["filtered"]:
            color = (255, 55, 55, 255)
            rejected += 1
        else:
            color = (255, 170, 0, 255)
            kept += 1
        draw.line([observed, predicted], fill=(255, 225, 0, 190), width=2)
        draw.ellipse((observed[0] - 4, observed[1] - 4, observed[0] + 4, observed[1] + 4), fill=color)
        draw.ellipse((predicted[0] - 4, predicted[1] - 4, predicted[0] + 4, predicted[1] + 4), outline=(0, 240, 90, 255), width=2)

    first = rows[0]
    banner = (
        f"training internal-point filter | frame {first['frame_index']} | "
        f"kept {kept} | filtered {rejected}"
    )
    legend = "outer cyan | kept internal orange | filtered internal red | prediction green | residual yellow"
    draw.rectangle((14, 14, min(2200, image.width - 14), 92), fill=(0, 0, 0, 190))
    draw.text((28, 28), banner, fill=(255, 255, 255, 255), stroke_width=1)
    draw.text((28, 58), legend, fill=(255, 255, 255, 255), stroke_width=1)
    image.save(output_path)
    return kept, rejected


def main():
    args = parse_args()
    filtered_by_frame = defaultdict(list)
    for row in filter_rows(args.filter_csv):
        filtered_by_frame[(row["frame_index"], row["frame_label"])].append(row)
    outer_by_frame = outer_rows(args.outer_points_csv)
    frame_dir = args.output_dir / "frames"
    thumb_dir = args.output_dir / "thumbnails"
    frame_dir.mkdir(parents=True, exist_ok=True)
    thumb_dir.mkdir(parents=True, exist_ok=True)

    summary = []
    for (frame_index, label), rows in sorted(filtered_by_frame.items(), key=lambda entry: int(entry[0][0])):
        image_path = image_for_label(args.image_dir, label)
        if image_path is None:
            summary.append((frame_index, label, "", 0, 0, "missing_image"))
            continue
        stem = f"frame_{int(frame_index):04d}_{label}_internal_filter"
        overlay_path = frame_dir / f"{stem}.png"
        kept, rejected = render(rows, outer_by_frame[(frame_index, label)], image_path, overlay_path)
        image = Image.open(overlay_path)
        ratio = args.thumbnail_width / image.width
        image.resize((args.thumbnail_width, round(image.height * ratio))).save(
            thumb_dir / f"{stem}.jpg", quality=88
        )
        summary.append((frame_index, label, stem, kept, rejected, "ok"))

    summary.sort(key=lambda row: -row[4])
    with (args.output_dir / "summary.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frame_index", "frame_label", "artifact_stem", "kept_count", "filtered_count", "status"])
        writer.writerows(summary)

    cards = []
    for frame_index, label, stem, kept, rejected, status in summary:
        if status == "ok":
            cards.append(
                "<article><a href='frames/{0}.png'><img src='thumbnails/{0}.jpg'></a>"
                "<p>frame {1}: kept {2}, filtered {3}</p></article>".format(
                    html.escape(stem), html.escape(str(frame_index)), kept, rejected
                )
            )
    page = """<!doctype html><meta charset='utf-8'><title>Internal Point Filter Overlays</title>
<style>body{font-family:Arial,sans-serif;margin:24px;background:#f4f6f8}.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(320px,1fr));gap:16px}article{background:white;border:1px solid #ccd3d9;padding:8px}img{width:100%;display:block}p{font-size:14px;margin:6px 2px}</style>
<h1>Training Internal-Point Filter Overlays</h1><p>Orange = retained; red = filtered. Ordered by filtered-point count.</p><div class='grid'>""" + "\n".join(cards) + "</div>"
    (args.output_dir / "index.html").write_text(page)
    print(f"rendered_frames={sum(row[-1] == 'ok' for row in summary)}")


if __name__ == "__main__":
    main()
