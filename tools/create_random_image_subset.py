#!/usr/bin/env python3
"""Create a reproducible random image subset as a directory of symlinks."""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    image_dir = args.image_dir.resolve()
    output = args.output.resolve()
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {image_dir}")
    if args.count <= 0:
        raise ValueError("--count must be positive")

    images = sorted(
        path for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if len(images) < args.count:
        raise ValueError(
            f"Requested {args.count} images, but only {len(images)} are available."
        )
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output already exists and is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    selected = sorted(random.Random(args.seed).sample(images, args.count))
    for image in selected:
        os.symlink(image, output / image.name)

    manifest = {
        "source_image_dir": str(image_dir),
        "source_image_count": len(images),
        "selected_image_count": len(selected),
        "sampling": "uniform_without_replacement",
        "seed": args.seed,
        "selected_images": [image.name for image in selected],
    }
    (output / "subset_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    (output / "selected_images.txt").write_text(
        "\n".join(image.name for image in selected) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
