#!/usr/bin/env python3
"""Create a ROS1 bag with sensor_msgs/Image messages from image files."""

from __future__ import annotations

import argparse
import re
import struct
from pathlib import Path

import cv2
from rosbags.rosbag1 import Writer
from rosbags.typesys import Stores, get_typestore


def ros_string(value: str) -> bytes:
    data = value.encode("utf-8")
    return struct.pack("<I", len(data)) + data


def serialize_sensor_msgs_image(
    image,
    *,
    seq: int,
    stamp_ns: int,
    frame_id: str,
    encoding: str,
) -> bytes:
    if image.ndim == 2:
        height, width = image.shape
        step = width * image.dtype.itemsize
        payload = image.tobytes(order="C")
    elif image.ndim == 3 and image.shape[2] == 3:
        height, width = image.shape[:2]
        step = width * 3 * image.dtype.itemsize
        payload = image.tobytes(order="C")
    else:
        raise ValueError(f"unsupported image shape: {image.shape}")
    secs = stamp_ns // 1_000_000_000
    nsecs = stamp_ns % 1_000_000_000
    return b"".join(
        [
            struct.pack("<I", seq),
            struct.pack("<II", secs, nsecs),
            ros_string(frame_id),
            struct.pack("<II", height, width),
            ros_string(encoding),
            struct.pack("<B", 0),
            struct.pack("<I", step),
            struct.pack("<I", len(payload)),
            payload,
        ]
    )


def timestamp_from_name(path: Path, fallback_index: int, fallback_start_ns: int):
    match = re.search(r"_(\d{6})_(\d+)\.[^.]+$", path.name)
    if match:
        return int(match.group(2))
    return fallback_start_ns + fallback_index * 100_000_000


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--output-bag", required=True)
    parser.add_argument("--topic", default="/vimbax_camera_37086/image_raw")
    parser.add_argument("--frame-id", default="vimbax_camera_DEV_1AB22C049FF1")
    parser.add_argument("--encoding", default="mono8")
    parser.add_argument("--glob", default="*.png")
    args = parser.parse_args()

    image_dir = Path(args.image_dir)
    output_bag = Path(args.output_bag)
    paths = sorted(
        p for p in image_dir.glob(args.glob) if p.is_file() and not p.name.startswith(".")
    )
    if not paths:
        raise RuntimeError(f"no images matched {image_dir / args.glob}")
    if output_bag.exists():
        output_bag.unlink()

    typestore = get_typestore(Stores.ROS1_NOETIC)
    with Writer(output_bag) as writer:
        connection = writer.add_connection(
            args.topic,
            "sensor_msgs/msg/Image",
            typestore=typestore,
            callerid="/images_to_ros1_bag",
        )
        for index, path in enumerate(paths):
            image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if image is None:
                raise RuntimeError(f"failed to read image: {path}")
            if args.encoding.lower() == "mono8":
                if image.ndim == 3:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                if image.dtype.name != "uint8":
                    raise RuntimeError(f"mono8 requires uint8 image: {path}")
            elif args.encoding.lower() == "bgr8":
                if image.ndim == 2:
                    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                if image.dtype.name != "uint8":
                    raise RuntimeError(f"bgr8 requires uint8 image: {path}")
            else:
                raise RuntimeError(f"unsupported encoding: {args.encoding}")

            stamp_ns = timestamp_from_name(path, index, 0)
            data = serialize_sensor_msgs_image(
                image,
                seq=index,
                stamp_ns=stamp_ns,
                frame_id=args.frame_id,
                encoding=args.encoding,
            )
            writer.write(connection, stamp_ns, data)
            if (index + 1) % 25 == 0:
                print(f"written {index + 1} images...", flush=True)

    print(f"topic: {args.topic}")
    print(f"images: {len(paths)}")
    print(f"output_bag: {output_bag}")


if __name__ == "__main__":
    main()
