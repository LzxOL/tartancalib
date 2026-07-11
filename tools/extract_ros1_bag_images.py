#!/usr/bin/env python3
"""Extract sensor_msgs/Image frames from a ROS1 bag by sequential chunk scan.

This intentionally does not depend on the bag index, so it can recover images
from bags whose index section is damaged.
"""

from __future__ import annotations

import argparse
import struct
from pathlib import Path

import cv2
import numpy as np


OP_MSGDATA = b"\x02"
OP_BAGHEADER = b"\x03"
OP_CHUNK = b"\x05"
OP_CONNECTION = b"\x07"


def parse_record_from_file(handle):
    start = handle.tell()
    header_len_raw = handle.read(4)
    if not header_len_raw:
        return None
    if len(header_len_raw) != 4:
        raise EOFError(f"short record header length at {start}")
    header_len = struct.unpack("<I", header_len_raw)[0]
    header_raw = handle.read(header_len)
    fields = parse_fields(header_raw)
    data_len_raw = handle.read(4)
    if len(data_len_raw) != 4:
        raise EOFError(f"short data length at {handle.tell()}")
    data_len = struct.unpack("<I", data_len_raw)[0]
    data = handle.read(data_len)
    if len(data) != data_len:
        raise EOFError(f"short data at {handle.tell()}")
    return start, fields, data


def parse_record_from_buffer(buffer: bytes, offset: int):
    header_len = struct.unpack_from("<I", buffer, offset)[0]
    offset += 4
    fields = parse_fields(buffer[offset : offset + header_len])
    offset += header_len
    data_len = struct.unpack_from("<I", buffer, offset)[0]
    offset += 4
    data = buffer[offset : offset + data_len]
    offset += data_len
    return fields, data, offset


def parse_fields(raw: bytes):
    fields = {}
    offset = 0
    while offset < len(raw):
        field_len = struct.unpack_from("<I", raw, offset)[0]
        offset += 4
        field = raw[offset : offset + field_len]
        offset += field_len
        key, value = field.split(b"=", 1)
        fields[key.decode("utf-8", errors="replace")] = value
    return fields


def read_u32(data: bytes, offset: int):
    return struct.unpack_from("<I", data, offset)[0], offset + 4


def parse_ros_string(data: bytes, offset: int):
    length, offset = read_u32(data, offset)
    value = data[offset : offset + length].decode("utf-8", errors="replace")
    return value, offset + length


def parse_sensor_msgs_image(data: bytes):
    offset = 0
    seq, offset = read_u32(data, offset)
    secs, offset = read_u32(data, offset)
    nsecs, offset = read_u32(data, offset)
    frame_id, offset = parse_ros_string(data, offset)
    height, offset = read_u32(data, offset)
    width, offset = read_u32(data, offset)
    encoding, offset = parse_ros_string(data, offset)
    is_bigendian = data[offset]
    offset += 1
    step, offset = read_u32(data, offset)
    image_data_len, offset = read_u32(data, offset)
    image_data = data[offset : offset + image_data_len]
    if len(image_data) != image_data_len:
        raise ValueError("sensor_msgs/Image data is truncated")
    return {
        "seq": seq,
        "secs": secs,
        "nsecs": nsecs,
        "frame_id": frame_id,
        "height": height,
        "width": width,
        "encoding": encoding,
        "is_bigendian": is_bigendian,
        "step": step,
        "data": image_data,
    }


def image_to_array(message):
    height = message["height"]
    width = message["width"]
    step = message["step"]
    encoding = message["encoding"].lower()
    raw = message["data"]

    if encoding in ("mono8", "8uc1"):
        array = np.frombuffer(raw, dtype=np.uint8).reshape(height, step)
        return array[:, :width]
    if encoding in ("bgr8", "rgb8"):
        array = np.frombuffer(raw, dtype=np.uint8).reshape(height, step)
        array = array[:, : width * 3].reshape(height, width, 3)
        if encoding == "rgb8":
            array = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
        return array
    if encoding in ("mono16", "16uc1"):
        dtype = ">u2" if message["is_bigendian"] else "<u2"
        array = np.frombuffer(raw, dtype=dtype).reshape(height, step // 2)
        return array[:, :width]
    raise ValueError(f"unsupported image encoding: {message['encoding']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bag", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--topic", default="")
    parser.add_argument("--prefix", default="")
    parser.add_argument("--format", choices=("png", "jpg"), default="png")
    args = parser.parse_args()

    bag_path = Path(args.bag)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    connections = {}
    count = 0
    topic_name = args.topic
    with bag_path.open("rb") as handle:
        magic = handle.readline()
        if not magic.startswith(b"#ROSBAG V2.0"):
            raise RuntimeError(f"unsupported bag magic: {magic!r}")

        while True:
            try:
                record = parse_record_from_file(handle)
            except EOFError as error:
                print(f"warning: stopped at truncated tail record: {error}")
                break
            if record is None:
                break
            _, fields, data = record
            op = fields.get("op")
            if op == OP_BAGHEADER:
                continue
            if op != OP_CHUNK:
                continue
            compression = fields.get("compression", b"none").decode()
            if compression != "none":
                raise RuntimeError(
                    f"unsupported chunk compression {compression!r}; "
                    "use rosbags/ROS to decompress this bag"
                )
            offset = 0
            while offset < len(data):
                inner_fields, inner_data, offset = parse_record_from_buffer(
                    data, offset
                )
                inner_op = inner_fields.get("op")
                if inner_op == OP_CONNECTION:
                    conn_id = struct.unpack("<I", inner_fields["conn"])[0]
                    topic = inner_fields["topic"].decode("utf-8", errors="replace")
                    connections[conn_id] = topic
                    if not topic_name:
                        topic_name = topic
                    continue
                if inner_op != OP_MSGDATA:
                    continue
                conn_id = struct.unpack("<I", inner_fields["conn"])[0]
                topic = connections.get(conn_id, "")
                if topic_name and topic != topic_name:
                    continue
                message = parse_sensor_msgs_image(inner_data)
                image = image_to_array(message)
                timestamp_ns = message["secs"] * 1_000_000_000 + message["nsecs"]
                safe_topic = topic.strip("/").replace("/", "_") or "image"
                prefix = args.prefix or safe_topic
                filename = (
                    f"{prefix}_{count:06d}_{timestamp_ns}.{args.format}"
                )
                path = output_dir / filename
                if not cv2.imwrite(str(path), image):
                    raise RuntimeError(f"failed to write {path}")
                count += 1
                if count % 10 == 0:
                    print(f"exported {count} images...", flush=True)

    print(f"topic: {topic_name}")
    print(f"exported_images: {count}")
    print(f"output_dir: {output_dir}")


if __name__ == "__main__":
    main()
