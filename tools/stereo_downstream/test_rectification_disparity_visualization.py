#!/usr/bin/env python3
"""Dependency-free geometry checks for the DS downstream visualization tool."""

import importlib.util
import sys
import tempfile
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).with_name("run_rectification_disparity_visualization.py")
SPEC = importlib.util.spec_from_file_location("stereo_downstream", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_roundtrip() -> None:
    camera = MODULE.load_ds_camera(MODULE.DEFAULT_OURS_LEFT, "test")
    pixels = np.array([[2200.0, 2200.0], [1200.0, 1600.0], [3100.0, 2900.0]])
    rays, valid = MODULE.ds_unproject(camera, pixels)
    projected, visible = MODULE.ds_project(camera, rays)
    assert np.all(valid & visible)
    assert np.max(np.linalg.norm(projected - pixels, axis=1)) < 1e-7


def test_invalid_and_vertical_offset() -> None:
    camera = MODULE.load_ds_camera(MODULE.DEFAULT_OURS_LEFT, "test")
    rays, valid = MODULE.ds_unproject(camera, np.array([[-1e6, -1e6], [camera.cx, camera.cy]]))
    assert not valid[0] and valid[1]
    projected, visible = MODULE.ds_project(camera, np.array([[0.0, 0.0, -1.0], [0.0, 0.0, 1.0]]))
    assert not visible[0] and visible[1]
    baseline = np.array([0.2, 0.5, 0.7, 0.9])
    assert MODULE.percentile(baseline + 1.0, 95.0) > MODULE.percentile(baseline, 95.0)
    assert np.isfinite(rays[1]).all() and np.isfinite(projected[1]).all()


def test_timestamp_pairing() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        left, right = root / "left", root / "right"
        left.mkdir()
        right.mkdir()
        (left / "000001_left_1000000000_mono8.png").touch()
        (left / "000002_left_2000000000_mono8.png").touch()
        (right / "000101_right_1000000500_mono8.png").touch()
        (right / "000102_right_2100000000_mono8.png").touch()
        pairs = MODULE.list_pairs(left, right, 1_000)
        assert len(pairs) == 1
        assert pairs[0].frame_id == 1
        assert pairs[0].timestamp_delta_ns == 500


if __name__ == "__main__":
    test_roundtrip()
    test_invalid_and_vertical_offset()
    test_timestamp_pairing()
    print("stereo downstream geometry checks: OK")
