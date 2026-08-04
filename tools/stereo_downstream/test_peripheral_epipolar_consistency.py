#!/usr/bin/env python3
"""Geometry and protocol checks for Peripheral Epipolar Consistency."""

import importlib.util
import sys
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).with_name("evaluate_peripheral_epipolar_consistency.py")
SPEC = importlib.util.spec_from_file_location("peripheral_epipolar", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def unit(vector: np.ndarray) -> np.ndarray:
    return vector / np.linalg.norm(vector)


def test_epipolar_angle_and_monotonic_perturbation() -> None:
    # p_1 = p_0 + t; a point seen by both cameras lies in both epipolar planes.
    system = type(
        "SyntheticStereo",
        (),
        {
            "rotation_cam1_cam0": np.eye(3),
            "translation_cam1_cam0": np.array([-0.20, 0.0, 0.0]),
        },
    )()
    point_cam0 = np.array([0.30, 0.15, 2.00])
    ray_left = unit(point_cam0).reshape(1, 3)
    ray_right = unit(point_cam0 + system.translation_cam1_cam0).reshape(1, 3)
    exact, valid = MODULE.symmetric_epipolar_angle_deg(system, ray_left, ray_right)
    assert valid[0] and exact[0] < 1e-10

    errors = []
    for vertical_shift in (0.002, 0.01, 0.05):
        perturbed = unit(ray_right[0] + np.array([0.0, vertical_shift, 0.0])).reshape(1, 3)
        error, valid = MODULE.symmetric_epipolar_angle_deg(system, ray_left, perturbed)
        assert valid[0]
        errors.append(error[0])
    assert errors[0] < errors[1] < errors[2]


def test_method_independent_binning() -> None:
    polar = [4.0, 44.0, 70.0, 92.0]
    expected = ["central_0_30", "middle_30_60", "peripheral_60_80", "outside_0_80"]
    assert [MODULE.region_name(value) for value in polar] == expected
    # The bin is derived from frozen raw sensor coordinates, not either camera model.
    frozen_id_region = [(17, 1, MODULE.region_name(value)) for value in polar]
    assert frozen_id_region == [(17, 1, name) for name in expected]


def test_residual_protocol_guard() -> None:
    MODULE.require_selection_residual("spherical_tangent", "spherical_tangent")
    try:
        MODULE.require_selection_residual("pixel", "spherical_tangent")
    except RuntimeError as error:
        assert "does not satisfy" in str(error)
    else:
        raise AssertionError("pixel bundle must be rejected for spherical_tangent protocol")


def test_metrics_audit() -> None:
    valid_rows = []
    for method in ("Kalibr", "Ours"):
        for region in ("all", *(name for name, _, _, _ in MODULE.REGIONS)):
            valid_rows.append(
                {
                    "method": method,
                    "region": region,
                    "frozen_match_count": 1,
                    "angular_valid_count": 1,
                    "p95_epipolar_angular_error_deg": 0.1,
                    "vertical_valid_count": 1,
                    "vertical_disparity_p95_px": 0.2,
                }
            )
    MODULE.validate_metrics(valid_rows)
    valid_rows[-1]["vertical_valid_count"] = 0
    try:
        MODULE.validate_metrics(valid_rows)
    except RuntimeError as error:
        assert "no valid rectified" in str(error)
    else:
        raise AssertionError("undefined regional vertical metric must be rejected")


if __name__ == "__main__":
    test_epipolar_angle_and_monotonic_perturbation()
    test_method_independent_binning()
    test_residual_protocol_guard()
    test_metrics_audit()
    print("peripheral epipolar consistency checks: OK")
