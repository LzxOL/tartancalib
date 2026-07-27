#!/usr/bin/env python3

from __future__ import annotations

import json
import math
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import run_ds_perturbation_sweep as sweep
import run_ds_w1_local_recovery_sweep as runner
import summarize_plot_ds_w1_local_recovery as summary


class PerturbationScalingTest(unittest.TestCase):
    def test_all_six_levels_are_exact(self) -> None:
        camera = sweep.Camera(
            -0.19160371906342305, 0.6148940464325672,
            1170.610850399456, 1169.9708627603625,
            2243.217561347974, 2274.6343942192116, family="ds-none",
        )
        direction = np.asarray([
            0.4823501909728892, 0.1786242099590995,
            0.6074610942764621, 0.6053224490451137,
            -0.002550287143638483, -0.0009656825871351518,
        ])
        direction /= np.linalg.norm(direction)
        mask = sweep.build_evaluation_mask(camera, 4512, 4512, 121)
        calibrated = runner.calibrate_levels(
            camera, direction, runner.DEFAULT_LEVELS, 4512, 4512, mask,
        )
        self.assertEqual([row["perturbation_level_deg"] for row in calibrated], list(runner.DEFAULT_LEVELS))
        for row in calibrated:
            self.assertLess(
                abs(row["initial_peripheral_ray_p95_deg"] - row["perturbation_level_deg"]),
                1e-9,
            )
        self.assertEqual(calibrated[0]["amplitude"], 0.0)


class StatisticalPrimitiveTest(unittest.TestCase):
    def test_hodges_lehmann_uses_walsh_averages(self) -> None:
        self.assertEqual(summary.hodges_lehmann(np.asarray([1.0, 2.0, 3.0])), 2.0)

    def test_bootstrap_is_reproducible(self) -> None:
        values = np.asarray([-1.0, 0.0, 2.0, 4.0])
        first = summary.bootstrap_ci(values, np.median, np.random.default_rng(9), 500)
        second = summary.bootstrap_ci(values, np.median, np.random.default_rng(9), 500)
        self.assertEqual(first, second)

    def test_directional_information_is_quadratic_form(self) -> None:
        fisher = np.diag([2.0, 3.0, 5.0])
        direction = np.asarray([1.0, 2.0, -1.0])
        self.assertEqual(summary.directional_information(fisher, direction), 19.0)

    def test_pair_counts_and_ties(self) -> None:
        rows = []
        pairs = ((1.0, 0.5), (0.5, 0.5), (0.25, 0.75))
        for seed, (outer, internal) in enumerate(pairs):
            rows.extend((
                {"perturbation_level_deg": 0.5, "seed": seed, "method": "Outer-only",
                 "peripheral_ray_p95_deg": outer, "solver_status": "converged"},
                {"perturbation_level_deg": 0.5, "seed": seed, "method": "Outer+Internal",
                 "peripheral_ray_p95_deg": internal, "solver_status": "converged"},
            ))
        result = summary.paired_statistics(pd.DataFrame(rows))[0]
        self.assertEqual(result["outer_internal_better_count"], 1)
        self.assertEqual(result["outer_only_better_count"], 1)
        self.assertEqual(result["tie_count"], 1)
        self.assertTrue(math.isfinite(result["wilcoxon_one_sided_p"]))


class ProtocolValidationTest(unittest.TestCase):
    def test_missing_columns_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "protocol_manifest.json").write_text(json.dumps({"levels_deg": [0.0]}))
            pd.DataFrame([{"run_key": "bad"}]).to_csv(root / "local_recovery_runs.csv", index=False)
            pd.DataFrame([{"bad": 1}]).to_csv(root / "pose_only_run_summary.csv", index=False)
            with self.assertRaisesRegex(RuntimeError, "missing columns"):
                summary.validate_sweep(root, allow_incomplete=True)

    def test_expected_paper_key_count(self) -> None:
        keys = {
            (level, seed, method)
            for level in runner.DEFAULT_LEVELS
            for seed in range(1, 101)
            for method in runner.METHODS
        }
        self.assertEqual(len(keys), 1200)


@unittest.skipUnless(os.environ.get("DS_W1_SWEEP_DIR"), "full sweep path not supplied")
class FullSweepAcceptanceTest(unittest.TestCase):
    def test_full_sweep_protocol(self) -> None:
        root = Path(os.environ["DS_W1_SWEEP_DIR"]).resolve()
        runs, poses, manifest = summary.validate_sweep(root, allow_incomplete=False)
        self.assertEqual(len(runs), 1200)
        self.assertEqual(runs.run_key.nunique(), 1200)
        self.assertEqual(len(poses), 1200)
        self.assertEqual(manifest["grid_size"], 121)


if __name__ == "__main__":
    unittest.main()
