#!/usr/bin/env python3
"""Paired DS observability and intrinsic-recovery experiment on selection prefixes.

The prefix schedule is read from a normal Stage5 Outer-only selection log.  It
is frozen before any information or recovery result is computed: both branches
use the same group prefix, truth scene, initial W1 perturbation, and Gaussian
noise.  Only frozen internal residuals distinguish Outer+Internal.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares
from scipy.stats import wilcoxon

import run_ds_perturbation_sweep as sweep
import run_ds_weak_mode_perturbation as weak


REPO = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = REPO / "result_may/ds_semi_synthetic_20260720_right_all_fixed_layout/reference/clean"
BUDGETS = (10, 20, 40, 60, 80, 100)
METHODS = ("Outer-only", "Outer+Internal")
COLORS = {"Outer-only": "#C44E52", "Outer+Internal": "#4C72B0"}
BOOTSTRAP_SEED = 20260721


def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--selection-decisions", type=Path,
        default=DEFAULT_ROOT / "outer_only/trial_backend_frame_board_selection_decisions.csv",
    )
    parser.add_argument(
        "--reference-scene", type=Path,
        default=DEFAULT_ROOT / "outer_only/final_persistent_backend_scene.txt",
    )
    parser.add_argument(
        "--training-points", type=Path,
        default=DEFAULT_ROOT / "outer_internal/backend_training_points.csv",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--budgets", default=",".join(map(str, BUDGETS)))
    parser.add_argument("--seed-start", type=int, default=1)
    parser.add_argument("--seed-count", type=int, default=50)
    parser.add_argument("--noise-sigma-px", type=float, default=0.25)
    parser.add_argument("--initial-peripheral-ray-p95-deg", type=float, default=0.5)
    parser.add_argument("--grid-size", type=int, default=181)
    parser.add_argument("--coverage-grid-size", type=int, default=20)
    parser.add_argument("--width", type=int, default=4512)
    parser.add_argument("--height", type=int, default=4512)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def digest(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def digest_file(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"Refusing to write empty CSV: {path}")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: f"{value:.15g}" if isinstance(value, float) else value
                             for key, value in row.items()})


def parse_budgets(raw: str) -> tuple[int, ...]:
    values = tuple(sorted({int(value.strip()) for value in raw.split(",") if value.strip()}))
    if not values or values[-1] != 100 or any(value <= 0 or value > 100 for value in values):
        raise ValueError("--budgets must be positive percentages and include 100")
    return values


def schedule_from_decisions(path: Path) -> list[tuple[int, int]]:
    """Read seed groups first, then committed incremental groups in log order."""
    groups: dict[tuple[int, int], tuple[int, int, int, int]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            frame, board = int(row["frame_index"]), int(row["board_id"])
            order = int(row.get("persistent_incremental_attempt_order") or -1)
            committed = row.get("persistent_incremental_commit_state", "") == "committed"
            accepted = row.get("persistent_incremental_batch_accepted", "") == "1"
            seed = order < 0
            if not seed and not (committed and accepted):
                continue
            key = (frame, board)
            rank = (0, 0, frame, board) if seed else (1, order, frame, board)
            if key not in groups or rank < groups[key]:
                groups[key] = rank
    ordered = [key for key, _ in sorted(groups.items(), key=lambda item: item[1])]
    if len(ordered) < 10:
        raise RuntimeError("Selection log did not contain enough accepted frame-board groups")
    return ordered


def load_rows(path: Path, scene: weak.Scene) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for source_index, row in enumerate(csv.DictReader(handle)):
            frame, board = int(row["frame_index"]), int(row["board_id"])
            if frame not in scene.frames or board not in scene.boards:
                continue
            local = np.asarray([float(row["target_x"]), float(row["target_y"]),
                                float(row["target_z"]), 1.0])
            rows.append({
                "source_index": source_index,
                "frame": frame,
                "board": board,
                "point_type": row["point_type"],
                "point": local,
                "camera_point": (scene.frames[frame] @ scene.boards[board] @ local)[:3],
            })
    if not rows:
        raise RuntimeError(f"No usable rows in {path}")
    return rows


def select_rows(rows: list[dict[str, Any]], keys: set[tuple[int, int]], method: str) -> list[dict[str, Any]]:
    allowed = {"outer"} if method == "Outer-only" else {"outer", "internal"}
    selected = [row for row in rows if (row["frame"], row["board"]) in keys and row["point_type"] in allowed]
    if not selected or not any(row["point_type"] == "outer" for row in selected):
        raise RuntimeError("Prefix does not contain outer observations")
    return selected


def fisher(scene: weak.Scene, rows: list[dict[str, Any]], width: int, height: int,
           inverse_variance: float) -> tuple[np.ndarray, dict[str, Any]]:
    # All synthetic pixels have the same sigma, so the weighted Schur Fisher is
    # exactly inverse_variance times the existing unit-weight implementation.
    points = [{"frame": row["frame"], "board": row["board"], "point": row["point"]} for row in rows]
    _, unit, audit = weak.compute_weak_modes(scene, points, width, height)
    return unit * inverse_variance, audit


def eig_summary(matrix: np.ndarray) -> tuple[np.ndarray, int, float, float]:
    values = np.maximum(np.linalg.eigvalsh(0.5 * (matrix + matrix.T)), 0.0)
    tolerance = max(1e-12, float(values[-1]) * 1e-12)
    positive = values[values > tolerance]
    return values, int(len(positive)), (float(positive[0]) if len(positive) else math.nan), (
        float(math.log10(positive[-1] / positive[0])) if len(positive) else math.inf)


def coverage(scene: weak.Scene, rows: list[dict[str, Any]], width: int, height: int, grid: int) -> float:
    points = np.asarray([row["camera_point"] for row in rows])
    pixels, valid = weak.project_ds(scene.camera, points)
    pixels = pixels[valid]
    x = np.clip((pixels[:, 0] * grid / width).astype(int), 0, grid - 1)
    y = np.clip((pixels[:, 1] * grid / height).astype(int), 0, grid - 1)
    return float(len(set(zip(x.tolist(), y.tolist()))) / (grid * grid))


def camera_values(camera: sweep.Camera, width: int, height: int) -> np.ndarray:
    return weak.camera_to_coordinates(camera, width, height)


def recover(initial: sweep.Camera, truth: sweep.Camera, rows: list[dict[str, Any]],
            noise: np.ndarray, width: int, height: int, mask: sweep.EvaluationMask) -> dict[str, Any]:
    points = np.asarray([row["camera_point"] for row in rows])
    clean, valid = weak.project_ds(truth, points)
    if not np.all(valid):
        raise RuntimeError("Truth projection invalid for frozen prefix observations")
    observed = clean + noise[[row["source_index"] for row in rows]]
    lower = np.asarray([-0.95, 0.01, math.log(400.0), math.log(400.0), 0.2, 0.2])
    upper = np.asarray([4.0, 0.99, math.log(6000.0), math.log(6000.0), 0.8, 0.8])

    def residual(values: np.ndarray) -> np.ndarray:
        candidate = weak.camera_from_coordinates(values, width, height)
        projected, candidate_valid = weak.project_ds(candidate, points)
        if not np.all(candidate_valid):
            return np.full(observed.size, 1e6)
        return (projected - observed).reshape(-1)

    try:
        result = least_squares(residual, np.clip(camera_values(initial, width, height), lower + 1e-8, upper - 1e-8),
                               bounds=(lower, upper), method="trf", loss="linear", x_scale="jac",
                               max_nfev=250, ftol=1e-12, xtol=1e-12, gtol=1e-12)
        final = weak.camera_from_coordinates(result.x, width, height)
        metrics = sweep.ray_metrics(mask, final)
        valid_model = weak.valid_camera(final) and float(metrics["valid_grid_ratio"]) >= 0.99
        return {
            "solver_status": "converged" if result.success and valid_model else "failed",
            "iterations": int(result.nfev), "final_xi": final.xi, "final_alpha": final.alpha,
            "final_fu": final.fu, "final_fv": final.fv, "final_cu": final.cu, "final_cv": final.cv,
            "xi_absolute_error": abs(final.xi - truth.xi),
            "mean_focal_relative_error": 0.5 * (abs(final.fu / truth.fu - 1.0) + abs(final.fv / truth.fv - 1.0)),
            "final_full_ray_p95_deg": float(metrics["full_ray_p95_deg"]),
            "final_peripheral_ray_p95_deg": float(metrics["peripheral_ray_p95_deg"]),
            "valid_grid_ratio": float(metrics["valid_grid_ratio"]),
        }
    except Exception as error:
        return {"solver_status": "failed", "iterations": 0, "failure_reason": f"{type(error).__name__}: {error}",
                "final_full_ray_p95_deg": math.nan, "final_peripheral_ray_p95_deg": math.nan,
                "xi_absolute_error": math.nan, "mean_focal_relative_error": math.nan, "valid_grid_ratio": 0.0}


def q(values: np.ndarray, percentile: float) -> float:
    return float(np.percentile(values, percentile)) if len(values) else math.nan


def summary_rows(observability: list[dict[str, Any]], recovery: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for budget in sorted({int(row["budget_percent"]) for row in recovery}):
        for method in METHODS:
            record: dict[str, Any] = {"budget_percent": budget, "method": method}
            info = next(row for row in observability if row["budget_percent"] == budget and row["method"] == method)
            record.update({key: info[key] for key in info if key not in {"method", "budget_percent"}})
            # The Fisher calculation has no random input.  Keep explicit zero-width
            # IQR fields so the plotting and tabular protocol remains uniform.
            for metric in ("raw_w1_information", "smallest_positive_eigenvalue", "rank",
                           "log10_condition_number", "occupied_grid_coverage",
                           "normalized_w1_information_total_effective_weight",
                           "normalized_w1_information_frame_board_group_weight"):
                record[f"{metric}_median"] = info[metric]
                record[f"{metric}_q25"] = info[metric]
                record[f"{metric}_q75"] = info[metric]
            group = [row for row in recovery if row["budget_percent"] == budget and row["method"] == method]
            for metric in ("final_full_ray_p95_deg", "final_peripheral_ray_p95_deg", "xi_absolute_error", "mean_focal_relative_error", "valid_grid_ratio"):
                values = np.asarray([float(row[metric]) for row in group if math.isfinite(float(row[metric]))])
                record[f"{metric}_median"] = q(values, 50)
                record[f"{metric}_q25"] = q(values, 25)
                record[f"{metric}_q75"] = q(values, 75)
            record["solver_success_rate"] = float(np.mean([row["solver_status"] == "converged" for row in group]))
            rows.append(record)
    return rows


def bootstrap_median(values: np.ndarray, rng: np.random.Generator, count: int = 10000) -> tuple[float, float]:
    if not len(values):
        return math.nan, math.nan
    estimates = np.asarray([np.median(values[rng.integers(0, len(values), len(values))]) for _ in range(count)])
    return q(estimates, 2.5), q(estimates, 97.5)


def paired_statistics(recovery: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    out: list[dict[str, Any]] = []
    for budget in sorted({int(row["budget_percent"]) for row in recovery}):
        by_seed: dict[int, dict[str, dict[str, Any]]] = {}
        for row in recovery:
            if int(row["budget_percent"]) == budget:
                by_seed.setdefault(int(row["seed"]), {})[row["method"]] = row
        pairs = [item for item in by_seed.values() if set(item) == set(METHODS) and
                 all(item[name]["solver_status"] == "converged" for name in METHODS)]
        delta = np.asarray([float(item["Outer-only"]["final_peripheral_ray_p95_deg"]) -
                            float(item["Outer+Internal"]["final_peripheral_ray_p95_deg"]) for item in pairs])
        pvalue = (float(wilcoxon(delta, alternative="greater", zero_method="wilcox").pvalue)
                  if len(delta) and np.any(np.abs(delta) > 1e-14) else 1.0)
        low, high = bootstrap_median(delta, rng)
        out.append({"budget_percent": budget, "complete_pair_count": len(delta),
                    "paired_median_improvement_peripheral_ray_p95_deg": q(delta, 50),
                    "bootstrap_ci95_low_deg": low, "bootstrap_ci95_high_deg": high,
                    "wilcoxon_one_sided_p": pvalue,
                    "outer_internal_better_count": int(np.count_nonzero(delta > 0.0)),
                    "outer_only_better_count": int(np.count_nonzero(delta < 0.0))})
    return out


def error_budget_auc_rows(recovery: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Integrate each seed before aggregation; no synthetic zero-budget point."""
    runs: list[dict[str, Any]] = []
    for seed in sorted({int(row["seed"]) for row in recovery}):
        for method in METHODS:
            curve = sorted([row for row in recovery if int(row["seed"]) == seed and row["method"] == method],
                           key=lambda row: row["budget_percent"])
            x = np.asarray([float(row["budget_percent"]) / 100.0 for row in curve])
            for metric, short_name in (("final_full_ray_p95_deg", "full_field"),
                                       ("final_peripheral_ray_p95_deg", "peripheral")):
                y = np.asarray([float(row[metric]) for row in curve])
                runs.append({"seed": seed, "method": method, "metric": short_name,
                             "integration_interval": "[0.10, 1.00]",
                             "error_budget_auc_deg": float(np.trapezoid(y, x))})
    summary: list[dict[str, Any]] = []
    for metric in ("full_field", "peripheral"):
        for method in METHODS:
            values = np.asarray([row["error_budget_auc_deg"] for row in runs
                                 if row["metric"] == metric and row["method"] == method])
            summary.append({"metric": metric, "method": method, "seed_count": len(values),
                            "error_budget_auc_median_deg": q(values, 50),
                            "error_budget_auc_q25_deg": q(values, 25),
                            "error_budget_auc_q75_deg": q(values, 75)})
    return runs, summary


def plot(output: Path, summary: list[dict[str, Any]], condition: list[dict[str, Any]],
         auc_summary: list[dict[str, Any]]) -> None:
    plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 9, "pdf.fonttype": 42, "ps.fonttype": 42})
    frame = {method: sorted([row for row in summary if row["method"] == method], key=lambda row: row["budget_percent"]) for method in METHODS}
    panels = (("raw_w1_information", "Weak-direction Information", True),
              ("smallest_positive_eigenvalue", "Weakest Intrinsic Information", True),
              ("final_full_ray_p95_deg", "Full-field Ray Error (deg)", False),
              ("final_peripheral_ray_p95_deg", "Peripheral Ray Error (deg)", False))
    stems = ("weak_direction_information", "weakest_intrinsic_information",
             "full_field_ray_error", "peripheral_ray_error")
    for (metric, label, logscale), stem in zip(panels, stems):
        fig, axis = plt.subplots(figsize=(4.45, 3.25))
        for method in METHODS:
            rows = frame[method]; x = np.asarray([row["budget_percent"] for row in rows])
            y = np.asarray([row[f"{metric}_median"] for row in rows], dtype=float)
            lo = np.asarray([row[f"{metric}_q25"] for row in rows], dtype=float)
            hi = np.asarray([row[f"{metric}_q75"] for row in rows], dtype=float)
            axis.plot(x, y, marker="o", markersize=4.2, linewidth=1.8, color=COLORS[method], label=method)
            axis.fill_between(x, lo, hi, color=COLORS[method], alpha=0.16, linewidth=0)
        if logscale: axis.set_yscale("log")
        axis.set_xlabel("Selected frame-board budget (%)"); axis.set_ylabel(label)
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.6); axis.set_xlim(8, 102)
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(loc="best", frameon=False, handlelength=2.3)
        if metric in {"raw_w1_information", "smallest_positive_eigenvalue"}:
            axis.text(0.98, 0.04, "Fixed scene/prefix: IQR = 0", transform=axis.transAxes,
                      ha="right", va="bottom", fontsize=7, color="#555555")
        fig.tight_layout()
        fig.savefig(output / f"incremental_prefix_ds_{stem}.pdf", bbox_inches="tight")
        fig.savefig(output / f"incremental_prefix_ds_{stem}.png", dpi=400, bbox_inches="tight")
        plt.close(fig)

    for suffix in ("pdf", "png"):
        stale = output / f"incremental_prefix_ds_observability_recovery_2x2.{suffix}"
        if stale.exists():
            stale.unlink()

    fig, axis = plt.subplots(figsize=(4.1, 2.8))
    for method in METHODS:
        rows = frame[method]; x = np.asarray([row["budget_percent"] for row in rows])
        y = np.asarray([row["log10_condition_number_median"] for row in rows], dtype=float)
        lo = np.asarray([row["log10_condition_number_q25"] for row in rows], dtype=float)
        hi = np.asarray([row["log10_condition_number_q75"] for row in rows], dtype=float)
        axis.plot(x, y, marker="o", markersize=3.8, linewidth=1.65, color=COLORS[method], label=method)
        axis.fill_between(x, lo, hi, color=COLORS[method], alpha=0.16, linewidth=0)
    axis.set_xlabel("Selected frame-board budget (%)"); axis.set_ylabel(r"$\log_{10}$ condition number")
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.6); axis.spines[["top", "right"]].set_visible(False); axis.legend(frameon=False)
    axis.text(0.02, 0.04, "Fixed scene/prefix: IQR = 0", transform=axis.transAxes,
              ha="left", va="bottom", fontsize=7, color="#555555")
    fig.tight_layout(); fig.savefig(output / "incremental_prefix_ds_condition_number.pdf", bbox_inches="tight")
    fig.savefig(output / "incremental_prefix_ds_condition_number.png", dpi=400, bbox_inches="tight"); plt.close(fig)

    fig, axis = plt.subplots(figsize=(4.45, 3.25))
    metric_labels = (("full_field", "Full-field"), ("peripheral", "Peripheral"))
    centers = np.arange(len(metric_labels), dtype=float)
    width = 0.34
    for offset, method in ((-0.5 * width, "Outer-only"), (0.5 * width, "Outer+Internal")):
        rows = [next(row for row in auc_summary if row["metric"] == key and row["method"] == method)
                for key, _ in metric_labels]
        values = np.asarray([row["error_budget_auc_median_deg"] for row in rows])
        low = values - np.asarray([row["error_budget_auc_q25_deg"] for row in rows])
        high = np.asarray([row["error_budget_auc_q75_deg"] for row in rows]) - values
        bars = axis.bar(centers + offset, values, width=width, color=COLORS[method], label=method,
                        yerr=np.vstack((low, high)), capsize=3, error_kw={"elinewidth": 0.9, "capthick": 0.9})
        for bar, value in zip(bars, values):
            axis.text(bar.get_x() + 0.5 * bar.get_width(), bar.get_height(), f"{value:.3f}",
                      ha="center", va="bottom", fontsize=7.5, color=COLORS[method])
    axis.set_xticks(centers, [label for _, label in metric_labels])
    axis.set_ylabel("Error-budget AUC (deg)")
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.6); axis.spines[["top", "right"]].set_visible(False)
    ymax = max(row["error_budget_auc_q75_deg"] for row in auc_summary)
    axis.set_ylim(0.0, 1.19 * ymax)
    axis.legend(frameon=False, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.18))
    axis.text(0.02, 0.04, "Median with IQR; lower is better",
              transform=axis.transAxes, fontsize=7, color="#555555")
    fig.tight_layout(rect=(0, 0, 1, 0.84)); fig.savefig(output / "incremental_prefix_ds_error_budget_auc.pdf", bbox_inches="tight")
    fig.savefig(output / "incremental_prefix_ds_error_budget_auc.png", dpi=400, bbox_inches="tight"); plt.close(fig)


def main() -> int:
    a = args(); output = a.output.resolve(); output.mkdir(parents=True, exist_ok=True)
    budgets = parse_budgets(a.budgets)
    if a.seed_count < 2 or a.noise_sigma_px <= 0.0:
        raise ValueError("--seed-count must be >=2 and --noise-sigma-px must be positive")
    scene = weak.parse_scene(a.reference_scene.resolve())
    schedule = schedule_from_decisions(a.selection_decisions.resolve())
    rows = load_rows(a.training_points.resolve(), scene)
    available = {(row["frame"], row["board"]) for row in rows}
    schedule = [key for key in schedule if key in available]
    if len(schedule) < 10:
        raise RuntimeError("Too few accepted groups overlap the frozen point source")
    prefixes: dict[int, list[tuple[int, int]]] = {budget: schedule[:math.ceil(len(schedule) * budget / 100.0)] for budget in budgets}
    write_csv(output / "accepted_frame_board_schedule.csv", [{"rank": i + 1, "frame_id": key[0], "board_id": key[1]} for i, key in enumerate(schedule)])
    write_csv(output / "prefix_manifest.csv", [{"budget_percent": budget, "frame_board_count": len(keys), "frame_count": len({key[0] for key in keys}), "prefix_fingerprint": digest(keys)} for budget, keys in prefixes.items()])

    inverse_variance = 1.0 / (a.noise_sigma_px * a.noise_sigma_px)
    full_outer = select_rows(rows, set(prefixes[100]), "Outer-only")
    full_fisher, full_audit = fisher(scene, full_outer, a.width, a.height, inverse_variance)
    full_values, _, _, _ = eig_summary(full_fisher)
    w1 = np.linalg.eigh(full_fisher)[1][:, 0]
    w1 /= np.linalg.norm(w1)
    mask = sweep.build_evaluation_mask(scene.camera, a.width, a.height, a.grid_size)
    amplitude, initial, initial_metrics = weak.calibrate_perturbation(scene.camera, w1, 1, a.initial_peripheral_ray_p95_deg,
                                                                        a.width, a.height, mask, 0.99)
    observability: list[dict[str, Any]] = []
    prepared: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for budget, keys in prefixes.items():
        for method in METHODS:
            selected = select_rows(rows, set(keys), method); prepared[(budget, method)] = selected
            matrix, audit = fisher(scene, selected, a.width, a.height, inverse_variance)
            values, rank, smallest, log_condition = eig_summary(matrix)
            total_weight = inverse_variance * len(selected)
            group_weights = [inverse_variance * sum((row["frame"], row["board"]) == key for row in selected) for key in keys]
            raw = float(w1 @ matrix @ w1)
            observability.append({"budget_percent": budget, "method": method, "frame_board_count": len(keys),
                                  "frame_count": len({key[0] for key in keys}), "point_count": len(selected),
                                  "raw_w1_information": raw, "smallest_positive_eigenvalue": smallest, "rank": rank,
                                  "log10_condition_number": log_condition, "occupied_grid_coverage": coverage(scene, selected, a.width, a.height, a.coverage_grid_size),
                                  "normalized_w1_information_total_effective_weight": raw / total_weight,
                                  "normalized_w1_information_frame_board_group_weight": raw / float(np.mean(group_weights)),
                                  "w1_fingerprint": digest(w1.tolist()), "fisher_eigenvalues": json.dumps(values.tolist()),
                                  "fisher_audit_frame_count": audit["frame_count"], "fisher_audit_board_count": audit["board_count"]})
    write_csv(output / "prefix_observability.csv", observability)

    recovery: list[dict[str, Any]] = []
    for seed in range(a.seed_start, a.seed_start + a.seed_count):
        noise = np.random.default_rng(seed).normal(0.0, a.noise_sigma_px, size=(len(rows), 2))
        noise_fingerprint = digest(noise.tolist())
        for budget in budgets:
            for method in METHODS:
                result = recover(initial, scene.camera, prepared[(budget, method)], noise, a.width, a.height, mask)
                recovery.append({"seed": seed, "budget_percent": budget, "method": method,
                                 "noise_sigma_px": a.noise_sigma_px, "noise_fingerprint": noise_fingerprint,
                                 "prefix_fingerprint": digest(prefixes[budget]), "initial_camera_fingerprint": sweep.camera_fingerprint(initial),
                                 "initial_peripheral_ray_p95_deg": initial_metrics["peripheral_ray_p95_deg"], **result})
    write_csv(output / "prefix_intrinsic_recovery_runs.csv", recovery)
    summaries = summary_rows(observability, recovery); write_csv(output / "prefix_summary.csv", summaries)
    paired = paired_statistics(recovery); write_csv(output / "prefix_paired_statistics.csv", paired)
    auc_runs, auc_summary = error_budget_auc_rows(recovery)
    write_csv(output / "error_budget_auc_runs.csv", auc_runs)
    write_csv(output / "error_budget_auc_seed_summary.csv", auc_summary)

    final: dict[str, Any] = {"protocol": "incremental_prefix_ds_observability_recovery_v1", "budgets_percent": budgets,
                             "selection_decisions": str(a.selection_decisions.resolve()), "reference_scene": str(a.reference_scene.resolve()),
                             "training_points": str(a.training_points.resolve()), "source_hashes": {"decisions": digest_file(a.selection_decisions.resolve()), "scene": digest_file(a.reference_scene.resolve()), "points": digest_file(a.training_points.resolve())},
                             "accepted_group_count": len(schedule), "w1_eigenvalue_full_outer": float(full_values[0]), "w1_fingerprint": digest(w1.tolist()),
                             "initial_peripheral_ray_p95_deg": initial_metrics["peripheral_ray_p95_deg"], "perturbation_amplitude": amplitude,
                             "noise_sigma_px": a.noise_sigma_px, "seed_count": a.seed_count, "full_outer_fisher_audit": full_audit}
    error_budget: list[dict[str, Any]] = []
    for method in METHODS:
        curve = sorted([row for row in observability if row["method"] == method], key=lambda row: row["budget_percent"])
        information = np.asarray([row["raw_w1_information"] for row in curve]); x = np.asarray([row["budget_percent"] for row in curve])
        final_info = float(information[-1]); final[f"{method}_final_w1_information"] = final_info
        final[f"{method}_w1_information_auc"] = float(np.trapezoid(information, x / 100.0))
        final[f"{method}_budget_at_90pct_final_w1_information"] = int(x[np.flatnonzero(information >= 0.9 * final_info)[0]])
        ray_curve = sorted([row for row in summaries if row["method"] == method], key=lambda row: row["budget_percent"])
        budget_fraction = np.asarray([float(row["budget_percent"]) / 100.0 for row in ray_curve])
        budget_row: dict[str, Any] = {"method": method, "integration_interval": "[0.10, 1.00]"}
        for metric, short_name in (("final_full_ray_p95_deg", "full_field"),
                                   ("final_peripheral_ray_p95_deg", "peripheral")):
            values = np.asarray([float(row[f"{metric}_median"]) for row in ray_curve])
            hit = [int(row["budget_percent"]) for row in ray_curve if float(row[f"{metric}_median"]) <= 0.01]
            auc = float(np.trapezoid(values, budget_fraction))
            budget = hit[0] if hit else None
            final[f"{method}_{short_name}_ray_error_auc_measured_10_to_100"] = auc
            final[f"{method}_budget_at_{short_name}_ray_p95_le_001deg"] = budget
            budget_row[f"{short_name}_ray_error_auc_measured_10_to_100"] = auc
            budget_row[f"budget_at_{short_name}_ray_p95_le_001deg"] = budget
        error_budget.append(budget_row)
    write_csv(output / "error_budget_summary.csv", error_budget)
    final["paired_statistics"] = paired
    (output / "experiment_summary.json").write_text(json.dumps(final, indent=2) + "\n", encoding="utf-8")
    plot(output, summaries, observability, auc_summary)
    print(json.dumps(final, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
