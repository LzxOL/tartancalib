#!/usr/bin/env python3
"""Render publication figures from a completed DS prefix experiment.

This script is deliberately separate from the experiment runner.  It only
reads the CSV summaries in an existing result directory and regenerates the
individual figures; no calibration or recovery is performed.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:  # Keep --help usable in minimal environments.
    plt = None  # type: ignore[assignment]


METHODS = ("Outer-only", "Outer+Internal")
COLORS = {"Outer-only": "#C44E52", "Outer+Internal": "#4C72B0"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir", type=Path, required=True,
        help="Completed experiment directory containing the summary CSV files.",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        help="Figure output directory; defaults to INPUT_DIR/figures.",
    )
    parser.add_argument(
        "--figures", default="all",
        help=("Comma-separated names: weak, weakest, full, peripheral, "
              "condition, auc, or all (default: all)."),
    )
    parser.add_argument("--dpi", type=int, default=400)
    parser.add_argument("--font-size", type=float, default=9.0)
    parser.add_argument("--format", dest="formats", default="png,pdf",
                        help="Comma-separated output formats (default: png,pdf).")
    parser.add_argument("--title", action="store_true",
                        help="Add a compact title containing the experiment name.")
    parser.add_argument("--experiment-label", default="",
                        help="Short experiment name used in figure titles and metadata.")
    parser.add_argument("--dataset-label", default="",
                        help="Dataset/camera label, for example '144928-clear Right'.")
    parser.add_argument("--model-label", default="DS",
                        help="Camera model label, for example DS or KB.")
    parser.add_argument("--perturbation-label", default="",
                        help="Perturbation description, for example 'P1, s=0.8'.")
    parser.add_argument("--perturbation-direction", default="",
                        help="Perturbation direction, for example P1.")
    parser.add_argument("--perturbation-scale", type=float,
                        help="Perturbation interpolation scale s.")
    parser.add_argument("--focal-scale", type=float,
                        help="Applied focal-length multiplier.")
    parser.add_argument("--delta-xi", type=float,
                        help="Applied DS xi offset.")
    parser.add_argument("--delta-alpha", type=float,
                        help="Applied DS alpha offset.")
    parser.add_argument("--delta-cu-px", type=float,
                        help="Applied principal-point u offset, in pixels.")
    parser.add_argument("--delta-cv-px", type=float,
                        help="Applied principal-point v offset, in pixels.")
    parser.add_argument("--initial-ray-p95-deg", type=float,
                        help="Initial peripheral Ray P95 used by the experiment, in degrees.")
    parser.add_argument("--noise-sigma-px", type=float,
                        help="Synthetic observation noise sigma, in pixels.")
    parser.add_argument("--seed-count", type=int,
                        help="Number of paired random seeds used by the experiment.")
    parser.add_argument("--budget-schedule", default="",
                        help="Selected frame-board budgets, for example '10,20,40,60,80,100'.")
    parser.add_argument("--reference-label", default="",
                        help="Reference model/scene description.")
    parser.add_argument("--metadata-json", type=Path,
                        help="Optional JSON object with additional experiment metadata.")
    parser.add_argument("--show-details", action="store_true",
                        help="Show supplied experiment details as a compact subtitle.")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Required summary file not found: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def number(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    if value in ("", "nan", "NaN", "None"):
        return float("nan")
    return float(value)


def configure_style(font_size: float, dpi: int) -> None:
    if plt is None:
        raise RuntimeError(
            "Figure rendering requires matplotlib; install it with "
            "python3 -m pip install matplotlib"
        )
    plt.rcParams.update({
        "font.family": "DejaVu Serif",
        "font.size": font_size,
        "axes.labelsize": font_size,
        "axes.titlesize": font_size + 1,
        "xtick.labelsize": max(font_size - 1, 7),
        "ytick.labelsize": max(font_size - 1, 7),
        "legend.fontsize": max(font_size - 1, 7),
        "figure.dpi": dpi,
        "savefig.dpi": dpi,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def selected(value: str, requested: set[str]) -> bool:
    return "all" in requested or value in requested


def experiment_metadata(parsed: argparse.Namespace, input_dir: Path) -> dict[str, object]:
    metadata: dict[str, object] = {}
    if parsed.metadata_json:
        with parsed.metadata_json.resolve().open(encoding="utf-8") as handle:
            loaded = json.load(handle)
        if not isinstance(loaded, dict):
            raise ValueError("--metadata-json must contain a JSON object")
        metadata.update(loaded)
    fields = {
        "experiment_label": parsed.experiment_label,
        "dataset_label": parsed.dataset_label,
        "model_label": parsed.model_label,
        "perturbation_label": parsed.perturbation_label,
        "perturbation_direction": parsed.perturbation_direction,
        "perturbation_scale": parsed.perturbation_scale,
        "focal_scale": parsed.focal_scale,
        "delta_xi": parsed.delta_xi,
        "delta_alpha": parsed.delta_alpha,
        "delta_cu_px": parsed.delta_cu_px,
        "delta_cv_px": parsed.delta_cv_px,
        "initial_ray_p95_deg": parsed.initial_ray_p95_deg,
        "noise_sigma_px": parsed.noise_sigma_px,
        "seed_count": parsed.seed_count,
        "budget_schedule": parsed.budget_schedule,
        "reference_label": parsed.reference_label,
        "input_dir": str(input_dir),
    }
    metadata.update({key: value for key, value in fields.items()
                     if value not in ("", None)})
    return metadata


def details_title(metadata: dict[str, object], show_details: bool) -> str | None:
    if not show_details:
        return None
    label = str(metadata.get("experiment_label", "")).strip()
    details = []
    for key in ("dataset_label", "model_label", "perturbation_label"):
        value = str(metadata.get(key, "")).strip()
        if value:
            details.append(value)
    direction = str(metadata.get("perturbation_direction", "")).strip()
    scale = metadata.get("perturbation_scale")
    if direction or scale is not None:
        perturbation = direction or "perturbation"
        if scale is not None:
            perturbation += f", s={float(scale):g}"
        details.append(perturbation)
    if metadata.get("initial_ray_p95_deg") is not None:
        details.append(f"initial P95={float(metadata['initial_ray_p95_deg']):g} deg")
    if metadata.get("noise_sigma_px") is not None:
        details.append(f"sigma={float(metadata['noise_sigma_px']):g} px")
    return "\n".join(part for part in (label, " | ".join(details)) if part) or None


def save_figure(fig: plt.Figure, output_dir: Path, stem: str,
                formats: tuple[str, ...], dpi: int) -> None:
    for fmt in formats:
        if fmt not in {"png", "pdf", "svg"}:
            raise ValueError(f"Unsupported format: {fmt}")
        kwargs = {"bbox_inches": "tight"}
        if fmt == "png":
            kwargs["dpi"] = dpi
        fig.savefig(output_dir / f"{stem}.{fmt}", **kwargs)
    plt.close(fig)


def grouped(rows: list[dict[str, str]], method: str) -> list[dict[str, str]]:
    return sorted((row for row in rows if row.get("method") == method),
                  key=lambda row: int(row["budget_percent"]))


def plot_prefix_metric(rows: list[dict[str, str]], output_dir: Path, stem: str,
                       metric: str, label: str, logscale: bool,
                       formats: tuple[str, ...], dpi: int, title: str | None) -> None:
    fig, axis = plt.subplots(figsize=(4.45, 3.25))
    for method in METHODS:
        method_rows = grouped(rows, method)
        x = np.asarray([number(row, "budget_percent") for row in method_rows])
        median = np.asarray([number(row, f"{metric}_median") for row in method_rows])
        q25 = np.asarray([number(row, f"{metric}_q25") for row in method_rows])
        q75 = np.asarray([number(row, f"{metric}_q75") for row in method_rows])
        axis.plot(x, median, marker="o", markersize=4.2, linewidth=1.8,
                  color=COLORS[method], label=method)
        axis.fill_between(x, q25, q75, color=COLORS[method], alpha=0.16,
                          linewidth=0, label="_nolegend_")
    if logscale:
        axis.set_yscale("log")
    axis.set_xlabel("Selected frame-board budget (%)")
    axis.set_ylabel(label)
    if title:
        axis.set_title(title, fontsize=9, pad=10)
    axis.set_xlim(8, 102)
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.6)
    axis.legend(loc="best", frameon=False, handlelength=2.3)
    fig.tight_layout()
    save_figure(fig, output_dir, f"incremental_prefix_ds_{stem}", formats, dpi)


def plot_condition(rows: list[dict[str, str]], output_dir: Path,
                   formats: tuple[str, ...], dpi: int, title: str | None) -> None:
    fig, axis = plt.subplots(figsize=(4.25, 2.9))
    for method in METHODS:
        method_rows = grouped(rows, method)
        x = np.asarray([number(row, "budget_percent") for row in method_rows])
        median = np.asarray([number(row, "log10_condition_number_median") for row in method_rows])
        q25 = np.asarray([number(row, "log10_condition_number_q25") for row in method_rows])
        q75 = np.asarray([number(row, "log10_condition_number_q75") for row in method_rows])
        axis.plot(x, median, marker="o", markersize=3.8, linewidth=1.65,
                  color=COLORS[method], label=method)
        axis.fill_between(x, q25, q75, color=COLORS[method], alpha=0.16,
                          linewidth=0, label="_nolegend_")
    axis.set_xlabel("Selected frame-board budget (%)")
    axis.set_ylabel(r"$\log_{10}$ condition number")
    if title:
        axis.set_title(title, fontsize=9, pad=10)
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.6)
    axis.legend(frameon=False)
    fig.tight_layout()
    save_figure(fig, output_dir, "incremental_prefix_ds_condition_number", formats, dpi)


def plot_auc(rows: list[dict[str, str]], output_dir: Path,
             formats: tuple[str, ...], dpi: int, title: str | None) -> None:
    metrics = (("full_field", "Full-field"), ("peripheral", "Peripheral"))
    fig, axis = plt.subplots(figsize=(4.45, 3.25))
    centers = np.arange(len(metrics), dtype=float)
    width = 0.34
    for offset, method in ((-0.5 * width, METHODS[0]), (0.5 * width, METHODS[1])):
        values = []
        q25 = []
        q75 = []
        for metric, _ in metrics:
            match = next(row for row in rows
                         if row.get("metric") == metric and row.get("method") == method)
            values.append(number(match, "error_budget_auc_median_deg"))
            q25.append(number(match, "error_budget_auc_q25_deg"))
            q75.append(number(match, "error_budget_auc_q75_deg"))
        values = np.asarray(values); lower = values - np.asarray(q25); upper = np.asarray(q75) - values
        bars = axis.bar(centers + offset, values, width=width, color=COLORS[method],
                        label=method, yerr=np.vstack((lower, upper)), capsize=3,
                        error_kw={"elinewidth": 0.9, "capthick": 0.9})
        for bar, value in zip(bars, values):
            axis.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                      f"{value:.3f}", ha="center", va="bottom", fontsize=7.5,
                      color=COLORS[method])
    axis.set_xticks(centers, [label for _, label in metrics])
    axis.set_ylabel("Error-budget AUC (deg)")
    if title:
        axis.set_title(title, fontsize=9, pad=10)
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.6)
    ymax = max(number(row, "error_budget_auc_q75_deg") for row in rows)
    axis.set_ylim(0.0, 1.19 * ymax)
    axis.legend(frameon=False, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.18))
    fig.tight_layout(rect=(0, 0, 1, 0.84))
    save_figure(fig, output_dir, "incremental_prefix_ds_error_budget_auc", formats, dpi)


def main() -> int:
    parsed = parse_args()
    if plt is None:
        raise RuntimeError(
            "Figure rendering requires matplotlib; install it with "
            "python3 -m pip install matplotlib"
        )
    input_dir = parsed.input_dir.resolve()
    output_dir = (parsed.output_dir or input_dir / "figures").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = experiment_metadata(parsed, input_dir)
    (output_dir / "figure_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    requested = {part.strip().lower() for part in parsed.figures.split(",") if part.strip()}
    formats = tuple(part.strip().lower() for part in parsed.formats.split(",") if part.strip())
    configure_style(parsed.font_size, parsed.dpi)
    title = details_title(metadata, parsed.show_details)
    if parsed.title and title is None:
        title = "DS incremental-prefix experiment"

    summary = read_csv(input_dir / "prefix_summary.csv")
    observability = read_csv(input_dir / "prefix_observability.csv")
    auc = read_csv(input_dir / "error_budget_auc_seed_summary.csv")
    if not summary or not observability or not auc:
        raise RuntimeError("One or more input CSV files are empty")

    plots = {
        "weak": ("raw_w1_information", "Weak-direction Information", True,
                 "weak_direction_information"),
        "weakest": ("smallest_positive_eigenvalue", "Weakest Intrinsic Information", True,
                    "weakest_intrinsic_information"),
        "full": ("final_full_ray_p95_deg", "Full-field Ray Error (deg)", False,
                 "full_field_ray_error"),
        "peripheral": ("final_peripheral_ray_p95_deg", "Peripheral Ray Error (deg)", False,
                       "peripheral_ray_error"),
    }
    for name, (metric, label, logscale, stem) in plots.items():
        if selected(name, requested):
            source = summary if name in {"full", "peripheral"} else observability
            plot_prefix_metric(source, output_dir, stem, metric, label, logscale,
                               formats, parsed.dpi, title if parsed.title or parsed.show_details else None)
    if selected("condition", requested):
        plot_condition(observability, output_dir, formats, parsed.dpi,
                       title if parsed.title or parsed.show_details else None)
    if selected("auc", requested):
        plot_auc(auc, output_dir, formats, parsed.dpi,
                 title if parsed.title or parsed.show_details else None)

    generated = sorted(path.name for path in output_dir.iterdir()
                       if path.is_file() and path.suffix.lstrip(".") in formats)
    print("Generated figures:")
    for name in generated:
        print(f"  {output_dir / name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
