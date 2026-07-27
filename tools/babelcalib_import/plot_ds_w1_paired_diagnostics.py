#!/usr/bin/env python3
"""Plot paired diagnostics from a completed DS-W1 local-recovery sweep."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd

import run_ds_perturbation_sweep as sweep
import run_ds_weak_mode_perturbation as weak


DEFAULT_LEVELS = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0)
METHODS = ("Outer-only", "Outer+Internal")
COLORS = {"Outer-only": "#D55E00", "Outer+Internal": "#0072B2"}
FIGURES = ("ecdf", "scatter", "parameters", "radial", "heatmap")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--sweep-dir", type=Path, required=True,
        help="Directory containing local_recovery_runs.csv, pixel runs, and protocol manifest.",
    )
    parser.add_argument(
        "--output", type=Path,
        help="Output directory. Defaults to --sweep-dir.",
    )
    parser.add_argument(
        "--levels", type=float, nargs="+", default=list(DEFAULT_LEVELS),
        help="Perturbation levels to audit and plot, in degrees.",
    )
    parser.add_argument(
        "--expected-seeds", type=int, default=100,
        help="Required paired seed count at every selected level.",
    )
    parser.add_argument(
        "--representative-level", type=float, default=0.5,
        help="Level used by scatter, parameter-distribution, and radial figures.",
    )
    parser.add_argument(
        "--peripheral-rho-threshold", type=float, default=0.7,
        help="Displayed peripheral-region boundary in the radial profile.",
    )
    parser.add_argument(
        "--radial-bins", type=int, default=20,
        help="Number of normalized-radius bins in the radial profile.",
    )
    parser.add_argument(
        "--figures", nargs="+", choices=("all", *FIGURES), default=["all"],
        help="Figures to generate. 'all' generates every diagnostic.",
    )
    parser.add_argument(
        "--formats", nargs="+", choices=("png", "pdf"), default=["png", "pdf"],
        help="Output figure formats.",
    )
    parser.add_argument("--dpi", type=int, default=400, help="Raster output DPI.")
    parser.add_argument(
        "--filename-suffix", default="",
        help="Optional suffix appended to generated figure and derived CSV stems.",
    )
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: f"{value:.15g}" if isinstance(value, float) else value
                for key, value in row.items()
            })


def configure_style(dpi: int) -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["STIX Two Text", "Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix", "font.size": 8.5,
        "axes.labelsize": 8.5, "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5, "legend.fontsize": 7.3,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.7, "savefig.dpi": dpi,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def style_axis(axis: plt.Axes, panel: str) -> None:
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.5, alpha=0.7)
    axis.set_axisbelow(True)
    axis.text(
        0.012, 0.985, panel, transform=axis.transAxes,
        va="top", ha="left", fontfamily="DejaVu Serif",
        fontweight="bold", fontsize=8,
    )


def suffixed_stem(stem: str, suffix: str) -> str:
    return f"{stem}_{suffix}" if suffix else stem


def save_figure(
    fig: plt.Figure, output: Path, stem: str, formats: tuple[str, ...], dpi: int,
) -> None:
    for suffix in formats:
        fig.savefig(
            output / f"{stem}.{suffix}", dpi=dpi,
            bbox_inches="tight", pad_inches=0.04,
        )
    plt.close(fig)


def load_data(
    sweep_dir: Path, levels: tuple[float, ...], expected_seeds: int,
) -> tuple[pd.DataFrame, pd.DataFrame, sweep.EvaluationMask]:
    required_paths = {
        "runs": sweep_dir / "local_recovery_runs.csv",
        "pixels": sweep_dir / "local_recovery_pixel_runs.csv",
        "protocol": sweep_dir / "protocol_manifest.json",
    }
    missing = [str(path) for path in required_paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required sweep input(s): " + ", ".join(missing))
    runs = pd.read_csv(required_paths["runs"])
    pixels = pd.read_csv(required_paths["pixels"])
    key_columns = {"run_key", "perturbation_level_deg", "seed", "method"}
    for name, frame in (("runs", runs), ("pixels", pixels)):
        absent = key_columns - set(frame.columns)
        if absent:
            raise RuntimeError(f"{name} CSV is missing columns: {sorted(absent)}")
        frame["perturbation_level_deg"] = frame.perturbation_level_deg.astype(float)
    runs = runs[runs.perturbation_level_deg.isin(levels)].copy()
    pixels = pixels[pixels.perturbation_level_deg.isin(levels)].copy()
    reference_seeds = set(
        runs.loc[runs.perturbation_level_deg == levels[0], "seed"].tolist()
    )
    if len(reference_seeds) != expected_seeds:
        raise RuntimeError(
            f"Level {levels[0]:g} contains {len(reference_seeds)} seed IDs; "
            f"expected {expected_seeds}"
        )
    expected_keys = {
        (level, seed, method)
        for level in levels
        for seed in reference_seeds
        for method in METHODS
    }
    for name, frame in (("runs", runs), ("pixels", pixels)):
        actual_keys = set(
            zip(frame.perturbation_level_deg, frame.seed, frame.method, strict=True)
        )
        duplicates = frame.duplicated(
            ["perturbation_level_deg", "seed", "method"], keep=False,
        )
        if duplicates.any():
            duplicate_keys = frame.loc[
                duplicates, ["perturbation_level_deg", "seed", "method"]
            ].drop_duplicates().head(5).to_dict("records")
            raise RuntimeError(f"Duplicate paired keys in {name}: {duplicate_keys}")
        missing_keys = expected_keys - actual_keys
        extra_keys = actual_keys - expected_keys
        if missing_keys or extra_keys:
            raise RuntimeError(
                f"Incomplete paired grid in {name}: missing={len(missing_keys)}, "
                f"extra={len(extra_keys)}; expected {len(expected_keys)} rows"
            )
    if runs.run_key.nunique() != len(runs) or pixels.run_key.nunique() != len(pixels):
        raise RuntimeError("Duplicate run keys in diagnostic source")
    if set(runs.run_key) != set(pixels.run_key):
        raise RuntimeError("local recovery and pixel run keys differ")
    protocol = json.loads(required_paths["protocol"].read_text(encoding="utf-8"))
    scene = weak.parse_scene(Path(protocol["reference_scene"]))
    mask = sweep.build_evaluation_mask(
        scene.camera, int(protocol["image_width"]), int(protocol["image_height"]),
        int(protocol["grid_size"]),
    )
    return runs, pixels, mask


def paired_rows(runs: pd.DataFrame, pixels: pd.DataFrame) -> pd.DataFrame:
    keys = ["perturbation_level_deg", "seed"]
    rows = []
    for (level, seed), group in pixels.groupby(keys):
        values = group.set_index("method")
        outer = values.loc["Outer-only"]
        internal = values.loc["Outer+Internal"]
        source = runs[(runs.perturbation_level_deg == level) & (runs.seed == seed)].set_index("method")
        rows.append({
            "perturbation_level_deg": float(level), "seed": int(seed),
            "full_pixel_p95_delta_px": float(outer.full_pixel_p95_px - internal.full_pixel_p95_px),
            "peripheral_pixel_p95_delta_px": float(outer.peripheral_pixel_p95_px - internal.peripheral_pixel_p95_px),
            "full_pixel_median_delta_px": float(outer.full_pixel_median_px - internal.full_pixel_median_px),
            "peripheral_pixel_median_delta_px": float(outer.peripheral_pixel_median_px - internal.peripheral_pixel_median_px),
            "xi_abs_error_delta": float(source.loc["Outer-only", "xi_abs_error"] - source.loc["Outer+Internal", "xi_abs_error"]),
            "alpha_abs_error_delta": float(source.loc["Outer-only", "alpha_abs_error"] - source.loc["Outer+Internal", "alpha_abs_error"]),
            "focal_error_delta": float(source.loc["Outer-only", "mean_focal_relative_error"] - source.loc["Outer+Internal", "mean_focal_relative_error"]),
            "principal_point_error_delta_px": float(source.loc["Outer-only", "principal_point_error_px"] - source.loc["Outer+Internal", "principal_point_error_px"]),
        })
    return pd.DataFrame(rows).sort_values(["perturbation_level_deg", "seed"])


def plot_ecdf(
    pairs: pd.DataFrame, output: Path, levels: tuple[float, ...],
    formats: tuple[str, ...], dpi: int, filename_suffix: str,
) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.0), constrained_layout=True)
    specs = (
        ("full_pixel_p95_delta_px", "Full Pixel P95 improvement (px)"),
        ("peripheral_pixel_p95_delta_px", "Peripheral Pixel P95 improvement (px)"),
        ("xi_abs_error_delta", r"$|\xi-\xi_{gt}|$ improvement"),
        ("focal_error_delta", "Mean focal error improvement"),
    )
    handles = []
    for axis, (metric, ylabel), panel in zip(axes.flat, specs, ("(a)", "(b)", "(c)", "(d)")):
        level_colors = plt.cm.viridis(np.linspace(0.12, 0.88, len(levels)))
        for level, color in zip(levels, level_colors):
            values = pairs.loc[pairs.perturbation_level_deg == level, metric].to_numpy(float)
            values = np.sort(values[np.isfinite(values)])
            y = np.arange(1, len(values) + 1) / len(values)
            line, = axis.step(values, y, where="post", color=color, linewidth=1.1, label=f"{level:g}")
            if metric == specs[0][0]:
                handles.append(line)
        axis.axvline(0.0, color="#555555", linestyle="--", linewidth=0.8)
        axis.set_xlabel("Outer-only minus Outer+Internal")
        axis.set_ylabel(ylabel)
        axis.set_ylim(0.0, 1.0)
        style_axis(axis, panel)
    fig.legend(handles, [f"level {level:g}" for level in levels], loc="upper center", ncol=len(levels),
               frameon=False, bbox_to_anchor=(0.5, 1.02), title="Target initial Peripheral Ray P95 (deg)")
    stem = suffixed_stem("paired_pixel_improvement_ecdf_2x2", filename_suffix)
    save_figure(fig, output, stem, formats, dpi)
    return output / f"{stem}.{formats[0]}"


def plot_scatter(
    runs: pd.DataFrame, pixels: pd.DataFrame, output: Path, level: float,
    formats: tuple[str, ...], dpi: int, filename_suffix: str,
) -> Path:
    left = runs[runs.perturbation_level_deg == level].set_index(["seed", "method"])
    right = pixels[pixels.perturbation_level_deg == level].set_index(["seed", "method"])
    joined = left.join(right[["full_pixel_p95_px", "peripheral_pixel_p95_px"]])
    records = []
    for seed in sorted(joined.index.get_level_values(0).unique()):
        row = joined.loc[seed]
        records.append({
            "full_outer": row.loc["Outer-only", "full_pixel_p95_px"],
            "full_internal": row.loc["Outer+Internal", "full_pixel_p95_px"],
            "peripheral_outer": row.loc["Outer-only", "peripheral_pixel_p95_px"],
            "peripheral_internal": row.loc["Outer+Internal", "peripheral_pixel_p95_px"],
            "xi_outer": row.loc["Outer-only", "xi_abs_error"],
            "xi_internal": row.loc["Outer+Internal", "xi_abs_error"],
            "focal_outer": row.loc["Outer-only", "mean_focal_relative_error"] * 100.0,
            "focal_internal": row.loc["Outer+Internal", "mean_focal_relative_error"] * 100.0,
        })
    data = pd.DataFrame(records)
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.0), constrained_layout=True)
    specs = (
        ("full_outer", "full_internal", "Full Pixel P95 (px)"),
        ("peripheral_outer", "peripheral_internal", "Peripheral Pixel P95 (px)"),
        ("xi_outer", "xi_internal", r"$|\xi-\xi_{gt}|$"),
        ("focal_outer", "focal_internal", "Mean focal error (%)"),
    )
    for axis, (xkey, ykey, label), panel in zip(axes.flat, specs, ("(a)", "(b)", "(c)", "(d)")):
        x = data[xkey].to_numpy(float)
        y = data[ykey].to_numpy(float)
        limit = max(float(np.max(x)), float(np.max(y))) * 1.05
        axis.scatter(x, y, s=14, alpha=0.65, color=COLORS["Outer+Internal"], edgecolors="none")
        axis.plot([0.0, limit], [0.0, limit], color="#555555", linestyle="--", linewidth=0.9)
        axis.set_xlim(0.0, limit)
        axis.set_ylim(0.0, limit)
        axis.set_xlabel("Outer-only")
        axis.set_ylabel("Outer+Internal")
        axis.set_title(label, fontsize=8.5, pad=3)
        style_axis(axis, panel)
    fig.text(0.5, -0.012, f"Representative target initial Peripheral Ray P95 = {level:g} deg; points below y=x favor Outer+Internal.",
             ha="center", fontsize=7.5, style="italic")
    stem = suffixed_stem("paired_error_scatter_2x2", filename_suffix)
    save_figure(fig, output, stem, formats, dpi)
    return output / f"{stem}.{formats[0]}"


def plot_parameter_distributions(
    runs: pd.DataFrame, output: Path, level: float,
    formats: tuple[str, ...], dpi: int, filename_suffix: str,
) -> Path:
    data = runs[runs.perturbation_level_deg == level].copy()
    specs = (
        ("xi_abs_error", r"$|\xi-\xi_{gt}|$"),
        ("alpha_abs_error", r"$|\alpha-\alpha_{gt}|$"),
        ("mean_focal_relative_error", "Mean focal error (%)"),
        ("principal_point_error_px", "Principal-point error (px)"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.0), constrained_layout=True)
    for axis, (metric, ylabel), panel in zip(axes.flat, specs, ("(a)", "(b)", "(c)", "(d)")):
        values = []
        labels = []
        for method in METHODS:
            value = data.loc[data.method == method, metric].to_numpy(float)
            if metric == "mean_focal_relative_error":
                value = value * 100.0
            values.append(value)
            labels.append(method)
        violin = axis.violinplot(values, positions=[1, 2], showmeans=True, showmedians=True, widths=0.75)
        for body, method in zip(violin["bodies"], METHODS):
            body.set_facecolor(COLORS[method]); body.set_edgecolor(COLORS[method]); body.set_alpha(0.25)
        for key in ("cmeans", "cmedians", "cbars", "cmins", "cmaxes"):
            if key in violin:
                violin[key].set_color("#444444"); violin[key].set_linewidth(0.7)
        axis.set_xticks([1, 2], ["Outer-only", "Outer+Internal"], rotation=15)
        axis.set_ylabel(ylabel)
        axis.set_ylim(bottom=0.0)
        style_axis(axis, panel)
    fig.text(0.5, -0.012, f"100 paired seeds at target initial Peripheral Ray P95 = {level:g} deg; markers show mean and median.",
             ha="center", fontsize=7.5, style="italic")
    stem = suffixed_stem("parameter_error_distribution_2x2", filename_suffix)
    save_figure(fig, output, stem, formats, dpi)
    return output / f"{stem}.{formats[0]}"


def camera_from_run(row: pd.Series) -> sweep.Camera:
    return sweep.Camera(
        float(row.final_xi), float(row.final_alpha),
        float(row.final_fu), float(row.final_fv),
        float(row.final_cu), float(row.final_cv), family="ds-none",
    )


def radial_rows(
    runs: pd.DataFrame, mask: sweep.EvaluationMask, level: float, bins: int = 20,
) -> list[dict[str, Any]]:
    source = runs[runs.perturbation_level_deg == level]
    values: dict[str, list[np.ndarray]] = {method: [] for method in METHODS}
    for _, row in source.iterrows():
        pixels, valid = weak.project_ds(camera_from_run(row), mask.reference_rays)
        valid &= mask.reference_valid
        error = np.linalg.norm(pixels - mask.pixels, axis=1)
        binned = []
        for index in range(bins):
            low, high = index / bins, (index + 1) / bins
            selected = valid & (mask.rho >= low) & (
                mask.rho <= high if index == bins - 1 else mask.rho < high
            )
            binned.append(float(np.percentile(error[selected], 95)) if np.any(selected) else math.nan)
        values[row.method].append(np.asarray(binned))
    rows = []
    for method in METHODS:
        matrix = np.asarray(values[method])
        for index in range(bins):
            column = matrix[:, index]
            rows.append({
                "perturbation_level_deg": level, "method": method,
                "rho_bin_index": index, "rho_center": (index + 0.5) / bins,
                "pixel_p95_mean_px": float(np.nanmean(column)),
                "pixel_p95_median_px": float(np.nanmedian(column)),
                "pixel_p95_q25_px": float(np.nanpercentile(column, 25)),
                "pixel_p95_q75_px": float(np.nanpercentile(column, 75)),
                "pixel_p95_p05_px": float(np.nanpercentile(column, 5)),
                "pixel_p95_p95_px": float(np.nanpercentile(column, 95)),
                "valid_run_count": int(np.count_nonzero(np.isfinite(column))),
            })
    return rows


def rho_tag(value: float) -> str:
    return f"{value:.2f}".replace(".", "p")


def plot_radial(
    radial: pd.DataFrame, output: Path, level: float, peripheral_threshold: float,
    formats: tuple[str, ...], dpi: int, filename_suffix: str,
) -> Path:
    fig, axis = plt.subplots(figsize=(7.0, 3.25), constrained_layout=True)
    for method in METHODS:
        group = radial[radial.method == method].sort_values("rho_center")
        x = group.rho_center.to_numpy(float)
        axis.fill_between(x, group.pixel_p95_p05_px, group.pixel_p95_p95_px,
                          color=COLORS[method], alpha=0.14, linewidth=0)
        axis.plot(x, group.pixel_p95_mean_px, color=COLORS[method], linewidth=1.5,
                  label=f"{method} mean")
        axis.plot(x, group.pixel_p95_median_px, color=COLORS[method], linestyle="--",
                  linewidth=1.25, label=f"{method} median")
    axis.axvline(
        peripheral_threshold, color="#555555", linestyle=":", linewidth=0.9,
        label=rf"$\rho={peripheral_threshold:g}$",
    )
    axis.set_xlabel(r"Normalized radius $\rho$")
    axis.set_ylabel("Pixel Displacement P95 (px)")
    axis.set_xlim(0.0, 1.0); axis.set_ylim(bottom=0.0)
    style_axis(axis, "(a)")
    axis.legend(frameon=False, ncol=3, loc="upper left")
    axis.set_title(f"Target initial Peripheral Ray P95 = {level:g} deg", fontsize=8.5, pad=3)
    base_stem = (
        "radial_pixel_displacement_profile"
        if math.isclose(peripheral_threshold, 0.7, abs_tol=1e-12)
        else f"radial_pixel_displacement_profile_rho_{rho_tag(peripheral_threshold)}"
    )
    stem = suffixed_stem(base_stem, filename_suffix)
    save_figure(fig, output, stem, formats, dpi)
    return output / f"{stem}.{formats[0]}"


def plot_heatmap(
    pairs: pd.DataFrame, output: Path, levels: tuple[float, ...],
    formats: tuple[str, ...], dpi: int, filename_suffix: str,
) -> Path:
    pivot = pairs.pivot(index="seed", columns="perturbation_level_deg", values="peripheral_pixel_p95_delta_px")
    pivot = pivot.reindex(columns=levels).sort_index()
    maximum = float(np.nanmax(np.abs(pivot.to_numpy(float))))
    fig, axis = plt.subplots(figsize=(7.0, 4.4), constrained_layout=True)
    image = axis.imshow(
        pivot.to_numpy(float), aspect="auto", origin="lower",
        cmap="RdBu_r", norm=TwoSlopeNorm(vcenter=0.0, vmin=-maximum, vmax=maximum),
    )
    axis.set_xlabel("Perturbation level / target initial Peripheral Ray P95 (deg)")
    axis.set_ylabel("Noise seed")
    axis.set_xticks(np.arange(len(levels)), [f"{level:g}" for level in levels])
    seed_count = len(pivot.index)
    tick_indices = np.unique(np.linspace(0, seed_count - 1, min(5, seed_count)).round().astype(int))
    axis.set_yticks(tick_indices, [str(pivot.index[index]) for index in tick_indices])
    axis.text(0.012, 0.985, "(a)", transform=axis.transAxes, va="top", ha="left",
              fontfamily="DejaVu Serif", fontweight="bold", fontsize=8)
    colorbar = fig.colorbar(image, ax=axis, pad=0.015)
    colorbar.set_label("Outer-only minus Outer+Internal (px)")
    fig.text(0.5, -0.012, "Positive values favor Outer+Internal; each row is one paired noise seed.",
             ha="center", fontsize=7.5, style="italic")
    stem = suffixed_stem("paired_improvement_heatmap", filename_suffix)
    save_figure(fig, output, stem, formats, dpi)
    return output / f"{stem}.{formats[0]}"


def main() -> int:
    args = parse_args()
    sweep_dir = args.sweep_dir.resolve()
    output = (args.output or sweep_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    levels = tuple(dict.fromkeys(float(level) for level in args.levels))
    if not levels or any(level < 0.0 for level in levels):
        raise ValueError("levels must contain one or more non-negative values")
    if args.expected_seeds <= 0:
        raise ValueError("expected seeds must be positive")
    if args.representative_level not in levels:
        raise ValueError(f"representative level must be one of {levels}")
    if not 0.0 < args.peripheral_rho_threshold <= 1.0:
        raise ValueError("peripheral rho threshold must be in (0, 1]")
    if args.radial_bins <= 0:
        raise ValueError("radial bins must be positive")
    if args.dpi <= 0:
        raise ValueError("dpi must be positive")
    figures = FIGURES if "all" in args.figures else tuple(dict.fromkeys(args.figures))
    formats = tuple(dict.fromkeys(args.formats))
    runs, pixels, mask = load_data(sweep_dir, levels, args.expected_seeds)
    pairs = paired_rows(runs, pixels)
    paired_stem = suffixed_stem("paired_pixel_improvement", args.filename_suffix)
    paired_path = output / f"{paired_stem}.csv"
    write_csv(paired_path, pairs.to_dict("records"))
    radial_path: Path | None = None
    if "radial" in figures:
        radial_stem = suffixed_stem("radial_pixel_displacement_profile", args.filename_suffix)
        radial_path = output / f"{radial_stem}.csv"
        radial = radial_rows(runs, mask, args.representative_level, args.radial_bins)
        write_csv(radial_path, radial)
    # All figures below reload materialized CSVs, so no plot uses hidden values
    # that are absent from the experiment artifacts.
    pairs = pd.read_csv(paired_path)
    configure_style(args.dpi)
    generated: list[Path] = []
    if "ecdf" in figures:
        generated.append(plot_ecdf(
            pairs, output, levels, formats, args.dpi, args.filename_suffix,
        ))
    if "scatter" in figures:
        generated.append(plot_scatter(
            runs, pixels, output, args.representative_level,
            formats, args.dpi, args.filename_suffix,
        ))
    if "parameters" in figures:
        generated.append(plot_parameter_distributions(
            runs, output, args.representative_level,
            formats, args.dpi, args.filename_suffix,
        ))
    if "radial" in figures:
        assert radial_path is not None
        generated.append(plot_radial(
            pd.read_csv(radial_path), output, args.representative_level,
            args.peripheral_rho_threshold, formats, args.dpi, args.filename_suffix,
        ))
    if "heatmap" in figures:
        generated.append(plot_heatmap(
            pairs, output, levels, formats, args.dpi, args.filename_suffix,
        ))
    caption_stem = suffixed_stem("paired_diagnostics_captions", args.filename_suffix)
    (output / f"{caption_stem}.md").write_text(
        f"Mean/median and P5-P95 are computed over {args.expected_seeds} paired seeds. "
        "Pixel displacement is obtained by projecting fixed GT rays through the final DS model; "
        "positive paired delta means Outer+Internal is better. Distribution and radial plots use "
        f"the representative target level {args.representative_level:g} deg; selected levels are "
        f"{', '.join(f'{level:g}' for level in levels)} deg. The radial plot marks the fixed "
        f"peripheral boundary rho={args.peripheral_rho_threshold:g}.\n",
        encoding="utf-8",
    )
    print(f"Validated {len(levels)} levels x {args.expected_seeds} seeds x 2 methods")
    print(paired_path)
    for path in generated:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
