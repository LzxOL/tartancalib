#!/usr/bin/env python3
"""Generate a single-column branching method flowchart for Fig. 2."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "pic"

INK = "#252B30"
MUTED = "#5E6971"
FLOW = "#46515A"
INPUT_FILL, INPUT_EDGE = "#F1F4F8", "#728191"
OUTER_FILL, OUTER_EDGE = "#FCF0E5", "#C66A25"
INIT_FILL, INIT_EDGE = "#F8F1DF", "#B79336"
INTERNAL_FILL, INTERNAL_EDGE = "#EAF3F8", "#2C7EAC"
MERGE_FILL, MERGE_EDGE = "#F4F5F6", "#65737C"
FINAL_FILL, FINAL_EDGE = "#EAF4EC", "#4A9568"


def block(ax, x, y, width, height, fill, edge, title, subtitle=None, *, dashed=False):
    linestyle = (0, (2.2, 1.6)) if dashed else "solid"
    ax.add_patch(
        FancyBboxPatch(
            (x, y), width, height,
            boxstyle="round,pad=0.018,rounding_size=0.035",
            facecolor=fill, edgecolor=edge, linewidth=1.05, linestyle=linestyle,
        )
    )
    if subtitle:
        ax.text(x + width / 2, y + height * 0.63, title, ha="center", va="center",
                fontsize=7.05, fontweight="bold", color=INK)
        ax.text(x + width / 2, y + height * 0.28, subtitle, ha="center", va="center",
                fontsize=5.65, color=MUTED)
    else:
        ax.text(x + width / 2, y + height / 2, title, ha="center", va="center",
                fontsize=7.05, fontweight="bold", color=INK)


def arrow(ax, start, end, color=FLOW, *, linewidth=1.05, mutation=10, connectionstyle="arc3"):
    ax.add_patch(
        FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=mutation,
                        linewidth=linewidth, color=color, shrinkA=0, shrinkB=0,
                        connectionstyle=connectionstyle)
    )


def label(ax, x, y, text, color, *, ha="center", va="center"):
    ax.text(x, y, text, ha=ha, va=va, fontsize=5.35, color=color,
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.22})


def stage_marker(ax, x, y, text, color):
    ax.text(x, y, text, ha="center", va="center", fontsize=5.9, color=color,
            fontweight="bold",
            bbox={"boxstyle": "circle,pad=0.28", "facecolor": "white",
                  "edgecolor": color, "linewidth": 0.8})


def main() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    fig, ax = plt.subplots(figsize=(3.5, 4.75))
    fig.subplots_adjust(left=0.025, right=0.975, bottom=0.025, top=0.985)
    ax.set_xlim(0, 3.5)
    ax.set_ylim(0, 4.75)
    ax.axis("off")

    # Input and shared detection stage.
    block(ax, 0.43, 4.20, 2.64, 0.40, INPUT_FILL, INPUT_EDGE,
          "Input", "Wide-angle frames  +  distributed multi-board target")
    block(ax, 0.43, 3.48, 2.64, 0.48, OUTER_FILL, OUTER_EDGE,
          "Board Detection and Outer4 Extraction",
          "Tag association and outer-corner observations")
    arrow(ax, (1.75, 4.20), (1.75, 3.96))
    stage_marker(ax, 0.18, 3.72, "1", OUTER_EDGE)

    # The two branches make explicit that Outer4 is both a bootstrap signal
    # and part of the final observation set.
    block(ax, 0.20, 2.48, 1.38, 0.55, INIT_FILL, INIT_EDGE,
          "Outer4-only Bootstrap", "Intermediate model + poses")
    block(ax, 1.92, 2.48, 1.38, 0.55, OUTER_FILL, OUTER_EDGE,
          "Outer4 Observations", "Retained for final calibration", dashed=True)
    arrow(ax, (1.32, 3.48), (0.89, 3.03), color=FLOW)
    arrow(ax, (2.18, 3.48), (2.61, 3.03), color=OUTER_EDGE)
    stage_marker(ax, 0.07, 2.76, "2", INIT_EDGE)

    block(ax, 0.20, 1.42, 1.38, 0.70, INTERNAL_FILL, INTERNAL_EDGE,
          "Internal Observation Recovery",
          "Unprojection  $\\rightarrow$  spherical interpolation\nlocal refinement")
    arrow(ax, (0.89, 2.48), (0.89, 2.12))
    label(ax, 0.89, 2.22, "intermediate model", INIT_EDGE, va="bottom")
    stage_marker(ax, 0.07, 1.77, "3", INTERNAL_EDGE)

    # Merge the two observation streams into the set used by the optimizer.
    block(ax, 0.43, 0.70, 2.64, 0.50, MERGE_FILL, MERGE_EDGE,
          "Observation Set", r"$\mathcal{O}=\mathcal{O}^{\mathrm{out}} \cup \mathcal{O}^{\mathrm{int}}$")
    arrow(ax, (0.89, 1.42), (1.24, 1.20), color=INTERNAL_EDGE)
    arrow(ax, (2.61, 2.48), (2.26, 1.20), color=OUTER_EDGE,
          connectionstyle="arc3,rad=-0.08")
    label(ax, 0.48, 1.25, "refined internal observations", INTERNAL_EDGE,
          ha="left", va="bottom")
    label(ax, 3.08, 1.72, "Outer4", OUTER_EDGE, ha="right")

    block(ax, 0.43, 0.08, 2.64, 0.38, FINAL_FILL, FINAL_EDGE,
          "Selection-based Two-pass Bundle Adjustment",
          "Intrinsics, board geometry, and frame poses")
    arrow(ax, (1.75, 0.70), (1.75, 0.46))
    stage_marker(ax, 0.18, 0.27, "4", FINAL_EDGE)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for extension in ("pdf", "png"):
        path = OUTPUT_DIR / f"pipeline_branching_flowchart.{extension}"
        fig.savefig(path, dpi=360, bbox_inches="tight", pad_inches=0.012)
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
