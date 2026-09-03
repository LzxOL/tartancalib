#!/usr/bin/env python3
"""Generate paper tables for the seed-1337 fixed-backend residual ablation."""

import generate_residual_mode_multiset_table as table


table.OUTPUT_DIR = (
    table.ROOT / "paper" / "experiments" / "fixed_backend_residual_ablation_seed1337"
)
table.DATASETS = [
    ("1444190", "Sequence A"),
    ("134853", "Sequence B"),
    ("144928", "Sequence C"),
]
table.RESULT_DIRS = {
    (dataset, method): (
        f"result_may/stage5_fixed_backend_seed1337_{dataset}clear_{suffix}"
    )
    for dataset in ("1444190", "134853", "144928")
    for method, suffix in (
        ("Pixel", "pixel"),
        ("Spherical", "spherical"),
        ("Hybrid", "hybrid"),
    )
}


if __name__ == "__main__":
    table.main()
    output = table.OUTPUT_DIR / "table_residual_modes_multiset.tex"
    latex = output.read_text(encoding="utf-8")
    latex = latex.replace(
        "All modes share the split, initialization, and frozen holdout "
        "observations within each sequence.",
        "All modes share the split, initialization, fixed backend frame-board "
        "inputs, and frozen holdout observations within each sequence.",
    )
    latex = latex.replace("\\setlength{\\tabcolsep}{3.3pt}", "\\setlength{\\tabcolsep}{1.6pt}")
    output.write_text(latex, encoding="utf-8")
