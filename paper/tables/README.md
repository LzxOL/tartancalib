# Paper Tables

This directory contains the only formal tables included by
`ICRA2026-V1.tex`. The main paper imports them in numerical order so their
LaTeX numbering remains Table I--IV.

| LaTeX source | Paper table | Purpose |
| --- | --- | --- |
| `01_reprojection_comparison.tex` | Table I | Main reprojection comparison across camera-model families and held-out multi-board sets. |
| `02_cross_backend_verification.tex` | Table II | Cross-backend verification using the same detected observations and target geometry. |
| `03_internal_seed_ablation.tex` | Table III | Ablation of local-pinhole and spherical internal-seed generation. |
| `04_spherical_ba_ablation.tex` | Table IV | Evaluation of pixel, spherical, and polar-aware hybrid residual objectives. |

## Editing convention

- Keep each table self-contained: it must include its own `table` or `table*`
  environment, caption, label, and formatting settings.
- Add a new table only after assigning the next numeric prefix and adding one
  corresponding `\input{tables/...}` entry in `ICRA2026-V1.tex`.
- Keep data-generation scripts and raw experiment outputs under
  `paper_experiments/`; do not place build previews, PDFs, PNGs, or temporary
  CSV exports in this directory.
