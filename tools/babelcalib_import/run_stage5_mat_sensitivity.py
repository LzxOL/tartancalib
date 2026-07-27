#!/usr/bin/env python3
"""Run fixed-test training-size sensitivity for outer-only vs outer+internal."""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
PYTHON = Path(__file__).resolve().parents[1] / ""  # kept for readable paths below
SCIPY_PYTHON = Path(
    "/Users/linzhaoxian/.cache/codex-runtimes/"
    "codex-primary-runtime/dependencies/python/bin/python3"
)
SUBSET = Path(__file__).with_name("subset_babelcalib_mat.py")
RUNNER = Path(__file__).with_name("run_stage5_mat_catalog.py")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-pool", type=Path, required=True)
    p.add_argument("--test-mat", type=Path, required=True)
    p.add_argument("--dataset-prefix", required=True)
    p.add_argument("--camera", choices=("left", "right"), default="right")
    p.add_argument("--ratios", default="20,40,60,70,80,100")
    p.add_argument("--seeds", default="1337,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019,2020")
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--max-workers", type=int, default=8)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def run(command: list[str], cwd: Path, dry_run: bool) -> None:
    print("+ " + " ".join(command), flush=True)
    if not dry_run:
        subprocess.run(command, cwd=cwd, check=True)


def parse_summary(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            values[key.strip()] = value.strip()
    return values


def main() -> int:
    args = parse_args()
    ratios = [int(value) for value in args.ratios.split(",") if value.strip()]
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    output_root = args.output_root.resolve()
    subset_root = output_root / "subsets"
    runs_root = output_root / "runs"
    subset_root.mkdir(parents=True, exist_ok=True)
    jobs: list[tuple[int, int, str, Path]] = []
    for ratio in ratios:
        for seed in seeds:
            subset = subset_root / f"train_{ratio}_seed{seed}.mat"
            if not subset.is_file():
                run([
                    str(SCIPY_PYTHON), str(SUBSET),
                    "--input", str(args.train_pool.resolve()),
                    "--output", str(subset),
                    "--ratio", f"{ratio / 100.0:.2f}",
                    "--seed", str(seed),
                ], REPO, args.dry_run)
            for mode, include in (("outer_only", "0"), ("outer_internal", "1")):
                jobs.append((ratio, seed, mode, subset))

    def one(job: tuple[int, int, str, Path]) -> dict[str, str]:
        ratio, seed, mode, subset = job
        include = "0" if mode == "outer_only" else "1"
        dataset_id = f"{args.dataset_prefix}-train{ratio}-seed{seed}-{mode}"
        run_dir = runs_root / f"train{ratio}_seed{seed}_{mode}"
        command = [
            sys.executable, str(RUNNER),
            "--train-mat", str(subset),
            "--test-mat", str(args.test_mat.resolve()),
            "--dataset-id", dataset_id,
            "--camera", args.camera,
            "--models", "ds",
            "--catalog-subdir", str(output_root / "catalog"),
            "--run-root", str(run_dir),
            "--target-mode", "multi_board",
            "--tag", f"seed{seed}",
            "--include-internal-points", include,
            "--cache-dir", str(output_root / "cache" / f"train{ratio}_seed{seed}_{mode}"),
        ]
        run(command, REPO, args.dry_run)
        summary_path = run_dir / f"stage5_mat_{dataset_id}_{args.camera}_seed{seed}" / "ds" / "backend_holdout_summary.txt"
        row: dict[str, str] = {"ratio_percent": str(ratio), "seed": str(seed), "mode": mode, "run_dir": str(run_dir)}
        if not args.dry_run and summary_path.is_file():
            row.update(parse_summary(summary_path))
        return row

    rows: list[dict[str, str]] = []
    if args.dry_run:
        for job in jobs:
            rows.append(one(job))
    else:
        with ThreadPoolExecutor(max_workers=max(1, args.max_workers)) as pool:
            futures = [pool.submit(one, job) for job in jobs]
            for future in as_completed(futures):
                rows.append(future.result())
                print(f"completed {len(rows)}/{len(jobs)}", flush=True)
    rows.sort(key=lambda row: (int(row["ratio_percent"]), row["mode"], int(row["seed"])))
    output_root.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with (output_root / "sensitivity_results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {output_root / 'sensitivity_results.csv'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
