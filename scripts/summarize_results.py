#!/usr/bin/env python
"""Turn the evaluation-notebook output CSVs into markdown tables for the paper.

Point it at the folder produced by `ms3seg-ventimorph-relnet-v2-7-fold0-validation-eva.ipynb`
(`MS3SEG_V27_FOLD0_VALIDATION/`, or the unzipped `*_RESULTS.zip`).

Usage:
  python scripts/summarize_results.py --results-dir MS3SEG_V27_FOLD0_VALIDATION --out paper_tables.md
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _load(path: Path) -> pd.DataFrame | None:
    return pd.read_csv(path) if path.exists() else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("paper_tables.md"))
    args = ap.parse_args()

    d = args.results_dir
    patient = _load(d / "fold0_validation_patient_metrics.csv")
    class_summary = _load(d / "fold0_validation_class_summary.csv")
    overall = _load(d / "fold0_validation_overall_summary.csv")
    confusion = _load(d / "fold0_validation_confusion_matrix.csv")

    lines: list[str] = ["# Fold-0 validation — paper tables", ""]
    lines.append(f"_source: `{d}` (evaluation notebook output)_\n")

    if patient is not None:
        lines.append(f"Patients scored: **{len(patient)}**\n")

    if overall is not None:
        lines += ["## Headline metrics", "", "| metric | value |", "|---|---|"]
        for _, r in overall.iterrows():
            lines.append(f"| {r['metric']} | {r['value']:.4f} |")
        lines.append("")

    if class_summary is not None:
        lines += ["## Per-class summary (mean ± std over patients)", "",
                  "| class | metric | mean | std | median | min | max |",
                  "|---|---|---|---|---|---|---|"]
        for _, r in class_summary.iterrows():
            lines.append(
                f"| {r['class']} | {r['metric']} | {r['mean']:.4f} | {r['std']:.4f} | "
                f"{r['median']:.4f} | {r['minimum']:.4f} | {r['maximum']:.4f} |"
            )
        lines.append("")

    if confusion is not None:
        lines += ["## Voxel confusion matrix", "", confusion.to_markdown(index=False), ""]

    args.out.write_text("\n".join(lines), encoding="utf-8")
    print(f"written: {args.out}")


if __name__ == "__main__":
    main()
