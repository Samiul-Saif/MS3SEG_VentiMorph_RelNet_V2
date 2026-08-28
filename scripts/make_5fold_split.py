#!/usr/bin/env python
"""Reproduce `patient_level_5fold_split.csv` exactly as the Fold-0 training notebook creates it.

The Fold-0 notebook (`notebook381a28a764`) builds the split once; folds 1-4 and the evaluation
notebook then reuse that file verbatim so every fold shares the same 80/20 development/locked-test
partition and the same 5 CV folds.

Logic (identical to the notebook, seed 42):
  1. abWMH volume (ml) per patient  = abwmh_voxels * spacing_x*spacing_y*spacing_z / 1000
  2. burden_bin                     = pd.qcut(volume.rank(method="first"), q=4, labels=False)
  3. train_test_split(test_size=0.20, random_state=42, stratify=burden_bin) -> 80 dev / 20 test
  4. StratifiedKFold(n_splits=5, shuffle=True, random_state=42) on the 80 dev patients
Output columns: patient_id, fold, role   (role in {train, val, test}; test rows use fold = -1)

Input: an `integrity_audit.csv` as written by the audit cell of any training notebook. Required
columns: patient_id, abwmh_voxels, spacing_x, spacing_y, spacing_z.

Usage:
  python scripts/make_5fold_split.py --audit integrity_audit.csv --out patient_level_5fold_split.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

SEED = 42


def build_split(audit_csv: Path) -> pd.DataFrame:
    audit = pd.read_csv(audit_csv, dtype={"patient_id": str})
    audit["patient_id"] = audit["patient_id"].astype(str).str.zfill(3)

    required = {"patient_id", "abwmh_voxels", "spacing_x", "spacing_y", "spacing_z"}
    missing = required - set(audit.columns)
    if missing:
        raise SystemExit(f"audit CSV missing columns: {sorted(missing)}")

    # Match the notebook: patient_stats is derived from the patient-id-sorted manifest.
    stats = audit.sort_values("patient_id").reset_index(drop=True).copy()
    voxel_volume_mm3 = stats.spacing_x * stats.spacing_y * stats.spacing_z
    stats["abwmh_volume_ml"] = stats.abwmh_voxels * voxel_volume_mm3 / 1000.0
    stats["burden_bin"] = pd.qcut(
        stats.abwmh_volume_ml.rank(method="first"), q=4, labels=False
    )

    trainval_ids, test_ids = train_test_split(
        stats.patient_id,
        test_size=0.20,
        random_state=SEED,
        stratify=stats.burden_bin,
    )

    trainval_stats = stats[stats.patient_id.isin(trainval_ids)].reset_index(drop=True)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    rows: list[dict] = []
    for fold, (tr, va) in enumerate(
        skf.split(trainval_stats.patient_id, trainval_stats.burden_bin)
    ):
        rows += [
            {"patient_id": trainval_stats.loc[i, "patient_id"], "fold": fold, "role": "train"}
            for i in tr
        ]
        rows += [
            {"patient_id": trainval_stats.loc[i, "patient_id"], "fold": fold, "role": "val"}
            for i in va
        ]
    rows += [{"patient_id": pid, "fold": -1, "role": "test"} for pid in sorted(test_ids)]

    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--audit", type=Path, required=True, help="integrity_audit.csv")
    ap.add_argument("--out", type=Path, default=Path("patient_level_5fold_split.csv"))
    args = ap.parse_args()

    split_df = build_split(args.audit)
    split_df.to_csv(args.out, index=False)

    for fold in range(5):
        n_tr = ((split_df.fold == fold) & (split_df.role == "train")).sum()
        n_va = ((split_df.fold == fold) & (split_df.role == "val")).sum()
        print(f"fold {fold}: train={n_tr} val={n_va}")
    print(f"locked test: {(split_df.role == 'test').sum()}")
    print(f"written: {args.out}")


if __name__ == "__main__":
    main()
