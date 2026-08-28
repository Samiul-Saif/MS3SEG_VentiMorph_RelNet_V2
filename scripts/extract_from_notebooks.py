#!/usr/bin/env python
"""Extract the results that are embedded in the committed notebook OUTPUTS into plain CSVs.

The training notebooks each write a `history.csv` and the evaluation notebook writes several
metric CSVs, but those files live only inside the Kaggle run outputs / transfer ZIPs. This
script reconstructs what is recoverable directly from the notebook cell outputs that ARE
committed here, so the repository is self-contained for the per-epoch learning curves and the
per-patient validation Dice.

It parses:
  * training notebooks -> `results/history/fold<k>_history.csv`
      from the `E### train=... val=... FG=... V=... n=... ab=... W=... B=... grad=... lr=...` lines
  * evaluation notebook -> `results/fold0_validation/patient_dice.csv`
      from the `[NN/16] <pid> | FG Dice=... | V=... | N=... | A=...` lines

Full per-patient metrics (HD95, ASSD, lesion F1, ...) and the class/overall summary tables are
only partially echoed in the notebook display output; the authoritative copies are in
`MS3SEG_V27_FOLD0_VALIDATION_RESULTS.zip`. The transcribed summary lives in
`results/fold0_validation/class_summary.csv` / `overall_summary.csv` (see docs/RESULTS_MAP.md).

Usage:  python scripts/extract_from_notebooks.py
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

TRAIN_NOTEBOOKS = {
    0: "notebook381a28a764%20(2).ipynb",
    1: "ms3seg-ventimorph-relnet-v2-7-fold1-training (1).ipynb",
    2: "ms3seg-ventimorph-relnet-v2-7-fold2-training (1).ipynb",
    3: "ms3seg-ventimorph-relnet-v2-7-fold3-training (1).ipynb",
    4: "ms3seg-ventimorph-relnet-v2-7-fold4-training (1).ipynb",
}
EVAL_NOTEBOOK = "ms3seg-ventimorph-relnet-v2-7-fold0-validation-eva.ipynb"

EPOCH_RE = re.compile(
    r"E(\d+) train=([\d.]+) val=([\d.]+) FG=([\d.]+) V=([\d.]+) n=([\d.]+) "
    r"ab=([\d.]+) W=([\d.]+) B=([\d.]+) grad=([\d.]+) gAlarm=(\d+) collapse=(\d+) lr=([\d.]+)"
)
PATIENT_RE = re.compile(
    r"\[(\d+)/\d+\]\s+(\d+)\s+\|\s+FG Dice=([\d.]+)\s+\|\s+V=([\d.]+)\s+\|\s+"
    r"N=([\d.]+)\s+\|\s+A=([\d.]+)"
)


def stream_lines(nb_path: Path):
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    for cell in nb.get("cells", []):
        for out in cell.get("outputs", []):
            if out.get("output_type") == "stream":
                yield from "".join(out.get("text", [])).splitlines()


def extract_history(fold: int, nb_path: Path) -> int:
    rows: dict[int, tuple] = {}
    for line in stream_lines(nb_path):
        m = EPOCH_RE.match(line.strip())
        if m:
            rows[int(m.group(1))] = m.groups()
    if not rows:
        return 0
    out = REPO / "results" / "history" / f"fold{fold}_history.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["epoch", "train_loss", "val_loss", "val_mean_fg_dice", "val_dice_vent",
                    "val_dice_nwmh", "val_dice_abwmh", "val_weighted_score",
                    "val_balanced_score", "mean_gradient_norm", "gradient_alarm_counter",
                    "collapse_counter", "lr"])
        for e in sorted(rows):
            w.writerow((e,) + rows[e][1:])
    print(f"fold {fold}: {len(rows)} epochs -> {out.relative_to(REPO)}")
    return len(rows)


def extract_patient_dice(nb_path: Path) -> int:
    rows = []
    for line in stream_lines(nb_path):
        m = PATIENT_RE.search(line)
        if m:
            rows.append(m.groups()[1:])
    if not rows:
        return 0
    out = REPO / "results" / "fold0_validation" / "patient_dice.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["patient_id", "mean_foreground_dice", "ventricle_dice",
                    "normal_wmh_dice", "abnormal_wmh_dice"])
        w.writerows(rows)
    print(f"eval: {len(rows)} patients -> {out.relative_to(REPO)}")
    return len(rows)


def main() -> None:
    for fold, name in TRAIN_NOTEBOOKS.items():
        p = REPO / name
        if p.exists():
            extract_history(fold, p)
        else:
            print(f"fold {fold}: notebook not found ({name})")
    ev = REPO / EVAL_NOTEBOOK
    if ev.exists():
        extract_patient_dice(ev)


if __name__ == "__main__":
    main()
