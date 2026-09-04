# results/ — numbers extracted from the committed notebook outputs

This folder makes the repository self-contained for the paper's core numbers. Everything here is
recovered from the **outputs already saved inside the notebooks** in this repo — nothing was
re-run. See [`../docs/RESULTS_MAP.md`](../docs/RESULTS_MAP.md) for the cell-by-cell provenance.

| File | What it is | How produced |
|---|---|---|
| `history/fold<k>_history.csv` | per-epoch validation curve for fold *k* (train/val loss, per-class Dice, weighted & balanced score, grad norm, lr) | `scripts/extract_from_notebooks.py`, parsed from each training notebook's `E### …` stdout |
| `fold0_validation/patient_dice.csv` | per-patient foreground / ventricle / nWMH / abWMH Dice for the 16 Fold-0 validation patients | same script, parsed from the evaluation notebook's `[NN/16] …` stdout |
| `fold0_validation/overall_summary.csv` | headline Fold-0 validation metrics | transcribed from `eval` notebook cell 13 (`fold0_validation_overall_summary.csv`) |
| `fold0_validation/class_summary.csv` | per-class mean/std/median/min/max for Dice, IoU, precision, recall, specificity, HD95, ASSD (n = 16) | transcribed from `eval` notebook cell 13 (`fold0_validation_class_summary.csv`) |
| `fold0_validation/roc_auc.csv` | per-class voxel-level ROC AUC, one-vs-rest (Ventricle 0.9992, Normal-WMH 0.9968, Abnormal-WMH 0.9949; background is `NaN`, a known sampling edge case at ~99% prevalence, fixed in the notebook code for the next run) | transcribed from `eval` notebook's appended section 19 (executed 2026-09-05) |

## Caveats

- `history/fold0_history.csv` covers epochs **11–59** only; the fold-0 pilot epochs 1–10 were not
  retained in that notebook's saved output. Best epoch (47) is included. Folds 1–4 include the
  pilot epochs.
- History Dice is **slice-level** (training-loop confusion matrix). The paper's primary numbers are
  the **3-D patient-level** metrics in `fold0_validation/`.
- The full per-patient metric table (HD95, ASSD, lesion F1, periventricular FP, uncertainty) is only
  partially echoed in the notebook display; the authoritative copy is
  `fold0_validation_patient_metrics.csv` inside `MS3SEG_V27_FOLD0_VALIDATION_RESULTS.zip` from the
  Kaggle run. Regenerate it by running the evaluation notebook.

## Regenerate

```bash
python scripts/extract_from_notebooks.py                 # history + patient_dice from notebooks
python scripts/summarize_results.py --results-dir <unzipped Kaggle results> --out paper_tables.md
```
