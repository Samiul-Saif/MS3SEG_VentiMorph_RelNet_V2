# Results map — every paper number → its source

Each row: the quantity in the paper, the notebook + cell that computes it, the output file it is
written to, and the value observed in the committed notebook outputs. **If the paper reports a
figure not listed here, add a row.**

Notebook short names:
- **train-f0** = `notebook381a28a764%20(2).ipynb` (Fold-0 training; also builds the split + audit)
- **train-f1..f4** = `ms3seg-ventimorph-relnet-v2-7-fold{1..4}-training (1).ipynb`
- **eval-f0** = `ms3seg-ventimorph-relnet-v2-7-fold0-validation-eva.ipynb`

---

## 1. Dataset / setup facts

| Paper statement | Source | Output file | Value |
|---|---|---|---|
| Cohort size = 100 patients | train-f*, manifest cell (14) | `patient_manifest.csv` | `Patients: 100` |
| 4 classes + voxel class distribution | train-f*, audit cell (15) | `class_distribution.csv` | bg 99.311 % / vent 0.402 % / nWMH 0.116 % / abWMH 0.171 % |
| Mask overlap check | train-f*, audit cell (15) | `integrity_audit.csv` | `{vent_nwmh: 2, vent_abwmh: 9, nwmh_abwmh: 227, triple: 0}` |
| Shapes / affines consistent | train-f*, audit cell (15) | stdout + `integrity_audit.csv` | `All shapes match: True`, `All affines match: True` |
| 80 development / 20 locked test | train-f0, split cell (18) | `patient_level_5fold_split.csv` | `Development: 80  Locked test: 20` |
| 5 folds, 64 train / 16 val each | train-f1..f4, split-load cell (18) | same | `Fold k has 64 training and 16 validation patients` |
| Fold-0 validation patient IDs | eval-f0, cell 5 | `evaluation_metadata.json` | `001,007,017,018,023,034,048,053,062,064,070,072,086,089,095,099` |
| Trainable parameters ≈ 5.97 M | train-f*, cell 26 | stdout | `Parameters: 5974441` |
| GPU = Tesla T4 (sm_75) | train-f*, cell 10 | stdout | `Compatible GPU: Tesla T4; capability=sm_75` |

## 2. Training — checkpoint that produces the paper's evaluation

| Paper statement | Source | Output file | Value |
|---|---|---|---|
| Selected checkpoint = Fold-0 best *balanced*, epoch 47 | train-f0, cells 40/44/46 | `.../v27_fresh_stable_fold0/best_model.pt`, `history.csv` | `epoch 47`, val mean-FG Dice 0.7637 |
| This is the `best_model.pt` loaded for evaluation | eval-f0, cell 9 | stdout | `Checkpoint epoch: 47`, `Recorded mean foreground Dice: 0.7637`, `5974441` params, all frozen |

## 3. Fold-0 validation — 3-D patient-level metrics (PRIMARY paper results)

All from **eval-f0**. Per-patient rows → `fold0_validation_patient_metrics.csv`; aggregates below.

### 3.1 Headline (`fold0_validation_overall_summary.csv`, eval-f0 cell 13)

| Metric | Value |
|---|---|
| mean patient foreground Dice | 0.7552 |
| std patient foreground Dice | 0.0464 |
| mean ventricle Dice | 0.8516 |
| mean normal-WMH Dice | 0.6361 |
| mean abnormal-WMH Dice | 0.7779 |
| mean abnormal-WMH lesion F1 | 0.6920 |
| mean normal-WMH lesion F1 | 0.4731 |
| mean periventricular abnormal-WMH FP rate | 0.2576 |

### 3.2 Per-class (`fold0_validation_class_summary.csv`, eval-f0 cell 13), mean ± std (n = 16)

| Class | Dice | IoU | Precision | Recall | Specificity | HD95 (mm) | ASSD (mm) |
|---|---|---|---|---|---|---|---|
| Ventricle | 0.852 ± 0.025 | 0.742 ± 0.037 | 0.839 ± 0.045 | 0.871 ± 0.067 | 0.9993 | 3.53 ± 1.72 | 0.58 ± 0.18 |
| Normal-WMH | 0.636 ± 0.079 | 0.471 ± 0.082 | 0.595 ± 0.108 | 0.700 ± 0.098 | 0.9995 | 9.84 ± 3.98 | 1.83 ± 0.84 |
| Abnormal-WMH | 0.778 ± 0.089 | 0.644 ± 0.112 | 0.761 ± 0.104 | 0.820 ± 0.127 | 0.9996 | 11.47 ± 10.88 | 1.92 ± 1.79 |

### 3.3 Voxel confusion matrix

`fold0_validation_confusion_matrix.csv` (eval-f0 cell 12); normalized version in
`figures/fold0_validation_confusion_matrix.png` (cell 14).

### 3.4 Figures (eval-f0)

| Figure | File | Cell |
|---|---|---|
| Patient-level Dice box plot | `figures/fold0_validation_dice_boxplot.png` | 14 |
| Normalized voxel confusion matrix | `figures/fold0_validation_confusion_matrix.png` | 14 |
| Best / median / worst qualitative panels | `figures/{best,median,worst}_case_<pid>_slice_<z>.png` | 15 |
| Best / median / worst patient IDs | stdout | 15 → `034` / `099` / `017` |
| Per-class ROC curves (voxel-level, one-vs-rest) | `figures/fold0_validation_roc_curves.png`, `fold0_validation_roc_auc.csv` | 19 (appended) |
| Seg-Grad-CAM explainability panels (worst/median/best × ventricle/nWMH/abWMH) | `figures/gradcam_{worst,median,best}_case_<pid>_slice_<z>.png` | 20 (appended) |

### 3.5 ROC / AUC (executed 2026-09-05, `results/fold0_validation/roc_auc.csv`)

Voxel-level, one-vs-rest, from the 16 Fold-0 validation patients' cached softmax probabilities:

| Class | AUC |
|---|---|
| Ventricle | 0.9992 |
| Normal-WMH | 0.9968 |
| Abnormal-WMH | 0.9949 |
| Background | `NaN` — sampling edge case at ~99% voxel prevalence (fixed in the committed code for the next run; does not affect the three classes above, which are valid) |

## 4. Cross-validation table (slice-level validation Dice, per training notebook)

From each `history.csv`, row of the best *balanced* epoch (train-f*, cells 40/42/44).
**Note:** these are training-loop (confusion-matrix, slice-level) Dice, not the 3-D
patient-level Dice of §3. Use §3 numbers as the paper's primary metric; use these only for the
per-fold learning-stability discussion.

| Fold | best epoch | Ventricle | nWMH | abWMH | mean FG | balanced | pilot gate |
|---|---|---|---|---|---|---|---|
| 0 | 47 | 0.8561 | 0.6409 | 0.7941 | 0.7637 | 0.7411 | pass |
| 1 | 19 | 0.8366 | 0.6269 | 0.7539 | 0.7391 | 0.7176 | pass (early-stopped ep 31) |
| 2 | 58 | 0.8503 | 0.6405 | 0.7444 | 0.7451 | 0.7236 | pass (1 rollback) |
| 3 | 33 | 0.8551 | 0.6427 | 0.7899 | 0.7626 | 0.7404 | **fail** (grad spike) — full run still completed, 1 rollback |
| 4 | 43 | 0.8510 | 0.6163 | 0.7678 | 0.7451 | 0.7208 | **fail** (grad spike / 17 skipped batches) — full run still completed, 1 rollback |

CV mean ± std (slice-level, best-epoch, n = 5): Ventricle 0.850 ± 0.008, nWMH 0.633 ± 0.011,
abWMH 0.770 ± 0.022, mean-FG 0.751 ± 0.011. (Recompute with `scripts/summarize_results.py`
inputs or directly from the five `history.csv` files if you need more decimals.)

## 4b. Table 7 — "VentiMorph-RelNet (ours)" row

`notebooks/MS3SEG_5Fold_Best_Points_Summary.ipynb` scrapes the five source notebooks' saved outputs
and averages.

| Sheet fold | Scraped from | Ventricle | nWMH | abWMH | Mean FG |
|---|---|---|---|---|---|
| Fold-1 | eval-f0 `overall_summary` (3-D patient-level) | 0.851588 | 0.636120 | 0.777938 | 0.755216 |
| Fold-2 | train-f1 best-balanced epoch 19 (slice-level) | 0.836591 | 0.626910 | 0.753907 | 0.739136 |
| Fold-3 | train-f2 best-balanced epoch 58 | 0.850324 | 0.640466 | 0.744385 | 0.745058 |
| Fold-4 | train-f3 best-balanced epoch 33 | 0.855117 | 0.642706 | 0.789882 | 0.762568 |
| Fold-5 | train-f4 best-balanced epoch 43 | 0.851024 | 0.616333 | 0.767844 | 0.745067 |
| **5-fold mean** | | **0.848929** | **0.632507** | **0.766791** | **0.749409** |
| Paper Table 7 | | 0.8489 | 0.6325 | 0.76679 | 0.7494 | ✓ |

Caveat: Fold-1 is a 3-D patient-level Dice, Folds 2–5 are training-loop (slice-level) Dice at the
selected epoch. The averages equal the paper's Table 7 row; note the metric mix in the manuscript
or run a full 5-fold patient-level evaluation.

## 5. Not evaluated / not yet run

| Item | Status |
|---|---|
| Locked 20-patient test set | **never run** — `RUN_LOCKED_TEST = False` (eval-f0 cell 12), `RUN_TEST_EVALUATION = False` (train-f* cell 47) |
| Ablations A0–A7 | code exists (train-f0 cell 52) but **deliberately not run for this paper** — an 8-config study at even a reduced 12-epoch budget is ~14–15 GPU-hours, more than the time available. `RUN_ABLATIONS = False`. See [`ABLATION_STUDY.md`](ABLATION_STUDY.md) for the full design (what each of the 8 configs isolates), a result-table template, and the exact plan to run it. |
| ROC / AUC curves | **executed** — see §3.5. Ventricle 0.9992, Normal-WMH 0.9968, Abnormal-WMH 0.9949 |
| Explainability (Seg-Grad-CAM) | **executed** — 3 panels (worst/median/best patient) embedded in the committed notebook, section 20 |

**For the paper:** ROC/AUC and Seg-Grad-CAM explainability are real, executed results (§3.5,
notebook sections 19–20) — cite them freely. The architecture-ablation study (A0–A7) was scoped
but not executed due to compute/time constraints; state that plainly rather than reporting
numbers for it.

## 6. Reproduce the tables

```bash
# after downloading MS3SEG_V27_FOLD0_VALIDATION_RESULTS.zip and unzipping it
python scripts/summarize_results.py --results-dir MS3SEG_V27_FOLD0_VALIDATION --out paper_tables.md
```
