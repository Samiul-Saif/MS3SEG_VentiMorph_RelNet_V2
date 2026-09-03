# notebooks/

## `MS3SEG_5Fold_Best_Points_Summary.ipynb`

Builds the 5-fold validation table for the paper's **Table 7** ("VentiMorph-RelNet (ours)" row)
by **reading the numbers out of the five source `.ipynb` files' saved cell outputs** — nothing
is typed in. It then sums, averages, picks the best fold per metric, and cross-checks against
the published Table 7.

| Sheet fold | Source notebook | Value scraped |
|---|---|---|
| Fold-1 | `ms3seg-…-fold0-validation-eva.ipynb` | patient-level 3-D validation summary (`overall_summary`) |
| Fold-2 | `ms3seg-…-fold1-training (1).ipynb` | validation Dice at the best balanced-score epoch |
| Fold-3 | `ms3seg-…-fold2-training (1).ipynb` | " |
| Fold-4 | `ms3seg-…-fold3-training (1).ipynb` | " |
| Fold-5 | `ms3seg-…-fold4-training (1).ipynb` | " |

Result (matches the paper exactly): Ventricle **0.8489**, nWMH **0.6325**, abWMH **0.76679**,
Mean FG **0.7494**.

- Runs offline in seconds (`pip install jupyter pandas`, Run All) — it only parses notebook JSON,
  no data / GPU / model needed.
- The source notebooks are found automatically (searches the repo dir and `~/Downloads`).
- `Mean F1` is the handwritten-sheet label; it is the mean foreground Dice
  (`mean_patient_foreground_dice` for Fold-1, `val_mean_fg_dice` for Folds 2-5). Fold-1 is a
  3-D patient-level Dice; Folds 2-5 are the training-loop (slice-level) validation Dice at the
  selected epoch — state this in the paper, or run a full 5-fold patient-level evaluation for
  strict consistency.
