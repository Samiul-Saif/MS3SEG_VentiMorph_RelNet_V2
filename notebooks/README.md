# notebooks/

Analysis notebooks (post-processing only — no training / no GPU needed).

## `five_fold_aggregation.ipynb`

Reproduces the **VentiMorph-RelNet (ours)** row of **Table 7** and the
**VentiMorph-RelNet V2.7 (frozen calibrated)** row of **Table 8** from the per-fold
patient-level validation Dice.

- **Input:** the five per-fold class means (Ventricle / nWMH / abWMH), entered in section 1
  (VentiMorph, already filled from the per-fold evaluation) and section 2 (U-Net / U-Net++
  baselines — paste yours; falls back to the published Table-8 values if left blank).
  Fold-1 (= CV fold 0) is auto-checked against `../results/fold0_validation/overall_summary.csv`.
- **Output:** `../results/five_fold/ventimorph_per_fold.csv`,
  `table7_numerical_comparison.csv`, `table8_five_fold_ablation.csv`, and a cross-check of the
  computed mean/SD against the published paper values.
- **Result:** the class **means** reproduce the paper exactly — Ventricle ≈ 0.8490, nWMH 0.6325,
  abWMH 0.76679, Mean FG 0.7494. The sample SD of the five fold means is *smaller* than the ±
  printed in Table 8 (see the "Notes" cell) — confirm which dispersion definition the manuscript
  uses.

Run: open in Jupyter and *Run All*, or
`pip install jupyter && jupyter nbconvert --to notebook --execute --inplace notebooks/five_fold_aggregation.ipynb`.
