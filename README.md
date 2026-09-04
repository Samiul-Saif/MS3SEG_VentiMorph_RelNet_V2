# MS3SEG VentiMorph-RelNet V2.7

Multi-class brain MRI segmentation of **ventricles**, **normal white-matter hyperintensities (nWMH)**
and **abnormal white-matter hyperintensities (abWMH)** from co-registered FLAIR / T1 / T2, with an
explicit ventricle-relation module (VentiMorph refiner) and a relation-conditioned prototype bank
(RelNet). Trained and evaluated on the **MS3SEG** cohort (100 patients) with patient-level 5-fold
cross-validation and a locked 20-patient hold-out test set.

This repository contains the exact Kaggle notebooks used to produce the results in the paper:

| Notebook | Purpose |
|---|---|
| `notebook381a28a764%20(2).ipynb` | **Fold-0 training** (also builds the 5-fold split and the audit files; produces `best_model.pt` used for evaluation) |
| `ms3seg-ventimorph-relnet-v2-7-fold1-training (1).ipynb` | Fold-1 training (reuses the saved split) |
| `ms3seg-ventimorph-relnet-v2-7-fold2-training (1).ipynb` | Fold-2 training |
| `ms3seg-ventimorph-relnet-v2-7-fold3-training (1).ipynb` | Fold-3 training |
| `ms3seg-ventimorph-relnet-v2-7-fold4-training (1).ipynb` | Fold-4 training |
| `ms3seg-ventimorph-relnet-v2-7-fold0-validation-eva.ipynb` | **Evaluation only** – loads `best_model.pt`, scores the 16 Fold-0 validation patients as full 3D volumes, writes every metric CSV and figure used in the paper |

The locked 20-patient test set is **never** evaluated in any notebook here (`RUN_LOCKED_TEST = False`).

---

## 1. Environment — Python and framework versions

All notebooks were run **on Kaggle** with the **GPU** accelerator image (2025). No custom Docker
image, no extra `pip install` is required — every dependency below is already present in the Kaggle
GPU image. The one commented install line in the notebooks (`nibabel`, `surface-distance`) is not
needed: `nibabel` is pre-installed and surface distances are computed with SciPy.

| Component | Version used |
|---|---|
| OS | Ubuntu 22.04 (Kaggle GPU image) |
| Python | 3.11 |
| PyTorch | 2.6.x (CUDA 12.x build; `torch.cuda.get_arch_list()` → `sm_70…sm_120`) |
| torchvision | 0.21.x |
| NumPy | 1.26.x |
| pandas | 2.2.x |
| SciPy | 1.15.x |
| scikit-learn | 1.2.x |
| nibabel | 5.x |
| matplotlib | 3.9–3.10 |

> These are the versions shipped by the Kaggle GPU image at run time. The notebooks do **not** pin
> versions, so to capture the exact versions of your own run execute
> [`scripts/print_env.py`](scripts/print_env.py) as the first cell and paste the output into your
> paper's reproducibility section. `requirements.txt` gives minimum-compatible floors for running
> outside Kaggle; `requirements-lock.txt` records the approximate Kaggle image.

Fixed global seed: **42** (`random`, `numpy`, `torch`, `torch.cuda`, `PYTHONHASHSEED`).
`cudnn.deterministic = False`, `cudnn.benchmark = True` (throughput over bit-exact determinism).

---

## 2. Hardware used

| Item | Value |
|---|---|
| Platform | Kaggle Notebooks (free tier) |
| GPU | **1 × NVIDIA Tesla T4**, 16 GB, compute capability `sm_75` |
| Mixed precision | `torch.amp` autocast + `GradScaler` on CUDA |
| CPU / RAM | Kaggle default (4 vCPU, ~29 GB) |
| DataLoader | `num_workers=0`, `pin_memory=False`, `persistent_workers=False` (deliberate — a multiprocessing worker stall on Kaggle previously interrupted a run) |
| Batch size | 4 (2.5-D slices, 9 × 256 × 256) |
| Approx. wall-clock | ≈ 6–12 min / epoch on T4; a full fold (pilot 10 ep + up to 60 ep, with early stopping) ≈ 3–6 h |

An earlier attempt on a Tesla **P100** (`sm_60`) failed because that Kaggle PyTorch build was not
compiled for `sm_60`; every notebook now contains a GPU-compatibility guard that blocks training
unless a supported GPU (T4 or newer) is selected.

---

## 3. How to run

### 3.1 Data — Kaggle inputs

Add these two datasets to the notebook (**Add Input**):

1. **`MS3SEG`** – the preprocessed cohort. Must contain
   `MS_100_patient_preprocessed/` (per-patient `NNN_FLAIR.nii(.gz)`, `NNN_T1WI_reg.nii(.gz)`,
   `NNN_T2WI_reg.nii(.gz)`) and `MS_100_patient_masks/` with `Vent_Masks/`, `nWMH_Masks/`,
   `abWMH_Masks/` (`NNN_Vent_Mask`, `NNN_nWMH_Mask`, `NNN_abWMH_Mask`; `{0,1}` or `{0,255}` both accepted).
2. **`ms3seg-v27-best-checkpoint`** – a private dataset holding
   `patient_level_5fold_split.csv` and (for evaluation) `best_model.pt`.
   Needed by folds 1–4 and by the evaluation notebook so that **every fold reuses the identical
   patient-level split**. Fold-0 training regenerates this split from scratch (seed 42) — see §3.4.

Paths are auto-discovered; the notebooks search `/kaggle/input/**` for the required folder names, so
the exact dataset slug does not matter.

### 3.2 Train one fold

1. Open the fold notebook, **Settings → Accelerator → GPU T4 ×1**.
2. **Run All.** Each notebook runs a 10-epoch stability *pilot* first, self-checks numerical
   stability, then **resumes the same run** and continues to a maximum of 60 epochs with early
   stopping (patience 12), LR plateau scheduling, collapse detection and up to 2 automatic rollbacks.
3. Checkpoints written to `/kaggle/working/.../<run_name>/`:
   `best_model.pt` (best *balanced* score), `best_weighted_model.pt`, `best_nwmh_model.pt`,
   `last_checkpoint.pt` (resumable), `history.csv`.
4. Download `/kaggle/working/MS3SEG_V27_FOLD<k>_TRANSFER.zip` (also refreshed every 5 epochs and on
   every new best) and/or use **Save Version** to persist outputs.
5. `RUN_TEST_EVALUATION` stays `False`.

To **resume** an interrupted run: re-add the transfer ZIP as an input (or keep the working dir),
Run All again — `auto_resume` picks up from `last_checkpoint.pt`.

#### Ablation study (Table-7/8-style architecture ablation, A0–A7)

`notebook381a28a764%20(2).ipynb` (Fold-0 training) has `RUN_ABLATIONS = True` with
`ABLATION_MAX_EPOCHS = 12` — each of the 8 configs trains for 12 epochs (instead of the full
60) on Fold-0's train/val split only (locked test never touched), which the cell estimates at
**~14–15 GPU-hours total** on a T4 (8 configs × 12 epochs × ~9 min/epoch). That's most of a
week's free Kaggle GPU quota, so budget accordingly, or run it over a few sessions (it iterates
config-by-config and writes `ablation_summary.csv` at the end — it is not resumable
mid-ablation, so if a session is interrupted you restart from A0). Raise `ABLATION_MAX_EPOCHS`
for a longer, stricter study, or set `RUN_ABLATIONS = False` to skip it.

### 3.3 Evaluate (paper numbers)

1. Open `ms3seg-ventimorph-relnet-v2-7-fold0-validation-eva.ipynb`.
2. Add `MS3SEG` and `ms3seg-v27-best-checkpoint` (the latter must contain `best_model.pt` +
   `patient_level_5fold_split.csv`), select **T4 GPU**, keep `RUN_LOCKED_TEST = False`.
3. **Run All.** Outputs land in `/kaggle/working/MS3SEG_V27_FOLD0_VALIDATION/` and are zipped to
   `MS3SEG_V27_FOLD0_VALIDATION_RESULTS.zip`:
   - `fold0_validation_patient_metrics.csv` – per-patient voxel + lesion metrics
   - `fold0_validation_lesion_metrics.csv` – per-patient per-class lesion detection
   - `fold0_validation_class_summary.csv` – mean/std/median/min/max per class per metric
   - `fold0_validation_overall_summary.csv` – headline numbers
   - `fold0_validation_confusion_matrix.csv` – 4×4 voxel confusion
   - `figures/` – Dice box plot, normalized confusion matrix, best/median/worst qualitative
     panels, per-class ROC curves, and Seg-Grad-CAM explainability panels
   - `fold0_validation_roc_auc.csv` – per-class voxel-level AUC
   - `evaluation_metadata.json` – checkpoint, split, decision rule, lesion-matching settings

   The notebook also computes, in sections 19–20 (no extra Kaggle inputs needed):
   - **ROC / AUC** — per-class (ventricle/nWMH/abWMH/background) voxel-level, one-vs-rest,
     from the already-cached softmax probabilities.
   - **Explainability (Seg-Grad-CAM)** — gradient-based class-activation maps for the
     worst/median/best patients, showing which input regions drove each class's prediction.
     Model weights stay frozen; only the input's gradient is used.

### 3.4 Regenerate the 5-fold split outside a notebook

```bash
python scripts/make_5fold_split.py --audit path/to/integrity_audit.csv --out patient_level_5fold_split.csv
```

This reproduces exactly the logic in the Fold-0 notebook: burden-quartile stratification on abWMH
volume → 80/20 development/locked-test `train_test_split` (seed 42) → `StratifiedKFold(5, shuffle,
seed 42)` on the 80 development patients. `integrity_audit.csv` is produced by the audit cell of any
training notebook (or by hand: columns `patient_id`, `abwmh_voxels`, `spacing_x/y/z`).

### 3.5 Rebuild the paper tables from the result CSVs

```bash
python scripts/summarize_results.py --results-dir MS3SEG_V27_FOLD0_VALIDATION --out paper_tables.md
```

---

## 4. Model summary

- **Input:** 9-channel 2.5-D — for each axial slice, a ±1 neighbour triplet of FLAIR, registered T1,
  registered T2 (3 + 3 + 3), z-scored with 0.5–99.5 percentile clipping.
- **Backbone:** per-modality stems → softmax **modality-gated fusion** → 5-level residual U-Net
  (GroupNorm + SiLU + SE), an **Adaptive Anisotropic Context** dilated block at the bottleneck.
- **VentiMorph refiner** (`guidance_mode="directional"`) at the 128-ch and 32-ch decoder stages:
  predicts a ventricle probability, derives near/mid/far soft-distance bands and a radial/tangential
  orientation field, feeds them back as guidance features.
- **RelNet:** relation-conditioned prototype bank (2 classes × 3 distance bands × 32-D) with an
  InfoNCE-style loss separating nWMH vs abWMH embeddings per periventricular band.
- **Uncertainty:** evidential (Dirichlet) head, softplus evidence → α, predictive uncertainty
  `4 / Σα`.
- **Composite loss:** class-weighted CE + soft Dice + ramped focal-Tversky (nWMH) + focal-Tversky
  (abWMH) + deep ventricle supervision (BCE-with-logits + Dice, mid + final) + boundary L1 +
  ramped nWMH/abWMH confusion penalty (periventricular-weighted) + prototype loss + evidential loss.
- **Optimizer:** AdamW (lr 1e-4, wd 1e-4, β=(0.9, 0.99)), `ReduceLROnPlateau(mode="max")`,
  grad-clip 2.0, AMP.
- **Trainable parameters:** **5,974,441** (base=24, emb_dim=32, 4 classes).
- **Decision rule at inference:** softmax → argmax, **no post-processing**.

Full hyper-parameters: [`configs/experiment_v27.yaml`](configs/experiment_v27.yaml).
Data / preprocessing constants: [`configs/data.yaml`](configs/data.yaml).
These YAML files mirror the in-notebook `ExperimentConfig` dataclass and constants — they are
documentation of the frozen configuration, not a separate code path.

---

## 5. Results and where each number comes from

The numbers below are also committed as CSVs under [`results/`](results/) (extracted straight from
the notebook outputs — `results/README.md` explains how). See
[`docs/RESULTS_MAP.md`](docs/RESULTS_MAP.md) for the full mapping (notebook → cell → output file →
metric). Headline Fold-0 validation numbers (16 patients, 3-D patient-level, checkpoint `best_model.pt`,
epoch 47):

| Metric | Value | Source |
|---|---|---|
| Mean foreground Dice | 0.755 ± 0.046 | eval nb, cell 13 → `fold0_validation_overall_summary.csv` |
| Ventricle Dice | 0.852 ± 0.025 | same |
| Normal-WMH Dice | 0.636 ± 0.079 | same |
| Abnormal-WMH Dice | 0.778 ± 0.089 | same |
| Abnormal-WMH lesion F1 | 0.692 | same |
| Normal-WMH lesion F1 | 0.473 | same |
| Periventricular abWMH FP rate | 0.258 | same |
| Ventricle / nWMH / abWMH HD95 (mm) | 3.53 / 9.84 / 11.47 | eval nb cell 13 → `fold0_validation_class_summary.csv` |
| Ventricle / nWMH / abWMH ASSD (mm) | 0.58 / 1.83 / 1.92 | same |

Per-fold validation Dice (slice-level, from each training notebook's `history.csv`, best *balanced*
epoch):

| Fold | best epoch | Ventricle | nWMH | abWMH | mean FG |
|---|---|---|---|---|---|
| 0 | 47 | 0.856 | 0.641 | 0.794 | 0.764 |
| 1 | 19 | 0.837 | 0.627 | 0.754 | 0.739 |
| 2 | 58 | 0.850 | 0.640 | 0.744 | 0.745 |
| 3 | 33 | 0.855 | 0.643 | 0.790 | 0.763 |
| 4 | 43 | 0.851 | 0.616 | 0.768 | 0.745 |

> Slice-level Dice (training-loop confusion matrix) and 3-D patient-level Dice (evaluation notebook)
> are computed differently and are not identical; the paper's primary numbers are the 3-D
> patient-level ones from the evaluation notebook. Folds 3 and 4 did not pass the *pilot* stability
> gate (transient gradient spikes) but completed full training; see their `history.csv`.

**Table 7 ("VentiMorph-RelNet (ours)" row).**
[`notebooks/MS3SEG_5Fold_Best_Points_Summary.ipynb`](notebooks/MS3SEG_5Fold_Best_Points_Summary.ipynb)
scrapes the selected per-fold Dice straight out of the five source notebooks' saved outputs
(Fold-1 = the 3-D `fold0-validation-eva` summary; Folds 2-5 = each training notebook's best
balanced-score epoch), then averages → Ventricle **0.8489**, nWMH **0.6325**, abWMH **0.76679**,
Mean FG **0.7494**. Runs offline in seconds. Its *Notes* explain the Fold-1-vs-Folds-2-5 metric
mix.

**If your paper reports a number not in the table above, add a row to `docs/RESULTS_MAP.md` pointing
at the notebook cell / CSV that produced it.**

---

## 6. Repository layout

```
.
├── README.md
├── requirements.txt              # minimum floors to run outside Kaggle
├── requirements-lock.txt         # approximate Kaggle GPU image (2025)
├── .gitignore
├── configs/
│   ├── experiment_v27.yaml       # all training hyper-parameters (mirrors ExperimentConfig)
│   ├── data.yaml                 # dataset layout, preprocessing, split, class stats
│   └── environment.md            # exact hardware / software notes
├── docs/
│   └── RESULTS_MAP.md            # every paper result → notebook cell → output file
├── notebooks/
│   ├── README.md
│   └── MS3SEG_5Fold_Best_Points_Summary.ipynb  # scrapes 5 source notebooks → Table 7 row
├── results/                      # numbers extracted from the committed notebook outputs
│   ├── README.md
│   ├── history/fold{0..4}_history.csv
│   └── fold0_validation/{patient_dice,overall_summary,class_summary}.csv
├── scripts/
│   ├── print_env.py              # run first on Kaggle to capture exact versions
│   ├── make_5fold_split.py       # standalone reproduction of the patient-level split
│   ├── extract_from_notebooks.py # rebuild results/ from the notebook outputs
│   └── summarize_results.py      # result CSVs → markdown tables
├── notebook381a28a764%20(2).ipynb
├── ms3seg-ventimorph-relnet-v2-7-fold1-training (1).ipynb
├── ms3seg-ventimorph-relnet-v2-7-fold2-training (1).ipynb
├── ms3seg-ventimorph-relnet-v2-7-fold3-training (1).ipynb
├── ms3seg-ventimorph-relnet-v2-7-fold4-training (1).ipynb
└── ms3seg-ventimorph-relnet-v2-7-fold0-validation-eva.ipynb
```

## 7. Data availability

The MS3SEG imaging data and masks are **not** in this repository (patient MRI). They are provided as
the private Kaggle dataset `MS3SEG`. Contact the authors for access.
