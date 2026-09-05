# Ablation study — design, status, and plan

**Status: designed and code-ready; not executed.** No ablation numbers are reported anywhere in
this repository or the paper. This document exists so the study can be evaluated on its design
and so its exact status is stated once, clearly, rather than implied.

## What the study tests

VentiMorph-RelNet V2.7 is built by adding architectural components on top of a plain multimodal
2.5-D U-Net one at a time. The ablation isolates each addition's contribution by re-training with
only that subset of components enabled, all on the same data, split, and optimizer settings.

| Config | Multimodal (T1+T2) | Context block | Ventricle-relation guidance | Prototype bank | Uncertainty head | What it isolates |
|---|:---:|:---:|:---:|:---:|:---:|---|
| `A0_flair_2p5d` | – | – | none | – | – | Baseline: FLAIR-only 2.5-D U-Net |
| `A1_multimodal` | ✓ | – | none | – | – | + adding registered T1/T2 via modality-gated fusion |
| `A2_context` | ✓ | ✓ | none | – | – | + Adaptive Anisotropic Context (dilated bottleneck) |
| `A3_vent_aux` | ✓ | ✓ | vent | – | – | + a plain ventricle auxiliary head (no distance/direction) |
| `A4_distance` | ✓ | ✓ | distance | – | – | + near/mid/far ventricle-distance bands |
| `A5_directional` | ✓ | ✓ | directional | – | – | + radial/tangential orientation relative to the ventricle |
| `A6_prototypes` | ✓ | ✓ | directional | ✓ | – | + relation-conditioned nWMH/abWMH prototype bank |
| `A7_full` | ✓ | ✓ | directional | ✓ | ✓ | Full model (as reported everywhere else in this repo) |

Each config is trained and evaluated identically (same Fold-0 train/val split, same optimizer,
same loss) so that the *only* variable across rows is architecture. This lets a reader see, for
example, whether the directional guidance (A5) meaningfully beats plain distance bands (A4), or
whether the prototype bank (A6) earns its complexity over A5.

## Where the code lives

`notebook381a28a764%20(2).ipynb`, near the end (the `ABLATIONS = {...}` cell). It trains each
config with `train_experiment()` (the same function used for the main Fold-0 run) and evaluates
each on the 16 Fold-0 validation patients with `evaluate_patients()` (the same 3-D patient-level
metrics used everywhere else in this repo) — locked test set never touched.

## Why it has not been run

`RUN_ABLATIONS = False`. Even capped to `ABLATION_MAX_EPOCHS = 12` (rather than the full
60-epoch budget used elsewhere), training all 8 configs costs an estimated **~14–15 GPU-hours**
on a Kaggle T4 (8 configs × 12 epochs × ~9 min/epoch) — more time than was available before
submission.

## Result table (template — to be filled in once run, not before)

| Config | Ventricle Dice | nWMH Dice | abWMH Dice | HD95 abWMH (mm) | Lesion F1 (abWMH) | Periventricular FP rate |
|---|---|---|---|---|---|---|
| A0_flair_2p5d | *pending* | *pending* | *pending* | *pending* | *pending* | *pending* |
| A1_multimodal | *pending* | *pending* | *pending* | *pending* | *pending* | *pending* |
| A2_context | *pending* | *pending* | *pending* | *pending* | *pending* | *pending* |
| A3_vent_aux | *pending* | *pending* | *pending* | *pending* | *pending* | *pending* |
| A4_distance | *pending* | *pending* | *pending* | *pending* | *pending* | *pending* |
| A5_directional | *pending* | *pending* | *pending* | *pending* | *pending* | *pending* |
| A6_prototypes | *pending* | *pending* | *pending* | *pending* | *pending* | *pending* |
| A7_full | *pending* | *pending* | *pending* | *pending* | *pending* | *pending* |

Columns match exactly what `evaluate_patients()` already computes and what
`ablation_summary.csv` will contain once the cell is run — this table is here so the eventual
result slots directly in without redesigning anything.

## Plan to complete it

1. Open `notebook381a28a764%20(2).ipynb` on Kaggle (same `MS3SEG` + `ms3seg-v27-best-checkpoint`
   inputs already used for everything else).
2. Set `RUN_ABLATIONS = True`.
3. Run All. It iterates configs A0→A7 sequentially and is **not resumable mid-ablation** — budget
   an uninterrupted block of time, or expect to restart from A0 if the session drops. Consider
   lowering `ABLATION_MAX_EPOCHS` further (e.g. to 8) for a cheaper first pass if GPU-hours are
   tight; the stability pilot elsewhere in this repo shows foreground classes separating within
   ~10 epochs, so even a short budget should be directionally informative.
4. It writes `ablation_summary.csv` to the Kaggle working directory — download it, replace the
   *pending* cells above with the real numbers, and update `docs/RESULTS_MAP.md` §5 to move this
   item from "not run" to "executed."
