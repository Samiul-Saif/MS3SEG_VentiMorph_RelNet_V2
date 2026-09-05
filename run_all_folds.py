"""
Convenience orchestrator: trains and evaluates every requested model on
every requested fold, and prints/saves the cross-fold, cross-model summary
table your comparison section needs.

Two-stage workflow (recommended on a single RTX 4060 Ti 16GB):

1. Pilot screen everything on fold 0 only, with a short epoch budget, to
   catch OOMs / broken configs / obviously-bad models cheaply:

    python run_all_folds.py --data-root D:/MS3SEG --output-root runs_pilot \
        --folds 0 --epochs-ventricle 8 --epochs-tri 15 \
        --models unet unet_plus_plus attention_unet segresnet unetr swin_unetr anato_unetpp \
        --also-run-baseline

2. Full 5-fold runs, only for the models you decided are worth the GPU-hours
   after the pilot (see README.md for a rough time budget on a 4060 Ti):

    python run_all_folds.py --data-root D:/MS3SEG --split-csv official_split.csv \
        --output-root runs_full --folds 0 1 2 3 4 \
        --epochs-ventricle 20 --epochs-tri 60 \
        --models unet unet_plus_plus anato_unetpp --also-run-baseline

3. For side comparators where you only want the plain-baseline number (skip
   the with-anatomy-channel pass entirely, roughly half the time):

    python run_all_folds.py ... --models attention_unet segresnet --baseline-only
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

from baselines import MODEL_NAMES, RECOMMENDED_BATCH_SIZE


def run(cmd: list[str]):
    print("\n>>>", " ".join(str(c) for c in cmd))
    subprocess.run([sys.executable] + cmd, check=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, default=None,
                    help="MS3SEG root folder. If omitted, auto-discovered under /kaggle/input "
                         "(Kaggle) or the current directory.")
    p.add_argument("--split-csv", type=Path, default=None)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--epochs-ventricle", type=int, default=20)
    p.add_argument("--epochs-tri", type=int, default=60)
    p.add_argument("--base-channels", type=int, default=32)
    p.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--models", nargs="+", default=["anato_unetpp"], choices=sorted(MODEL_NAMES),
                    help="Which tri-mask architectures to run, each with the anatomy channel on.")
    p.add_argument("--cascade", action="store_true", default=True,
                    help="Use the fully automatic Stage-1-prediction pipeline (default) instead "
                         "of ground-truth ventricle masks for the anatomical channel.")
    p.add_argument("--oracle-ventricle", dest="cascade", action="store_false",
                    help="Use ground-truth ventricle masks for the anatomy channel instead of "
                         "the Stage-1 model's own predictions (upper-bound ablation only).")
    p.add_argument("--also-run-baseline", action="store_true",
                    help="For every --models entry, also train the same architecture with "
                         "--no-ventricle-channel, for a clean with/without-anatomy-channel table.")
    p.add_argument("--baseline-only", action="store_true",
                    help="Skip the with-anatomy-channel (cascade) pass entirely and train only "
                         "the --no-ventricle-channel variant for each --models entry. For side "
                         "comparators where you only want the plain-baseline number, at roughly "
                         "half the time of running both variants.")
    args = p.parse_args()
    if args.baseline_only and args.also_run_baseline:
        raise ValueError("--baseline-only and --also-run-baseline are mutually exclusive.")

    all_rows = []
    for fold in args.folds:
        fold_dir = args.output_root / f"fold{fold}"
        common = ["--fold", str(fold)]
        if args.data_root:
            common += ["--data-root", str(args.data_root)]
        if args.split_csv:
            common += ["--split-csv", str(args.split_csv)]

        vent_ckpt = fold_dir / "ventricle_net_ventricle_best.pt"
        run(["train.py", "--stage", "ventricle", "--output-dir", str(fold_dir),
             "--epochs", str(args.epochs_ventricle), "--base-channels", str(args.base_channels)] + common)

        for model_name in args.models:
            batch_size = RECOMMENDED_BATCH_SIZE.get(model_name, 8)

            anatomy_ckpt = fold_dir / f"{model_name}_anatomy_tri_best.pt"
            baseline_ckpt = fold_dir / f"{model_name}_baseline_tri_best.pt"

            if not args.baseline_only:
                tri_args = ["train.py", "--stage", "tri", "--model", model_name,
                            "--output-dir", str(fold_dir), "--epochs", str(args.epochs_tri),
                            "--base-channels", str(args.base_channels),
                            "--batch-size", str(batch_size)] + common
                if args.cascade:
                    tri_args += ["--cascade", "--ventricle-checkpoint", str(vent_ckpt)]
                run(tri_args)

                eval_csv = fold_dir / f"eval_{model_name}_anatomy_val_fold{fold}.csv"
                eval_args = ["evaluate.py", "--model", model_name,
                             "--tri-checkpoint", str(anatomy_ckpt),
                             "--base-channels", str(args.base_channels), "--split", "val",
                             "--output-csv", str(eval_csv)] + common
                if args.cascade:
                    eval_args += ["--cascade", "--ventricle-checkpoint", str(vent_ckpt)]
                run(eval_args)
                df = pd.read_csv(eval_csv)
                df["fold"] = fold
                df["anatomy_channel"] = True
                all_rows.append(df)

            if args.baseline_only or args.also_run_baseline:
                base_tri_args = ["train.py", "--stage", "tri", "--model", model_name,
                                  "--output-dir", str(fold_dir), "--no-ventricle-channel",
                                  "--epochs", str(args.epochs_tri), "--base-channels", str(args.base_channels),
                                  "--batch-size", str(batch_size)] + common
                run(base_tri_args)
                base_eval_csv = fold_dir / f"eval_{model_name}_baseline_val_fold{fold}.csv"
                base_eval_args = ["evaluate.py", "--model", model_name, "--no-ventricle-channel",
                                   "--tri-checkpoint", str(baseline_ckpt),
                                   "--base-channels", str(args.base_channels), "--split", "val",
                                   "--output-csv", str(base_eval_csv)] + common
                run(base_eval_args)
                dfb = pd.read_csv(base_eval_csv)
                dfb["fold"] = fold
                dfb["anatomy_channel"] = False
                all_rows.append(dfb)

    combined = pd.concat(all_rows, ignore_index=True)
    args.output_root.mkdir(parents=True, exist_ok=True)
    combined.to_csv(args.output_root / "all_folds_summary.csv", index=False)
    print("\n=== Cross-fold summary (mean over patients and folds) ===")
    print(combined.groupby(["model", "anatomy_channel"])[
        ["mean_fg_dice", "ventricle_dice", "normal_wmh_dice", "abnormal_wmh_dice"]
    ].mean().sort_values("mean_fg_dice", ascending=False))


if __name__ == "__main__":
    main()
