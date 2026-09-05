"""
Patient-level, full-volume evaluation with the same three metrics the
MS3SEG paper reports (Eqs. 1-3): volumetric Dice, IoU, and 95th-percentile
Hausdorff distance in mm, computed per class over the whole reconstructed
3D volume (not averaged per-slice), using each patient's true voxel
spacing. This makes the numbers directly comparable to Tables 7-9 of the
dataset paper and to the fold-0 numbers already collected from the
capacity-matched U-Net / U-Net++ / MSR / SARR baselines.

    python evaluate.py --data-root /path/to/MS3SEG --split-csv ... \
        --fold 0 --tri-checkpoint runs/fold0/tri_best.pt \
        --ventricle-checkpoint runs/fold0/ventricle_best.pt --cascade
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import nibabel as nib
import torch
import torch.nn.functional as F
from scipy import ndimage

from data import (
    build_manifest, compose_target, load_normalized_volume,
    load_or_build_split, ventricle_distance_from_prob, ventricle_distance_volume,
)
from model import VentricleNet
from baselines import MODEL_NAMES, build_model, unpack_output
from train import load_checkpoint

CLASS_NAMES = {1: "ventricle", 2: "normal_wmh", 3: "abnormal_wmh"}


def _triplet(volume, z):
    depth = volume.shape[2]
    idx = [max(0, z - 1), z, min(depth - 1, z + 1)]
    return np.stack([volume[:, :, i] for i in idx], axis=0)


def _single_slice(volume, z):
    """1-channel extraction for the ventricle-proximity map - see
    data.MS3SEGDataset._single_slice for why this must not be a 3-slice `_triplet`."""
    return volume[:, :, z][None, :, :]


def hd95_mm(pred: np.ndarray, gt: np.ndarray, spacing) -> float:
    if pred.sum() == 0 and gt.sum() == 0:
        return 0.0
    if pred.sum() == 0 or gt.sum() == 0:
        return float("nan")
    pred_surface = pred & ~ndimage.binary_erosion(pred)
    gt_surface = gt & ~ndimage.binary_erosion(gt)
    dt_gt = ndimage.distance_transform_edt(~gt_surface, sampling=spacing)
    dt_pred = ndimage.distance_transform_edt(~pred_surface, sampling=spacing)
    d_pred_to_gt = dt_gt[pred_surface]
    d_gt_to_pred = dt_pred[gt_surface]
    return float(max(np.percentile(d_pred_to_gt, 95), np.percentile(d_gt_to_pred, 95)))


def dice_iou(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-6):
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    dice = (2 * inter + eps) / (pred.sum() + gt.sum() + eps)
    iou = (inter + eps) / (union + eps)
    return float(dice), float(iou)


@torch.no_grad()
def predict_volume(model_tri, row, device, use_ventricle_channel, cascade, ventricle_model, batch_size=8):
    flair = load_normalized_volume(row.flair)
    t1 = load_normalized_volume(row.t1)
    t2 = load_normalized_volume(row.t2)
    depth = flair.shape[2]

    if use_ventricle_channel and not cascade:
        dist_vol = ventricle_distance_volume(row.vent)
    elif use_ventricle_channel and cascade:
        dist_vol = np.zeros_like(flair)
        for z in range(depth):
            flair_mid = torch.from_numpy(flair[:, :, z]).float()[None, None].to(device)
            prob = torch.sigmoid(ventricle_model(flair_mid)).squeeze().cpu().numpy()
            dist_vol[:, :, z] = ventricle_distance_from_prob(prob, spacing_xy=(1.0, 1.0))
    else:
        dist_vol = None

    pred_vol = np.zeros_like(flair, dtype=np.uint8)
    for start in range(0, depth, batch_size):
        zs = list(range(start, min(start + batch_size, depth)))
        images = []
        for z in zs:
            channels = [_triplet(flair, z), _triplet(t1, z), _triplet(t2, z)]
            if dist_vol is not None:
                channels.append(_single_slice(dist_vol, z))
            images.append(np.concatenate(channels, axis=0))
        batch = torch.from_numpy(np.stack(images)).float().to(device)
        out = model_tri(batch)
        logits, _ = unpack_output(out)
        pred = logits.argmax(1).cpu().numpy().astype(np.uint8)
        for i, z in enumerate(zs):
            pred_vol[:, :, z] = pred[i]
    return pred_vol


def evaluate_patients(model_tri, manifest, patient_ids, device, use_ventricle_channel,
                       cascade=False, ventricle_model=None, model_name=None) -> pd.DataFrame:
    model_tri.eval()
    table = manifest.set_index("patient_id")
    rows = []
    for pid in patient_ids:
        pid = str(pid).zfill(3)
        row = table.loc[pid]
        gt = compose_target(row)
        pred = predict_volume(model_tri, row, device, use_ventricle_channel, cascade, ventricle_model)
        spacing = nib.load(row.flair).header.get_zooms()[:3]

        record = {"patient_id": pid, "model": model_name}
        fg_dices = []
        for class_id, name in CLASS_NAMES.items():
            p = pred == class_id
            g = gt == class_id
            dice, iou = dice_iou(p, g)
            hd95 = hd95_mm(p, g, spacing)
            record[f"{name}_dice"] = dice
            record[f"{name}_iou"] = iou
            record[f"{name}_hd95_mm"] = hd95
            fg_dices.append(dice)
        record["mean_fg_dice"] = float(np.mean(fg_dices))
        rows.append(record)
        print(f"{pid} | FG={record['mean_fg_dice']:.4f} | "
              f"V={record['ventricle_dice']:.4f} N={record['normal_wmh_dice']:.4f} A={record['abnormal_wmh_dice']:.4f}")

    df = pd.DataFrame(rows)
    print("\n=== Summary (mean +/- std over patients) ===")
    for col in df.columns:
        if col in ("patient_id", "model") or not pd.api.types.is_numeric_dtype(df[col]):
            continue
        print(f"{col}: {df[col].mean():.4f} +/- {df[col].std():.4f}")
    return df


def build_arg_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, default=None,
                    help="MS3SEG root folder. If omitted, auto-discovered under /kaggle/input "
                         "(Kaggle) or the current directory.")
    p.add_argument("--split-csv", type=Path, default=None)
    p.add_argument("--fold", type=int, required=True)
    p.add_argument("--model", choices=sorted(MODEL_NAMES), default="anato_unetpp")
    p.add_argument("--tri-checkpoint", type=Path, required=True)
    p.add_argument("--base-channels", type=int, default=32)
    p.add_argument("--ventricle-channel", dest="use_ventricle_channel", action="store_true", default=True)
    p.add_argument("--no-ventricle-channel", dest="use_ventricle_channel", action="store_false")
    p.add_argument("--cascade", action="store_true")
    p.add_argument("--ventricle-checkpoint", type=Path, default=None)
    p.add_argument("--split", choices=["val", "test"], default="val")
    p.add_argument("--output-csv", type=Path, default=None)
    return p


def parse_args():
    return build_arg_parser().parse_args()


def run(args) -> pd.DataFrame:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifest = build_manifest(args.data_root)
    split_df = load_or_build_split(manifest, args.split_csv, args.tri_checkpoint.parent)

    if args.split == "val":
        patient_ids = split_df[(split_df.fold == args.fold) & (split_df.role == "val")].patient_id.tolist()
    else:
        patient_ids = split_df[split_df.role == "test"].patient_id.tolist()
        print("Evaluating the LOCKED TEST SET. Only do this once, on your final chosen model.")

    in_channels = 10 if args.use_ventricle_channel else 9
    model = build_model(args.model, in_channels=in_channels, num_classes=4,
                         base_channels=args.base_channels, deep_supervision=True).to(device)
    load_checkpoint(args.tri_checkpoint, model)

    ventricle_model = None
    if args.cascade:
        ventricle_model = VentricleNet(base=args.base_channels // 4 if args.base_channels >= 16 else 8).to(device)
        load_checkpoint(args.ventricle_checkpoint, ventricle_model)
        ventricle_model.eval()

    df = evaluate_patients(model, manifest, patient_ids, device, args.use_ventricle_channel,
                            cascade=args.cascade, ventricle_model=ventricle_model, model_name=args.model)
    out_csv = args.output_csv or (args.tri_checkpoint.parent / f"eval_{args.model}_{args.split}_fold{args.fold}.csv")
    df.to_csv(out_csv, index=False)
    print("Saved:", out_csv)
    return df


def main():
    run(parse_args())


if __name__ == "__main__":
    main()
