"""
Single, uninterrupted training run per (stage, fold). No multi-session
resume dance, no RNG-state-on-the-wrong-device bug: checkpoints always save
RNG state as CPU tensors and are always loaded with map_location="cpu",
then moved to DEVICE explicitly.

Usage (run once per fold; repeat for fold in 0..4):

    # Stage 1 - ventricle prior (fast, ~15-20 epochs is usually enough)
    python train.py --stage ventricle --fold 0 --data-root /path/to/MS3SEG \
        --output-dir runs/fold0

    # Stage 2 - proposed AnatoUNet++, conditioned on the Stage-1 checkpoint
    python train.py --stage tri --model anato_unetpp --fold 0 --data-root /path/to/MS3SEG \
        --output-dir runs/fold0 \
        --ventricle-checkpoint runs/fold0/ventricle_best.pt --cascade

    # Any other baseline in baselines.MODEL_NAMES (unet, unet_plus_plus,
    # attention_unet, segresnet, unetr, swin_unetr), with or without the
    # anatomy channel:
    python train.py --stage tri --model unetr --fold 0 --data-root /path/to/MS3SEG \
        --output-dir runs/fold0_unetr --no-ventricle-channel --batch-size 6

See baselines.RECOMMENDED_BATCH_SIZE for per-model starting points on a
16GB card, and run_all_folds.py to sweep every model x every fold in one go.
"""

from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from data import (
    DatasetConfig, MS3SEGDataset, SEED, build_manifest, check_gpu_compatibility,
    compute_class_weights, load_or_build_split, seed_everything, ventricle_distance_from_prob,
)
from model import VentricleNet
from baselines import MODEL_NAMES, build_model, unpack_output
from losses import DiceCELoss, binary_dice_ce_loss


def build_arg_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=["ventricle", "tri"], required=True)
    p.add_argument("--model", choices=sorted(MODEL_NAMES), default="anato_unetpp",
                    help="Which tri-mask architecture to train (ignored for --stage ventricle, "
                         "which always uses the small VentricleNet). One of: "
                         + ", ".join(sorted(MODEL_NAMES)))
    p.add_argument("--fold", type=int, required=True)
    p.add_argument("--data-root", type=Path, default=None,
                    help="MS3SEG root folder. If omitted, auto-discovered under /kaggle/input "
                         "(Kaggle) or the current directory.")
    p.add_argument("--split-csv", type=Path, default=None,
                    help="Official MS3SEG patient_level_5fold_split.csv, if you have it. "
                         "Strongly recommended so results line up with the paper's own splits.")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--base-channels", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=2.0)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--deep-supervision", action="store_true", default=True)
    p.add_argument("--no-deep-supervision", dest="deep_supervision", action="store_false")
    p.add_argument("--lesion-tversky-weight", type=float, default=0.0,
                    help="Off by default. Set e.g. 0.3 to run the single controlled ablation "
                         "described in the write-up.")
    p.add_argument("--ventricle-channel", dest="use_ventricle_channel", action="store_true", default=True)
    p.add_argument("--no-ventricle-channel", dest="use_ventricle_channel", action="store_false",
                    help="Ablation / plain-baseline mode: no anatomical prior channel, "
                         "in_channels=9 instead of 10. Applies to whichever --model is selected, "
                         "so any architecture can be compared with/without the anatomy channel.")
    p.add_argument("--cascade", action="store_true",
                    help="tri stage only: build the ventricle channel from a frozen Stage-1 "
                         "model's *predictions* instead of ground truth, i.e. the fully "
                         "automatic, deployable pipeline. Requires --ventricle-checkpoint.")
    p.add_argument("--ventricle-checkpoint", type=Path, default=None)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--early-stopping-patience", type=int, default=15,
                    help="Stop once val_mean_fg_dice hasn't improved for this many epochs. "
                         "The best checkpoint is unaffected either way; this only saves GPU-hours "
                         "on models that converge early. 0 disables early stopping.")
    return p


def parse_args():
    return build_arg_parser().parse_args()


def default_args(**overrides):
    """Build an argparse.Namespace with defaults, for calling `run(...)` directly
    from a Jupyter cell instead of going through the command line. Example:

        from train import run, default_args
        run(default_args(stage="tri", model="unet", fold=0,
                          data_root="/data/MS3SEG", output_dir="runs/fold0_unet"))
    """
    parser = build_arg_parser()
    required = ["--stage", "--fold", "--output-dir"]
    args_list = []
    for key, value in overrides.items():
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            if value:
                args_list.append(flag)
        else:
            args_list += [flag, str(value)]
    ns = parser.parse_args(args_list)
    for r in required:
        dest = r.lstrip("-").replace("-", "_")
        if getattr(ns, dest, None) is None:
            raise ValueError(f"default_args(...) is missing required argument {r}")
    ns.data_root = Path(ns.data_root) if ns.data_root is not None else None
    ns.output_dir = Path(ns.output_dir)
    if ns.split_csv is not None:
        ns.split_csv = Path(ns.split_csv)
    if ns.ventricle_checkpoint is not None:
        ns.ventricle_checkpoint = Path(ns.ventricle_checkpoint)
    return ns


def save_checkpoint(path: Path, model, optimizer, scheduler, epoch, best_score, extra=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch": epoch,
        "best_score": best_score,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": [s.cpu() for s in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else None,
        "numpy_rng_state": np.random.get_state(),
        "extra": extra or {},
    }, path)


def load_checkpoint(path: Path, model, optimizer=None, scheduler=None):
    # weights_only=False: PyTorch >=2.6 defaults torch.load to weights_only=True,
    # which refuses to unpickle the non-tensor objects our checkpoints intentionally
    # carry (numpy/torch RNG state for exact resume). Safe here since we only ever
    # load checkpoints this same code just wrote.
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    if optimizer is not None and "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    if scheduler is not None and "scheduler_state" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state"])
    torch.set_rng_state(ckpt["torch_rng_state"].to(torch.uint8))
    if ckpt.get("cuda_rng_state_all") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.to(torch.uint8) for s in ckpt["cuda_rng_state_all"]])
    np.random.set_state(ckpt["numpy_rng_state"])
    return ckpt


def select_ventricle_input(image: torch.Tensor) -> torch.Tensor:
    """VentricleNet takes a single FLAIR mid-slice channel; MS3SEGDataset
    always returns the full 9-channel (FLAIR/T1/T2 triplet) tensor
    regardless of stage, so the ventricle stage needs to pick out channel
    index 1 (the middle slice of the FLAIR triplet, i.e. the "current"
    slice) itself."""
    return image[:, 1:2]


def apply_cascade_ventricle_channel(image: torch.Tensor, ventricle_model) -> torch.Tensor:
    """Batched replacement of the dataset's ground-truth-based ventricle-proximity
    channel (last channel) with the frozen Stage-1 model's own prediction - the
    fully automatic, deployable pipeline (--cascade). Done here, post-collation,
    in the main process, rather than inside Dataset.__getitem__: DataLoader
    workers are forked processes that cannot touch a CUDA context the parent
    already initialized, and doing it here also means the ventricle model runs
    once per batch instead of once per sample."""
    with torch.no_grad():
        prob = torch.sigmoid(ventricle_model(image[:, 1:2])).squeeze(1).cpu().numpy()
    dist = np.stack([ventricle_distance_from_prob(p, spacing_xy=(1.0, 1.0)) for p in prob])
    image = image.clone()
    image[:, -1] = torch.from_numpy(dist).float().to(image.device)
    return image


@torch.no_grad()
def slice_level_val_dice(model, loader, device, num_classes, is_tri: bool,
                          ventricle_model=None) -> dict:
    model.eval()
    inter = torch.zeros(num_classes, device=device)
    denom = torch.zeros(num_classes, device=device)
    for batch in loader:
        image = batch["image"].to(device)
        if not is_tri:
            image = select_ventricle_input(image)
        elif ventricle_model is not None:
            image = apply_cascade_ventricle_channel(image, ventricle_model)
        target = batch["target"].to(device)
        out = model(image)
        logits = unpack_output(out)[0] if is_tri else out
        if not is_tri:
            prob = torch.sigmoid(logits.squeeze(1))
            pred_oh = torch.stack([1 - prob, prob], dim=1)
            target_oh = F.one_hot((target == 1).long(), 2).permute(0, 3, 1, 2).float()
        else:
            pred_oh = torch.softmax(logits, 1)
            target_oh = F.one_hot(target, num_classes).permute(0, 3, 1, 2).float()
        inter += (pred_oh * target_oh).sum((0, 2, 3))
        denom += pred_oh.sum((0, 2, 3)) + target_oh.sum((0, 2, 3))
    dice = (2 * inter + 1e-6) / (denom + 1e-6)
    return {"per_class": dice.cpu().numpy(), "mean_fg": dice[1:].mean().item()}


def run(args):
    seed_everything(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_status, gpu_msg = check_gpu_compatibility()
    print(gpu_msg)
    if gpu_status == "incompatible":
        raise RuntimeError("CUDA GPU is visible but incompatible with this PyTorch build - fix the "
                            "accelerator setting (see message above) before training, otherwise every "
                            "GPU-hour spent here is wasted.")
    print("Device:", device)

    manifest = build_manifest(args.data_root)
    print("Dataset root:", manifest.iloc[0].flair.rsplit("MS_100_patient_preprocessed", 1)[0] or ".")
    split_df = load_or_build_split(manifest, args.split_csv, args.output_dir)

    train_ids = split_df[(split_df.fold == args.fold) & (split_df.role == "train")].patient_id.tolist()
    val_ids = split_df[(split_df.fold == args.fold) & (split_df.role == "val")].patient_id.tolist()
    print(f"Fold {args.fold}: {len(train_ids)} train / {len(val_ids)} val patients")

    is_tri = args.stage == "tri"
    use_vent_channel = args.use_ventricle_channel if is_tri else False
    ds_cfg_train = DatasetConfig(use_ventricle_channel=use_vent_channel, augment=True, include_empty_slices=True)
    ds_cfg_val = DatasetConfig(use_ventricle_channel=use_vent_channel, augment=False, include_empty_slices=True)

    train_ds = MS3SEGDataset(manifest, train_ids, ds_cfg_train)
    val_ds = MS3SEGDataset(manifest, val_ids, ds_cfg_val)

    ventricle_model = None
    if is_tri and args.cascade:
        if args.ventricle_checkpoint is None:
            raise ValueError("--cascade requires --ventricle-checkpoint")
        ventricle_model = VentricleNet(base=args.base_channels // 4 if args.base_channels >= 16 else 8).to(device)
        load_checkpoint(args.ventricle_checkpoint, ventricle_model)
        ventricle_model.eval()
        for p in ventricle_model.parameters():
            p.requires_grad_(False)
    # Note: train_ds/val_ds still request use_ventricle_channel=True (a
    # ground-truth-based placeholder in the last channel) when --cascade is
    # set; apply_cascade_ventricle_channel() overwrites that placeholder with
    # the Stage-1 model's own prediction, batched, after collation - see its
    # docstring for why this must not happen inside Dataset.__getitem__.

    sampler_gen = torch.Generator().manual_seed(SEED)
    sampler = WeightedRandomSampler(
        torch.as_tensor(train_ds.sample_weights, dtype=torch.double),
        num_samples=len(train_ds), replacement=True, generator=sampler_gen,
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                               num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    if is_tri:
        in_channels = 10 if use_vent_channel else 9
        model = build_model(args.model, in_channels=in_channels, num_classes=4,
                             base_channels=args.base_channels,
                             deep_supervision=args.deep_supervision).to(device)
        class_weights = compute_class_weights(manifest, train_ids).to(device)
        criterion = DiceCELoss(class_weights, lesion_tversky_weight=args.lesion_tversky_weight)
    else:
        model = VentricleNet(base=args.base_channels // 4 if args.base_channels >= 16 else 8).to(device)
        criterion = None

    print(f"Model: {args.model if is_tri else 'ventricle_net'} | in_channels={in_channels if is_tri else 1}")
    print("Trainable parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if is_tri:
        # Include whether the anatomy channel was used in the checkpoint name.
        # Without this, training the same --model both with and without the
        # channel into the same --output-dir (e.g. via --also-run-baseline)
        # silently overwrites the first variant's checkpoint with the second's.
        tag = f"{args.model}_anatomy" if use_vent_channel else f"{args.model}_baseline"
    else:
        tag = "ventricle_net"
    best_ckpt = args.output_dir / f"{tag}_{args.stage}_best.pt"
    last_ckpt = args.output_dir / f"{tag}_{args.stage}_last.pt"

    start_epoch = 1
    best_score = -1.0
    if args.resume and last_ckpt.exists():
        ckpt = load_checkpoint(last_ckpt, model, optimizer, scheduler)
        start_epoch = ckpt["epoch"] + 1
        best_score = ckpt["best_score"]
        print(f"Resumed from epoch {ckpt['epoch']}, best_score={best_score:.4f}")

    history = []
    epochs_without_improvement = 0
    try:
        for epoch in range(start_epoch, args.epochs + 1):
            model.train()
            t0 = time.time()
            running_loss = 0.0
            for batch in train_loader:
                image = batch["image"].to(device, non_blocking=True)
                if not is_tri:
                    image = select_ventricle_input(image)
                elif ventricle_model is not None:
                    image = apply_cascade_ventricle_channel(image, ventricle_model)
                target = batch["target"].to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                    if is_tri:
                        out = model(image)
                        main_logits, aux_logits = unpack_output(out)
                        loss = criterion.with_deep_supervision(main_logits, aux_logits, target) \
                            if (args.deep_supervision and aux_logits) else criterion(main_logits, target)
                    else:
                        logits = model(image)
                        loss = binary_dice_ce_loss(logits, (target == 1).long())
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                running_loss += loss.item()

            val_metrics = slice_level_val_dice(model, val_loader, device,
                                                num_classes=4 if is_tri else 2, is_tri=is_tri,
                                                ventricle_model=ventricle_model)
            score = val_metrics["mean_fg"]
            scheduler.step(score)
            train_loss = running_loss / max(1, len(train_loader))
            dt = time.time() - t0
            lr_now = optimizer.param_groups[0]["lr"]
            print(f"E{epoch:03d} loss={train_loss:.4f} val_mean_fg_dice={score:.4f} "
                  f"per_class={np.round(val_metrics['per_class'], 4)} lr={lr_now:.2e} time={dt:.0f}s")
            history.append({"epoch": epoch, "train_loss": train_loss, "val_mean_fg_dice": score, "lr": lr_now})

            save_checkpoint(last_ckpt, model, optimizer, scheduler, epoch, best_score)
            if score > best_score:
                best_score = score
                epochs_without_improvement = 0
                save_checkpoint(best_ckpt, model, optimizer, scheduler, epoch, best_score)
                print(f"  -> new best ({best_score:.4f}), saved {best_ckpt}")
            else:
                epochs_without_improvement += 1
                if args.early_stopping_patience > 0 and epochs_without_improvement >= args.early_stopping_patience:
                    print(f"Early stopping: no improvement for {epochs_without_improvement} epochs "
                          f"(patience={args.early_stopping_patience}). Best stays at {best_score:.4f}.")
                    break
    except KeyboardInterrupt:
        print("Interrupted - saving emergency checkpoint.")
        save_checkpoint(args.output_dir / f"{tag}_{args.stage}_interrupted.pt", model, optimizer, scheduler, epoch, best_score)
        raise

    import pandas as pd
    pd.DataFrame(history).to_csv(args.output_dir / f"{tag}_{args.stage}_history_fold{args.fold}.csv", index=False)
    print("Done. Best val mean foreground Dice:", best_score)
    return {"best_ckpt": best_ckpt, "best_score": best_score, "history": history}


def main():
    run(parse_args())


if __name__ == "__main__":
    main()
