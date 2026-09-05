"""
Run this FIRST, before spending any GPU time. Builds every registered
model, runs one dummy forward pass through the exact adapter train.py uses,
and reports parameter counts / output shapes / which ones actually import
and run on your installed torch+MONAI. Catches and reports failures
per-model instead of crashing, since MONAI's UNETR/SwinUNETR constructor
kwargs have drifted across versions and this can't be verified without
executing it on your machine.

    python sanity_check.py
    python sanity_check.py --device cuda   # also checks it actually fits
"""

from __future__ import annotations

import argparse
import traceback

import torch

from baselines import MODEL_NAMES, build_model, unpack_output
from model import VentricleNet


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--in-channels", type=int, default=10)
    p.add_argument("--num-classes", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=2)
    args = p.parse_args()

    device = torch.device(args.device)
    dummy = torch.randn(args.batch_size, args.in_channels, 256, 256, device=device)
    print(f"Device: {device} | dummy input: {tuple(dummy.shape)}\n")

    try:
        vent_model = VentricleNet(base=8).to(device)
        n_params = sum(p.numel() for p in vent_model.parameters() if p.requires_grad)
        vent_dummy = torch.randn(args.batch_size, 1, 256, 256, device=device)
        with torch.no_grad():
            vent_out = vent_model(vent_dummy)
        ok_shape = tuple(vent_out.shape) == (args.batch_size, 1, 256, 256)
        print(f"[{'ventricle_net':16s}] params={n_params:>10,} logits={tuple(vent_out.shape)} "
              f"-> {'OK' if ok_shape else f'UNEXPECTED SHAPE {tuple(vent_out.shape)}'}")
        del vent_model, vent_out
        if device.type == "cuda":
            torch.cuda.empty_cache()
    except Exception as e:
        print(f"[{'ventricle_net':16s}] FAILED: {type(e).__name__}: {e}")
        traceback.print_exc(limit=2)

    results = []
    for name in sorted(MODEL_NAMES):
        try:
            model = build_model(name, in_channels=args.in_channels, num_classes=args.num_classes).to(device)
            n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            model.eval()
            with torch.no_grad():
                out = model(dummy)
            main_logits, aux = unpack_output(out)
            ok_shape = tuple(main_logits.shape) == (args.batch_size, args.num_classes, 256, 256)
            status = "OK" if ok_shape else f"UNEXPECTED SHAPE {tuple(main_logits.shape)}"
            print(f"[{name:16s}] params={n_params:>10,} main_logits={tuple(main_logits.shape)} "
                  f"aux_heads={len(aux)} -> {status}")
            results.append((name, True, n_params))
            del model, out
            if device.type == "cuda":
                torch.cuda.empty_cache()
        except Exception as e:
            print(f"[{name:16s}] FAILED: {type(e).__name__}: {e}")
            traceback.print_exc(limit=2)
            print()
            results.append((name, False, None))

    print("\n=== Summary ===")
    for name, ok, n_params in results:
        print(f"  {name:16s} {'OK' if ok else 'FAILED'}"
              + (f"  ({n_params:,} params)" if ok else ""))
    n_failed = sum(1 for _, ok, _ in results if not ok)
    if n_failed:
        print(f"\n{n_failed} model(s) failed to build/run. For MONAI-backed models, check "
              f"`import monai; print(monai.__version__)` and the constructor signature with "
              f"`help(monai.networks.nets.UNETR)` / `help(monai.networks.nets.SwinUNETR)` - "
              f"the failing kwargs in baselines.py may need adjusting for your installed version.")


if __name__ == "__main__":
    main()
