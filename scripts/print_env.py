#!/usr/bin/env python
"""Print the exact environment used for a run.

Run this as the FIRST cell of any notebook (or `!python scripts/print_env.py`) and keep the
output with the saved notebook version. It is the authoritative record of Python / framework
versions and GPU for the paper's reproducibility section.
"""
from __future__ import annotations

import platform
import sys


def _v(name: str):
    try:
        mod = __import__(name)
    except Exception as exc:  # noqa: BLE001
        return f"NOT INSTALLED ({exc.__class__.__name__})"
    return getattr(mod, "__version__", "unknown")


def main() -> None:
    print("=" * 70)
    print("MS3SEG VentiMorph-RelNet V2.7 - environment capture")
    print("=" * 70)
    print(f"timestamp (UTC)     : {__import__('datetime').datetime.utcnow().isoformat()}Z")
    print(f"python              : {sys.version.split()[0]} ({platform.python_implementation()})")
    print(f"platform            : {platform.platform()}")
    print()
    for pkg in ["numpy", "pandas", "scipy", "sklearn", "nibabel", "matplotlib",
                "torch", "torchvision"]:
        print(f"{pkg:<20}: {_v(pkg)}")

    try:
        import torch

        print()
        print(f"torch.version.cuda  : {torch.version.cuda}")
        print(f"cudnn               : {torch.backends.cudnn.version()}")
        print(f"cuda available      : {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"gpu name            : {torch.cuda.get_device_name(0)}")
            print(f"gpu capability      : sm_{''.join(map(str, torch.cuda.get_device_capability(0)))}")
            print(f"arch list           : {sorted(torch.cuda.get_arch_list())}")
            props = torch.cuda.get_device_properties(0)
            print(f"gpu memory (GB)     : {props.total_memory / 1024**3:.1f}")
    except Exception as exc:  # noqa: BLE001
        print(f"torch introspection failed: {exc}")

    print("=" * 70)


if __name__ == "__main__":
    main()
