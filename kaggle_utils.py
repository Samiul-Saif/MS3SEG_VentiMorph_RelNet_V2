"""
Small helpers for the Kaggle session-boundary workflow: `/kaggle/working` is
wiped when a session ends, so anything you want to keep (checkpoints,
history CSVs, the final comparison table) has to be zipped and either
downloaded or re-attached as a Kaggle Dataset input to the next session -
exactly the pattern your original notebooks already used
(`MS3SEG_V27_FOLDx_TRANSFER.zip`, "Download this ZIP before leaving
Kaggle"). These two functions automate that instead of doing it by hand.
"""

from __future__ import annotations

import shutil
from pathlib import Path

IS_KAGGLE = Path("/kaggle").exists()


def zip_run_outputs(output_root: Path, zip_path: Path | None = None) -> Path:
    """Zips `output_root` (e.g. runs_pilot/ or runs_full/) so it survives the
    session. On Kaggle, leave the result under /kaggle/working - "Save
    Version" will publish it as a downloadable Dataset automatically; you
    can then "Add Input" that dataset in the next session to resume."""
    output_root = Path(output_root)
    zip_path = Path(zip_path) if zip_path else output_root.with_suffix(".zip")
    base_name = str(zip_path.with_suffix(""))
    archive = shutil.make_archive(base_name, "zip", root_dir=output_root)
    print(f"Zipped {output_root} -> {archive}")
    if IS_KAGGLE:
        print("On Kaggle: click 'Save Version' (Save & Run All) to publish this as a Dataset you "
              "can 'Add Input' to your next session to resume training from where this one left off.")
    return Path(archive)


def restore_run_outputs(dataset_dir: Path, dest_root: Path) -> bool:
    """Looks for a previously-zipped run (from `zip_run_outputs`) inside a
    Kaggle Dataset you've added as an input, and unpacks it into
    `dest_root` so --resume / --cascade checkpoint paths line up exactly as
    they did in the session that produced them. Returns True if it found
    and restored something."""
    dataset_dir = Path(dataset_dir)
    dest_root = Path(dest_root)
    if not dataset_dir.exists():
        print(f"No such input directory: {dataset_dir} (nothing to restore)")
        return False

    zips = list(dataset_dir.rglob("*.zip"))
    if not zips:
        existing_dir = dataset_dir if (dataset_dir / "fold0").exists() or any(dataset_dir.iterdir()) else None
        if existing_dir is not None:
            dest_root.mkdir(parents=True, exist_ok=True)
            shutil.copytree(existing_dir, dest_root, dirs_exist_ok=True)
            print(f"Copied unzipped outputs from {existing_dir} -> {dest_root}")
            return True
        print(f"No .zip and no run outputs found under {dataset_dir}")
        return False

    dest_root.mkdir(parents=True, exist_ok=True)
    for z in zips:
        shutil.unpack_archive(str(z), extract_dir=str(dest_root))
        print(f"Restored {z} -> {dest_root}")
    return True
