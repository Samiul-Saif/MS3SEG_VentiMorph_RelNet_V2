"""
MS3SEG data loading.

Directory layout expected under --data-root (same as the original MS3SEG
Kaggle dataset the previous notebooks used):

    <data-root>/MS_100_patient_preprocessed/<pid>/<pid>_FLAIR.nii[.gz]
    <data-root>/MS_100_patient_preprocessed/<pid>/<pid>_T1WI_reg.nii[.gz]
    <data-root>/MS_100_patient_preprocessed/<pid>/<pid>_T2WI_reg.nii[.gz]
    <data-root>/MS_100_patient_masks/<Vent root>/<pid>/<pid>_Vent_Mask.nii[.gz]
    <data-root>/MS_100_patient_masks/<nWMH root>/<pid>/<pid>_nWMH_Mask.nii[.gz]
    <data-root>/MS_100_patient_masks/<abWMH root>/<pid>/<pid>_abWMH_Mask.nii[.gz]

If the dataset's own official split file (patient_level_5fold_split.csv,
shipped in the MS3SEG GitHub repo / Figshare record, columns
patient_id,fold,role) is placed at --split-csv, it is used directly so
results stay comparable to the numbers in Table 6-9 of the MS3SEG paper.
Otherwise a lesion-burden-stratified 5-fold split is regenerated (same
recipe the original custom-model notebooks used) and written next to the
outputs so every later run reuses exactly the same patients.
"""

from __future__ import annotations

import re
import random
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import nibabel as nib
from scipy import ndimage
from sklearn.model_selection import train_test_split, StratifiedKFold

import torch
from torch.utils.data import Dataset

SEED = 42


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _find_unique(folder: Path, patterns: list[str], desc: str) -> Path:
    found, seen = [], set()
    for pattern in patterns:
        for p in folder.glob(pattern):
            if p.is_file() and p not in seen:
                found.append(p)
                seen.add(p)
    if len(found) != 1:
        raise FileNotFoundError(f"{desc}: expected exactly one match in {folder}, found {found}")
    return found[0]


def discover_data_root(explicit: Path | None = None) -> Path:
    """Resolves the MS3SEG dataset root. If `explicit` is given (e.g. via
    --data-root) it's used as-is. Otherwise this searches /kaggle/input
    (Kaggle notebooks) and the current directory for a folder containing
    MS_100_patient_preprocessed/ + MS_100_patient_masks/, the same
    auto-discovery your original notebooks did for /kaggle/input/datasets/..."""
    if explicit is not None:
        return Path(explicit)

    search_roots = []
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        search_roots.append(kaggle_input)
    search_roots.append(Path.cwd())

    for base in search_roots:
        if not base.exists():
            continue
        for preprocessed_dir in base.rglob("MS_100_patient_preprocessed"):
            candidate = preprocessed_dir.parent
            if (candidate / "MS_100_patient_masks").exists():
                return candidate

    raise FileNotFoundError(
        "Could not auto-discover the MS3SEG dataset root (looked for a folder containing "
        "MS_100_patient_preprocessed/ and MS_100_patient_masks/ under /kaggle/input and the "
        "current directory). On Kaggle: Add Input -> your MS3SEG dataset. Elsewhere: pass "
        "--data-root explicitly."
    )


def check_gpu_compatibility() -> tuple[str, str]:
    """Same check your original notebooks used, adapted for reuse. Returns
    (status, message) where status is one of:
      "ok"           - a CUDA GPU is visible and this PyTorch build has kernels for it.
      "cpu_only"     - no CUDA device at all (e.g. deliberate CPU debugging). Not an error.
      "incompatible" - a CUDA GPU IS visible but this PyTorch build has no compiled kernels
                        for its architecture. This is the dangerous case: forward/backward
                        calls fail (or previously, on an older PyTorch build, silently ran
                        garbage), wasting Kaggle GPU-hour quota for no result. This exact
                        failure previously happened on Kaggle's P100 accelerator option with
                        an sm_60-incompatible PyTorch build.
    Call this first and print the message before starting any run."""
    if not torch.cuda.is_available():
        return "cpu_only", ("No CUDA device visible - running on CPU. If you meant to use a GPU: "
                             "on Kaggle, Notebook Settings -> Accelerator -> GPU T4 x2 (or P100), "
                             "save, and the session restarts with a GPU attached.")
    name = torch.cuda.get_device_name(0)
    capability = torch.cuda.get_device_capability(0)
    required_arch = f"sm_{capability[0]}{capability[1]}"
    compiled_arches = set(torch.cuda.get_arch_list())
    ok = required_arch in compiled_arches
    msg = f"GPU: {name} | capability={required_arch} | PyTorch compiled for: {sorted(compiled_arches)}"
    if not ok:
        msg += ("\nIncompatible: this PyTorch build has no kernels for your GPU architecture. "
                "On Kaggle, switch Notebook Settings -> Accelerator to 'GPU T4 x2', restart the "
                "session, and re-run before training.")
        return "incompatible", msg
    return "ok", msg


def _select_mask_root(mask_roots: list[Path], kind: str) -> Path:
    if kind == "vent":
        matches = [p for p in mask_roots if "vent" in p.name.lower()]
    elif kind == "nwmh":
        matches = [
            p for p in mask_roots
            if ("nwmh" in p.name.lower() or "normal" in p.name.lower())
            and "abwmh" not in p.name.lower() and "abnormal" not in p.name.lower()
        ]
    else:
        matches = [p for p in mask_roots if "abwmh" in p.name.lower() or "abnormal" in p.name.lower()]
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one {kind} mask root, found {matches}")
    return matches[0]


def build_manifest(data_root: Path | None = None) -> pd.DataFrame:
    data_root = discover_data_root(data_root)
    preprocessed_dir = data_root / "MS_100_patient_preprocessed"
    masks_dir = data_root / "MS_100_patient_masks"
    mask_roots = [p for p in masks_dir.iterdir() if p.is_dir()]
    vent_root = _select_mask_root(mask_roots, "vent")
    nwmh_root = _select_mask_root(mask_roots, "nwmh")
    abwmh_root = _select_mask_root(mask_roots, "abwmh")

    patient_dirs = sorted(p for p in preprocessed_dir.iterdir() if p.is_dir() and re.fullmatch(r"\d{3}", p.name))
    records = []
    for pdir in patient_dirs:
        pid = pdir.name
        records.append({
            "patient_id": pid,
            "flair": str(_find_unique(pdir, [f"{pid}_FLAIR.nii", f"{pid}_FLAIR.nii.gz"], f"{pid} FLAIR")),
            "t1": str(_find_unique(pdir, [f"{pid}_T1WI_reg.nii", f"{pid}_T1WI_reg.nii.gz", "*T1*reg*.nii*"], f"{pid} T1")),
            "t2": str(_find_unique(pdir, [f"{pid}_T2WI_reg.nii", f"{pid}_T2WI_reg.nii.gz", "*T2*reg*.nii*"], f"{pid} T2")),
            "vent": str(_find_unique(vent_root / pid, [f"{pid}_Vent_Mask.nii", f"{pid}_Vent_Mask.nii.gz", "*Vent*Mask*.nii*"], f"{pid} vent")),
            "nwmh": str(_find_unique(nwmh_root / pid, [f"{pid}_nWMH_Mask.nii", f"{pid}_nWMH_Mask.nii.gz", "*nWMH*Mask*.nii*"], f"{pid} nWMH")),
            "abwmh": str(_find_unique(abwmh_root / pid, [f"{pid}_abWMH_Mask.nii", f"{pid}_abWMH_Mask.nii.gz", "*abWMH*Mask*.nii*"], f"{pid} abWMH")),
        })
    manifest = pd.DataFrame(records).sort_values("patient_id").reset_index(drop=True)
    if len(manifest) != 100:
        raise RuntimeError(f"Expected 100 patients, found {len(manifest)}")
    return manifest


def load_or_build_split(manifest: pd.DataFrame, split_csv: Path | None, output_dir: Path) -> pd.DataFrame:
    if split_csv is not None and Path(split_csv).exists():
        split_df = pd.read_csv(split_csv, dtype={"patient_id": str})
        split_df["patient_id"] = split_df["patient_id"].str.zfill(3)
        print(f"Using official split file: {split_csv}")
        return split_df

    print("No official split file found -> regenerating a lesion-burden-stratified 5-fold split "
          "(recorded to disk so every later run is reproducible).")
    rows = []
    for _, row in manifest.iterrows():
        abwmh = np.asarray(nib.load(row.abwmh, mmap=True).dataobj) > 0
        rows.append({"patient_id": row.patient_id, "abwmh_voxels": int(abwmh.sum())})
    stats = pd.DataFrame(rows)
    stats["burden_bin"] = pd.qcut(stats.abwmh_voxels.rank(method="first"), q=4, labels=False)

    trainval_ids, test_ids = train_test_split(
        stats.patient_id, test_size=0.20, random_state=SEED, stratify=stats.burden_bin,
    )
    trainval = stats[stats.patient_id.isin(trainval_ids)].reset_index(drop=True)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    split_rows = []
    for fold, (tr, va) in enumerate(skf.split(trainval.patient_id, trainval.burden_bin)):
        split_rows += [{"patient_id": trainval.loc[i, "patient_id"], "fold": fold, "role": "train"} for i in tr]
        split_rows += [{"patient_id": trainval.loc[i, "patient_id"], "fold": fold, "role": "val"} for i in va]
    split_rows += [{"patient_id": pid, "fold": -1, "role": "test"} for pid in sorted(test_ids)]

    split_df = pd.DataFrame(split_rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    split_df.to_csv(output_dir / "patient_level_5fold_split.csv", index=False)
    return split_df


@lru_cache(maxsize=48)
def _load_volume(path: str) -> np.ndarray:
    return np.asarray(nib.load(path, mmap=True).dataobj, dtype=np.float32)


@lru_cache(maxsize=36)
def load_normalized_volume(path: str) -> np.ndarray:
    arr = _load_volume(path).copy()
    finite = np.isfinite(arr)
    nz = finite & (np.abs(arr) > 1e-6)
    if nz.sum() < 100:
        return np.nan_to_num(arr).astype(np.float32)
    low, high = np.percentile(arr[nz], [0.5, 99.5])
    high = max(high, low + 1.0)
    arr = np.clip(arr, low, high)
    arr = (arr - arr[nz].mean()) / (arr[nz].std() + 1e-6)
    arr[~finite] = 0.0
    return arr.astype(np.float32)


@lru_cache(maxsize=48)
def load_binary_mask(path: str) -> np.ndarray:
    return (_load_volume(path) > 0).astype(np.uint8)


def compose_target(row) -> np.ndarray:
    vent = load_binary_mask(row.vent)
    nwmh = load_binary_mask(row.nwmh)
    abwmh = load_binary_mask(row.abwmh)
    target = np.zeros(vent.shape, dtype=np.uint8)
    target[vent > 0] = 1
    target[nwmh > 0] = 2
    target[abwmh > 0] = 3
    return target


@lru_cache(maxsize=64)
def ventricle_distance_volume(vent_path: str) -> np.ndarray:
    """Signed-style ventricle proximity map in [0, 1], 1 = at the ventricle wall,
    decaying to 0 over ~15mm. Computed slice-wise from a binary ventricle mask
    (ground truth during training/validation on this research dataset, or a
    Stage-1 model's prediction at deployment time -- see model.py / train.py)."""
    img = nib.load(vent_path, mmap=True)
    vent = np.asarray(img.dataobj) > 0.5
    spacing = img.header.get_zooms()[:2]
    out = np.zeros(vent.shape, dtype=np.float32)
    for z in range(vent.shape[2]):
        dist_mm = ndimage.distance_transform_edt(~vent[:, :, z], sampling=spacing)
        out[:, :, z] = np.exp(-dist_mm / 15.0).astype(np.float32)
        out[vent[:, :, z], z] = 1.0
    return out


def ventricle_distance_from_prob(vent_prob_2d: np.ndarray, spacing_xy: tuple[float, float]) -> np.ndarray:
    """Same proximity encoding as `ventricle_distance_volume`, but built at inference time
    from a Stage-1 model's predicted ventricle probability map for a single 2D slice."""
    vent_mask = vent_prob_2d > 0.5
    dist_mm = ndimage.distance_transform_edt(~vent_mask, sampling=spacing_xy)
    out = np.exp(-dist_mm / 15.0).astype(np.float32)
    out[vent_mask] = 1.0
    return out


@dataclass
class DatasetConfig:
    use_ventricle_channel: bool = True
    augment: bool = False
    include_empty_slices: bool = True


class MS3SEGDataset(Dataset):
    """2.5D (3-slice triplet) multimodal dataset. Returns a 9-channel tensor
    (FLAIR/T1/T2, each as a 3-slice stack) plus, optionally, a 10th channel
    holding the ground-truth ventricle proximity map (see DatasetConfig)."""

    def __init__(self, manifest: pd.DataFrame, patient_ids: list[str], cfg: DatasetConfig):
        self.manifest = manifest.set_index("patient_id")
        self.patient_ids = [str(p).zfill(3) for p in patient_ids]
        self.cfg = cfg

        self.samples: list[tuple[str, int]] = []
        self.sample_weights: list[float] = []
        for pid in self.patient_ids:
            row = self.manifest.loc[pid]
            target = compose_target(row)
            for z in range(target.shape[2]):
                sl = target[:, :, z]
                vent_px = int((sl == 1).sum())
                wmh_px = int((sl == 2).sum() + (sl == 3).sum())
                if not cfg.include_empty_slices and (vent_px + wmh_px) == 0:
                    continue
                if 0 < wmh_px <= 64:
                    weight = 2.5
                elif wmh_px > 0:
                    weight = 2.0
                elif vent_px > 0:
                    weight = 1.25
                else:
                    weight = 1.0
                self.samples.append((pid, z))
                self.sample_weights.append(weight)

    def __len__(self) -> int:
        return len(self.samples)

    @staticmethod
    def _triplet(volume: np.ndarray, z: int) -> np.ndarray:
        depth = volume.shape[2]
        idx = [max(0, z - 1), z, min(depth - 1, z + 1)]
        return np.stack([volume[:, :, i] for i in idx], axis=0)

    @staticmethod
    def _single_slice(volume: np.ndarray, z: int) -> np.ndarray:
        """Like `_triplet` but for the single-channel ventricle-proximity map:
        one slice, not a 3-slice stack, so it contributes exactly 1 channel
        (9 intensity channels + 1 anatomy channel = 10, matching in_channels=10
        everywhere else). Using `_triplet` here was a bug - it silently made
        this a 12-channel input instead."""
        return volume[:, :, z][None, :, :]

    def _augment(self, image: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        import torchvision.transforms as T
        import torchvision.transforms.functional as TF
        from torchvision.transforms import InterpolationMode

        if random.random() < 0.5:
            image = torch.flip(image, dims=[2])
            target = torch.flip(target, dims=[1])
        if random.random() < 0.6:
            angle, translate, scale, shear = T.RandomAffine.get_params(
                degrees=(-8.0, 8.0), translate=(0.02, 0.02), scale_ranges=(0.97, 1.03),
                shears=None, img_size=[image.shape[1], image.shape[2]],
            )
            image = TF.affine(image, angle, translate, scale, shear,
                               interpolation=InterpolationMode.BILINEAR, fill=0.0)
            target = TF.affine(target.unsqueeze(0).float(), angle, translate, scale, shear,
                                interpolation=InterpolationMode.NEAREST, fill=0.0).squeeze(0).long()
        n_intensity_channels = image.shape[0] - (1 if self.cfg.use_ventricle_channel else 0)
        for start in range(0, n_intensity_channels, 3):
            block = image[start:start + 3]
            block = block * random.uniform(0.95, 1.05) + random.uniform(-0.05, 0.05)
            if random.random() < 0.15:
                block = block + torch.randn_like(block) * 0.015
            image[start:start + 3] = block
        return image, target

    def __getitem__(self, index: int) -> dict:
        pid, z = self.samples[index]
        row = self.manifest.loc[pid]
        flair = load_normalized_volume(row.flair)
        t1 = load_normalized_volume(row.t1)
        t2 = load_normalized_volume(row.t2)

        channels = [self._triplet(flair, z), self._triplet(t1, z), self._triplet(t2, z)]
        if self.cfg.use_ventricle_channel:
            dist = ventricle_distance_volume(row.vent)
            channels.append(self._single_slice(dist, z))
        image = np.concatenate(channels, axis=0)
        expected_channels = 10 if self.cfg.use_ventricle_channel else 9
        assert image.shape[0] == expected_channels, (
            f"Dataset produced {image.shape[0]} channels, expected {expected_channels} "
            f"(use_ventricle_channel={self.cfg.use_ventricle_channel}). This must match "
            f"in_channels passed to build_model() or the first conv layer will reject the batch."
        )
        target = compose_target(row)[:, :, z]

        image_t = torch.from_numpy(image.copy()).float()
        target_t = torch.from_numpy(target.copy()).long()
        if self.cfg.augment:
            image_t, target_t = self._augment(image_t, target_t)
        return {"image": image_t, "target": target_t, "patient_id": pid, "slice_index": z}


def compute_class_weights(manifest: pd.DataFrame, patient_ids: list[str]) -> torch.Tensor:
    table = manifest.set_index("patient_id")
    counts = np.zeros(4, dtype=np.int64)
    for pid in patient_ids:
        counts += np.bincount(compose_target(table.loc[str(pid).zfill(3)]).ravel(), minlength=4)
    freq = counts / counts.sum()
    weights = 1.0 / np.sqrt(freq + 1e-12)
    weights = weights / weights.mean()
    print("Class counts:", counts, "Class weights:", weights)
    return torch.tensor(weights, dtype=torch.float32)
