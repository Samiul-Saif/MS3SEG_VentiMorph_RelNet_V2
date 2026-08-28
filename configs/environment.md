# Environment (hardware + software)

## Hardware

| Item | Value |
|---|---|
| Compute platform | Kaggle Notebooks (free tier), GPU accelerator |
| GPU | 1 × NVIDIA Tesla T4, 16 GB VRAM, compute capability `sm_75` |
| CPU / RAM | Kaggle default (≈ 4 vCPU / ≈ 29 GB RAM) |
| Disk | Kaggle working dir (`/kaggle/working`), inputs read-only under `/kaggle/input` |

Verified at run time from a training notebook:

```
Selected device: cuda
Compatible GPU: Tesla T4; capability=sm_75;
PyTorch architectures=['sm_100', 'sm_120', 'sm_70', 'sm_75', 'sm_80', 'sm_86', 'sm_90']
```

A prior run on a Tesla **P100** (`sm_60`) failed — that Kaggle PyTorch build had no `sm_60`
kernels. Every notebook now guards training behind a GPU-capability check.

## Software

| Component | Version | How known |
|---|---|---|
| OS | Ubuntu 22.04 | Kaggle GPU image |
| Python | 3.11 | Kaggle GPU image (2025) |
| PyTorch | 2.6.x, CUDA 12.x | `torch.cuda.get_arch_list()` includes `sm_100`/`sm_120` (Blackwell) → 2.6+ |
| torchvision | 0.21.x | pairs with torch 2.6 |
| NumPy | 1.26.x | Kaggle GPU image |
| pandas | 2.2.x | Kaggle GPU image |
| SciPy | 1.15.x | Kaggle GPU image |
| scikit-learn | 1.2.x | Kaggle GPU image |
| nibabel | 5.x | pre-installed on Kaggle |
| matplotlib | 3.9–3.10 | deprecation warning `boxplot(labels=)` → renamed in 3.9 |

> The notebooks do not pin versions. To get an authoritative record for your run, add a first cell
> that runs `scripts/print_env.py` (or `!python /kaggle/working/.../scripts/print_env.py`) and keep
> its output with the notebook version.

## Determinism

- Global seed **42** applied to `random`, `numpy`, `torch`, `torch.cuda`, and `PYTHONHASHSEED`.
- `torch.backends.cudnn.deterministic = False`, `torch.backends.cudnn.benchmark = True`
  — chosen for throughput; runs are reproducible in distribution, not bit-exact.
- `WeightedRandomSampler` uses a `torch.Generator` seeded with 42.
- Resumable checkpoints (`last_checkpoint.pt`) also save and restore CPU/NumPy/CUDA RNG state.
