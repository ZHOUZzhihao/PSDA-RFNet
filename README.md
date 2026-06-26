# Dual-Branch Diagnosis Network

Dual-branch spectral-directional attention network with reliability-guided fusion for cross-gear few-shot fault diagnosis.

The method organizes order spectra into five transmission-related physical bands,
then learns complementary representations from spectral structures and
directional vibration responses. A reliability-guided fusion module combines the
two prototype branches and applies weak gear-condition calibration for
cross-tooth-profile target adaptation.

**Pre-processed data version** — no raw CSV or signal processing required. All
training data is pre-computed as .npz files under `dataset/`.

## Quick test

```python
import json, torch
from pathlib import Path
from npz_dataset import NpzDataset
from models.network import DiagnosisNet

# Load pre-processed data
src = NpzDataset("dataset/source.npz")
print(f"Source: {len(src)} samples")

# Instantiate model
cfg = json.loads(Path("configs/config.json").read_text())
model = DiagnosisNet(cfg, "full")
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")  # 330,886

# Dummy forward pass
B = 4
support = {
    "band_spectra": torch.randn(B, 3, 5, 64),
    "band_coords": torch.zeros(5, 64),
    "label": torch.randint(0, 5, (B,)),
    "full_spectrum": torch.randn(B, 3, 2049),
    "c_gear": torch.zeros(B),
}
query = dict(support, label=torch.randint(0, 5, (B,)))
out = model.classify_episode(support, query)
print(f"fused_logits: {out['fused_logits'].shape}")  # [4, 5]
```

## Requirements

- Python 3.9+
- PyTorch 2.0+
- NumPy

```bash
pip install torch numpy
```

## Directory structure

```
├── dataset/
│   ├── source.npz             # Source domain (420 samples)
│   ├── support.npz            # Target support set (280 samples)
│   └── query.npz              # Target query set (560 samples)
├── configs/
│   └── config.json            # Model and training configuration
├── models/
│   ├── __init__.py
│   ├── network.py             # Main DiagnosisNet model
│   ├── stem.py                # Multi-scale spectral input stem
│   ├── spec_attn.py           # Spectral attention branch (intra/inter-band)
│   ├── dir_attn.py            # Directional cross-attention branch
│   ├── fusion.py              # Adaptive reliability fusion (with condition calibration)
│   └── fusion_util.py         # Base reliability-guided fusion
├── npz_dataset.py             # .npz data loader
├── scripts/
│   └── run.py                 # Training and evaluation script
└── README.md
```

## Training and evaluation

Protocol Src10→Mixed (source gear type A → target gear type B mixed-torque):

```bash
cd github_up
# Single GPU
PYTHONPATH=. python scripts/run.py --device cuda:0 --K 1 2 3 --seeds 31 41 49 52 58

# Multi-GPU (5 workers)
for i in 0 1 2 3 4; do
  CUDA_VISIBLE_DEVICES=$i PYTHONPATH=. python scripts/run.py \
    --device cuda:0 --worker-id $i --num-workers 5 &
done
```

Results are saved to `results/worker*_results.json`.

## Model variants

| Key | Description |
|-----|-------------|
| `full` | Full model (spectral + directional + adaptive fusion with condition calibration) |
| `base` | Base model (spectral + directional + base reliability fusion, 5-dim) |

## Dataset format

Each .npz file contains pre-processed band spectra and full spectra:

| Key | Shape | Description |
|-----|-------|-------------|
| `band_w0` / `band_w1` | [N, 3, 5, 64] | 5 physical-band spectra (window 0/1) |
| `full_w0` / `full_w1` | [N, 3, 2049] | Full order spectra |
| `labels` | [N] | Fault class (0-4) |
| `recording_ids` | [N] | Recording identifiers |
| `acquisition_ids` | [N] | Acquisition IDs |
| `c_gear` | [N] | Condition flag (gear type) |
| `band_coords` | [5, 64] | Physical band coordinates (shared) |

## Citation

If you use this code, please cite our paper.
