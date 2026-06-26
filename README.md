# PSDA-RFNet

Physical-band Spectral-Directional Attention and Reliability-guided Fusion Network for cross-gear few-shot fault diagnosis.

## Quick test (no dataset needed)

The model can be loaded and tested with dummy tensors without the dataset:

```python
import json, torch
from pathlib import Path
from models.h3_ratio_dca import H3RatioDCA

cfg = json.loads(Path("configs/h3_r1c.json").read_text())
model = H3RatioDCA(cfg, "rdca_gearnet")
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
print(f"branch_weights: {out['branch_weights']}")     # [4, 2]
```

## Requirements

- Python 3.9+
- PyTorch 2.0+
- NumPy, SciPy (for angle resampling in full pipeline)

```bash
pip install torch numpy scipy
```

## Directory structure

```
├── dataset/                     # <-- place gearbox CSV data here
├── configs/
│   └── h3_r1c.json              # Model and training configuration
├── models/
│   ├── __init__.py
│   ├── h3_ratio_dca.py          # PSDA-RFNet main model
│   ├── pbsa.py                  # Physical-band Spectral Attention
│   ├── dca.py                   # Directional Cross-Attention
│   ├── spectral_stem.py         # Spectral tokenizer
│   ├── gear_r_maf.py            # GearRMAF fusion
│   ├── r_maf.py                 # R-MAF base fusion
│   └── cosine_graph.py          # Cosine graph encoder (for ablations)
├── data/
│   ├── acquisition_dataset_v15.py
│   ├── dataset_index_v15.py
│   ├── csv_reader_v15.py
│   ├── angle_resample_v15.py
│   ├── parse_filename_v15.py
│   ├── order_spectrum_v8.py
│   └── order_spectrum_ratio.py
└── scripts/
    └── run.py                   # Training & evaluation script
```

## Full training (dataset required)

Place the gearbox vibration dataset directly in this directory:

```
github/
├── dataset/                  # <-- put data here
│   ├── 人字齿数据包/
│   │   └── <fault_name>/
│   │       └── *.csv
│   └── 斜齿数据包/
│       └── <fault_name>/
│           └── *.csv
├── models/
├── data/
├── configs/
└── scripts/
```

Then run from the `github/` directory:

```bash
PYTHONPATH=. python scripts/run.py --device cuda:0 --K 1 2 3 --seeds 31 41 49 52 58
```

If the dataset is elsewhere, use `--data-root` or the environment variable:

```bash
export PC_DGMA_DATA_ROOT=/another/path/dataset

# Single GPU
PYTHONPATH=. python scripts/run.py --device cuda:0 --K 1 2 3 --seeds 31 41 49 52 58

# Multi-GPU (5 workers, one per seed)
for i in 0 1 2 3 4; do
  CUDA_VISIBLE_DEVICES=$i PYTHONPATH=. python scripts/run.py \
    --device cuda:0 --worker-id $i --num-workers 5 &
done
```

Results are saved to `results/src10_mixed/worker*_results.json`.

## Model variants

| Key | Description |
|-----|-------------|
| `rdca_gearnet` | PSDA-RFNet (PBSA + DCA + GearRMAF) |
| `h3_ratio_dca_5d` | Ratio-DCA + R-MAF(5d) baseline |

## Citation

If you use this code, please cite our paper.
