# Single-Cell Foundation Models

Foundation models for single-cell microscopy images using 2D/3D Masked Autoencoders (MAE).

**Datasets:** OpenCell (current), WTC-11 (current), HPA, JUMP (future)
**Base Code:** Adapted from [SelfMedMAE](https://github.com/cvlab-stonybrook/SelfMedMAE/tree/main)

---

## Installation

### 1. Create Environment

```bash
conda create -n sc_project python=3.11 -y
conda activate sc_project

# Optional: for notebook development
conda install -c conda-forge jupyterlab -y

# Install PyTorch with CUDA
conda install -y -c pytorch -c nvidia -c conda-forge \
  pytorch=2.1.* \
  torchvision=0.16.* \
  pytorch-cuda=11.8

# Install dependencies
pip install -r requirements.txt
```

### 2. Verify Installation

```bash
# Test that normalization is working correctly
python test/test_normalization.py --config configs/opencell/opencell_3d.yaml
```

---

## Project Structure

```
src/
├── data/
│   ├── opencell/           # OpenCell dataset & transforms
│   │   ├── dataset.py              # Single-cell dataset
│   │   ├── localization_dataset.py # Classification dataset
│   │   └── transforms.py           # Channel-wise normalization
│   └── base/               # Base classes for future datasets
│
├── lib/
│   ├── models/             # MAE2D, MAE3D, ViT classifiers
│   ├── networks/           # ViT encoder/decoder components
│   ├── trainers/           # Training logic
│   └── utils/
│
├── evaluation/
│   └── opencell/           # OpenCell-specific metrics
│
└── scripts/                # Training/evaluation scripts

configs/opencell/           # Configuration files
├── opencell_2d.yaml                # MAE 2D pretraining (single-cell)
├── opencell_3d.yaml                # MAE 3D pretraining (single-cell)
├── opencell_localization_2d.yaml   # Fine-tuning
└── opencell_localization_3d.yaml   # Fine-tuning

test/                       # Test scripts
└── test_normalization.py   # Test data pipeline and normalization
```

---

## Quick Start

### OpenCell Dataset

**Location:** `/path/to/datasets/opencell`
**Source:** [OpenCell Download](https://opencell.sf.czbiohub.org/download)
**Channels:** 2 (nucleus, protein)
**Format:** 3D TIFF files (Z, C, Y, X)
**Image dimensions:** Pre-cropped single cells (100, 2, 176, 176) - Z, C, H, W

### Training

#### 1. MAE Pretraining (3D)

```bash
python src/train_with_trainer.py --config configs/opencell/opencell_3d.yaml
```

**Key features:**
- 3D Masked Autoencoder
- 75% masking ratio
- Channel-wise normalization (critical for multi-channel data)
- Intensity augmentation

#### 2. MAE Pretraining (2D)

```bash
python src/train_with_trainer.py --config configs/opencell/opencell_2d.yaml
```

Uses max-projected images (Z-axis) for faster training.

#### 3. Fine-tuning: Protein Localization

```bash
python src/train_with_trainer.py --config configs/opencell/opencell_localization_3d.yaml
```

**Task:** Predict protein subcellular localization (17 classes)
**Model:** ViT classifier with pretrained MAE encoder
**Strategy:** Linear probing or full fine-tuning

### Evaluation

```bash
python src/evaluate_localization.py \
    --config configs/opencell/opencell_localization_3d.yaml \
    --checkpoint /path/to/checkpoint.pth.tar \
    --output results.json
```

**Metrics:** mAP, AUC, F1 (macro/micro), per-class metrics

---

## WTC-11 Dataset

### Dataset Overview

**Proteins:** 25 proteins (including AAVS1 control)
**Channels:** 2 (nucleus, protein)
**Localization categories:** 15 classes
**Format:** K-fold cross-validation (typically 5-fold)

**Localization categories:**
cell_contact, centrosome, chromatin, cytoplasmic, cytoskeleton, er, focal_adhesions, golgi, mitochondria, nuclear_membrane, nuclear_punctae, nucleolus_fc_dfc, nucleolus_gc, peroxisome, vesicles

### Training Pipeline

#### Step 1: Extract ESM2 Embeddings

```bash
# Extract ESM2 embeddings for 25 WTC proteins
sbatch scripts/wtc/extract_wtc_esm2_embeddings.sbatch
```

**Output:**
```
wtc11/esm2_embeddings/
├── embeddings.npy        # (25, 1280) - one vector per protein
├── protein_names.txt     # 25 gene symbols
└── wtc_sequences.json    # UniProt cache
```

**Note:** AAVS1 is a control locus with no protein sequence → zero embedding

#### Step 2: Create Fold-Specific ESM2 Embeddings

```bash
python src/wtc/create_wtc_kfold_esm2_embeddings.py \
    --protein_emb_file   /path/to/wtc11/esm2_embeddings/embeddings.npy \
    --protein_names_file /path/to/wtc11/esm2_embeddings/protein_names.txt \
    --kfold_dir          /path/to/wtc11/kfold5 \
    --output_dir         /path/to/wtc11/esm2_embeddings_kfold5
```

#### Step 3: Train MAE with Protein Conditioning

**FFT models (frequency-domain conditioning):**
```bash
sbatch --array=0-4 scripts/wtc/train_mae3d_cross_attention_fft_kfold_1gpu.sbatch
sbatch --array=0-4 scripts/wtc/train_mae2d_cross_attention_fft_kfold_1gpu.sbatch
```

**CLIP models (contrastive learning on top of FFT):**
```bash
sbatch --array=0-4 scripts/wtc/train_mae3d_cross_attention_clip_kfold_1gpu.sbatch
sbatch --array=0-4 scripts/wtc/train_mae2d_cross_attention_clip_kfold_1gpu.sbatch
```

### Localization Task (Downstream)

#### Extract MAE Embeddings

```bash
# For 3D FFT model (example)
sbatch --array=0-4 scripts/wtc/extract_wtc_mae3d_fft_kfold_embeddings.sbatch

# For other variants: 2d_fft, 3d_clip, 2d_clip
```

#### Train Linear Probe

```bash
# For 3D FFT model (example)
sbatch --array=0-4 scripts/wtc/train_localization_emb_3d_fft_kfold.sbatch

# Or manually:
python src/train_localization_wtc.py \
    --config configs/wtc/wtc_localization_emb_3d_fft_kfold.yaml
```

#### Evaluate

```bash
# For 3D FFT model (example)
sbatch --array=0-4 scripts/wtc/evaluate_localization_emb_3d_fft_kfold.sbatch

# Or manually:
python src/evaluate_localization_wtc.py \
    --config configs/wtc/wtc_localization_emb_3d_fft_kfold.yaml \
    --checkpoint /path/to/checkpoint.pth.tar
```

**Available model variants:**
- `3d_fft`: 3D MAE with FFT conditioning
- `2d_fft`: 2D MAE with FFT conditioning
- `3d_clip`: 3D MAE with CLIP contrastive learning
- `2d_clip`: 2D MAE with CLIP contrastive learning

---

## Key Features

### Channel-Wise Normalization

**Critical for multi-channel microscopy data:**
- Each channel (nucleus, protein) normalized independently
- Ensures model learns biological features, not brightness differences
- Essential for datasets with 4-5 channels (HPA, JUMP)

**Configured in configs:**
```yaml
channel_wise_norm: true          # Enable channel-wise normalization
intensity_augmentation: true     # Random intensity scaling/shifting
```

### Data Augmentation

**Spatial:**
- Random flipping (X, Y, Z axes)
- Random 90° rotations

**Intensity:**
- Random scaling (±10%)
- Random shifting (±10%)
- Robust to imaging variations

---

## Configuration

All experiments configured via YAML files in `configs/opencell/`.

**Example:** `configs/opencell/opencell_3d.yaml`

```yaml
# Model
arch: vit_base
in_chans: 2
input_size: [100, 176, 176]  # D, H, W
patch_size: [10, 8, 8]
mask_ratio: 0.75

# Training
batch_size: 2
epochs: 100
lr: 1.5e-4

# Normalization (CRITICAL)
channel_wise_norm: true
intensity_augmentation: true
```

**Override via command line:**
```bash
python src/train_with_trainer.py \
    --config configs/opencell/opencell_3d.yaml \
    --lr 1e-4 \
    --batch_size 4 \
    --epochs 50
```

---

## Common Tasks

### Test Normalization

```bash
# 3D
python test/test_normalization.py --config configs/opencell/opencell_3d.yaml

# 2D
python test/test_normalization.py --config configs/opencell/opencell_2d.yaml --use_2d
```

### Resume Training

```yaml
# In config file
resume: /path/to/checkpoint.pth.tar
```

### Multi-GPU Training

```bash
# Using torchrun (recommended)
torchrun --nproc_per_node=4 src/train_with_trainer.py \
    --config configs/opencell/opencell_3d.yaml
```

---

## Checkpoints & Outputs

**Default output directory:**
```
/path/to/datasets/opencell/
├── mae_opencell_3d/              # MAE 3D
│   └── {run_name}/
│       ├── ckpts/                # Checkpoints
│       └── logs/                 # Logs
├── mae_opencell_2d/              # MAE 2D
│   └── {run_name}/
│       ├── ckpts/
│       └── logs/
└── localization_results/
    └── {run_name}/
        ├── ckpts/
        └── results.json
```

**Checkpoint format:**
```python
{
    'epoch': epoch,
    'state_dict': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'args': args,
}
```

---

## WandB Logging

All experiments logged to Weights & Biases:

```yaml
# In config
proj_name: mae3d
run_name: mae3d_vit_base_opencell
```

**Logged metrics:**
- Training/validation loss
- Learning rate
- Reconstruction visualizations (MAE)
- Classification metrics (localization)
- GPU utilization

---

## Troubleshooting

### Import Errors

**Error:** `ModuleNotFoundError: No module named 'data.opencell_dataset'`

**Fix:** Update imports to new structure:
```python
# OLD
from data.opencell_dataset import OpenCellDataset

# NEW
from data.opencell.dataset import OpenCellDataset
```

### Config Not Found

**Error:** `FileNotFoundError: configs/opencell_3d.yaml`

**Fix:** Configs now in `configs/opencell/`:
```bash
# OLD
--config configs/opencell_3d.yaml

# NEW
--config configs/opencell/opencell_3d.yaml
```

### CUDA Out of Memory

**Solutions:**
- Reduce `batch_size` in config
- Use gradient accumulation
- Use 2D instead of 3D
- Enable mixed precision training (already enabled)

---

## Next Steps

### Completed
- [x] MAE 3D pretraining on OpenCell
- [x] MAE 2D pretraining on OpenCell
- [x] Protein localization classification
- [x] Channel-wise normalization
- [x] Organized project structure

### Planned
- [ ] Add HPA dataset (4 channels)
- [ ] Add JUMP dataset (5+ channels)
- [ ] Cross-dataset transfer learning

---

## Citation

**Base Code:**
```
SelfMedMAE: Self-supervised Masked Autoencoder for Medical Image Analysis
GitHub: https://github.com/cvlab-stonybrook/SelfMedMAE
```

**OpenCell Dataset:**
```
OpenCell: Proteome-scale endogenous tagging enables the cartography of human cellular organization
Website: https://opencell.czbiohub.org
```

---

## Documentation

- **Main Guide:** This README (you're reading it!)
- **Test Installation:** `python test/test_normalization.py --config configs/opencell/opencell_3d.yaml`

## Support

**Need Help?**
- Check troubleshooting section above
- Test your setup with normalization test script