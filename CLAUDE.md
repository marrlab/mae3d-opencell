# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a research codebase for training foundation models on single-cell microscopy images using 2D/3D Masked Autoencoders (MAE). The project supports multiple datasets (OpenCell, WTC-11) and includes pretraining, fine-tuning, and downstream tasks (protein localization, abundance prediction, protein-protein interaction).

**Base code adapted from:** [SelfMedMAE](https://github.com/cvlab-stonybrook/SelfMedMAE/tree/main)

## Key Commands

### Training

```bash
# MAE pretraining (3D or 2D)
python src/train_with_trainer.py --config configs/opencell/opencell_3d.yaml
python src/train_with_trainer.py --config configs/opencell/opencell_2d.yaml

# Multi-GPU training (recommended)
torchrun --nproc_per_node=4 src/train_with_trainer.py --config configs/opencell/opencell_3d.yaml

# Fine-tuning (protein localization)
python src/train_with_trainer.py --config configs/opencell/opencell_localization_3d.yaml

# Cross-attention models with protein language embeddings
python src/train_mae3d_cross_attention_opencell.py --config configs/opencell/opencell_3d_cross_attention.yaml
```

### Evaluation

```bash
# Localization task
python src/evaluate_localization.py \
    --config configs/opencell/opencell_localization_3d.yaml \
    --checkpoint /path/to/checkpoint.pth.tar \
    --output results.json

# PPI (Protein-Protein Interaction)
python src/evaluate_ppi.py --config configs/opencell/opencell_ppi_3d.yaml
```

### Embedding Extraction

```bash
# Extract MAE embeddings for downstream tasks
python src/extract_mae3d_embeddings.py --config configs/opencell/opencell_3d.yaml --checkpoint /path/to/checkpoint.pth.tar

# Extract ESM2 protein language embeddings
python src/extract_esm2_embeddings.py --output_dir /path/to/output
```

### Testing

```bash
# Test normalization and data pipeline
python test/test_normalization.py --config configs/opencell/opencell_3d.yaml

# For 2D models
python test/test_normalization.py --config configs/opencell/opencell_2d.yaml --use_2d
```

## Architecture Overview

### Core Components

1. **Trainer System** (Object-oriented pattern in `src/lib/trainers/`)
   - `BaseTrainer`: Abstract base class handling distributed training, checkpointing, optimization
   - Concrete trainers inherit from `BaseTrainer` and implement 4 methods:
     - `build_model()`: Model architecture creation
     - `build_optimizer()`: Optimizer setup
     - `build_dataloader()`: Dataset and dataloader creation
     - `epoch_train()`: Training loop for one epoch
   - See `src/TRAINER_README.md` for detailed explanation

2. **Models** (`src/lib/models/`)
   - `MAE3D` / `MAE2D`: Base masked autoencoders
   - `MAE3DChannelCrossAttention*`: Multi-channel fusion variants with protein language models
     - `FFT`: Frequency-domain conditioning
     - `CLIP`: Contrastive learning with protein sequences
     - `ESM2`: ESM2 protein language model integration
     - `Distill`: Knowledge distillation variants
   - `ViT*Classifier`: Vision transformer classifiers for downstream tasks
   - `PPIMetric*`: Models for protein-protein interaction prediction

3. **Networks** (`src/lib/networks/`)
   - `mae_vit.py`: Vision Transformer encoder/decoder components
   - `patch_embed_layers.py`: 2D/3D patch embedding layers
   - `cross_attention.py`: Channel cross-attention mechanisms for multi-channel fusion
   - `protein_modality/`: ESM2 protein language model integration modules

4. **Datasets** (`src/data/`)
   - `opencell/`: OpenCell dataset implementations
     - `dataset.py`: Single-cell images (2 channels: nucleus, protein)
     - `fov_dataset.py`: Field-of-view images (full 600×600 images)
     - `localization_dataset.py`: Classification dataset for protein localization (17 classes)
     - `ppi_dataset.py`: Protein-protein interaction pairs
     - `abundance_dataset.py`: Protein abundance regression
     - `vlm_dataset.py`: Vision-language model QA dataset
   - `wtc/`: WTC-11 dataset (25 proteins, 15 localization categories)

### Data Processing Critical Details

**Channel-wise normalization is CRITICAL for multi-channel microscopy data:**
- Each channel (nucleus, protein) must be normalized independently
- Without this, the model learns brightness differences rather than biological features
- Always set `channel_wise_norm: true` in configs
- Implemented in `src/data/opencell/transforms.py`

**Input formats:**
- **OpenCell single-cell**: (100, 2, 176, 176) - Z, C, H, W
- **OpenCell FOV**: (51, 2, 600, 600) - full field-of-view
- **WTC-11**: Variable dimensions, 2 channels

### Model Variants Explained

**Standard MAE (MAE3D / MAE2D):**
- Vanilla masked autoencoder
- Masks 75% of image patches
- Reconstructs missing patches from visible ones
- No conditioning on external modalities

**Cross-attention variants (MAE3DChannelCrossAttention*):**
These models fuse multi-channel microscopy with protein sequence information:

- **Base**: Cross-attention between image channels and protein embeddings
- **FFT**: Adds frequency-domain reconstruction loss (helps with fine details)
- **CLIP**: Contrastive learning to align image and protein embeddings (better for retrieval)
- **ESM2**: Uses ESM2 protein language model (1280-dim) for conditioning
- **Distill**: Knowledge distillation from teacher model
- **Z-Distill**: Distills 3D model into 2D slices
- **Concat**: Simple concatenation baseline (vs attention)
- **SupLoss**: Adds supervised loss during pretraining

**Why cross-attention?** Multi-channel microscopy contains both structural (nucleus) and protein-specific signals. Cross-attention allows the model to selectively attend to relevant channels based on protein context.

## Configuration System

All experiments are configured via YAML files in `configs/`. Configs are hierarchical and use OmegaConf syntax:

```yaml
# Key sections in configs:
arch: vit_base                    # Model architecture
trainer_name: MAE3DTrainer        # Which trainer class to use
dataset: opencell                 # Dataset name
data_path: /path/to/data          # Data directory
csv_path: /path/to/metadata       # Metadata CSV location

# Critical data processing settings:
channel_wise_norm: true           # ALWAYS true for multi-channel data
intensity_augmentation: true      # Random intensity scaling/shifting
roi_x: 176                        # Crop size (or full image size)
roi_y: 176
roi_z: 100

# Model settings:
in_chans: 2                       # Number of input channels
patch_size: [10, 8, 8]           # Patch size [Z, Y, X] for 3D
mask_ratio: 0.75                  # Masking ratio for MAE

# Training:
batch_size: 2                     # Per-GPU batch size
epochs: 100
lr: 1.5e-4
```

Override any config value via command line:
```bash
python src/train_with_trainer.py --config configs/opencell/opencell_3d.yaml --lr 1e-4 --batch_size 4
```

## Workflow Patterns

### 1. Standard Pretraining → Fine-tuning

```bash
# Step 1: Pretrain MAE
python src/train_with_trainer.py --config configs/opencell/opencell_3d.yaml

# Step 2: Fine-tune for localization
# Update config to point to pretrained checkpoint
python src/train_with_trainer.py --config configs/opencell/opencell_localization_3d.yaml

# Step 3: Evaluate
python src/evaluate_localization.py --config configs/opencell/opencell_localization_3d.yaml --checkpoint /path/to/best.pth.tar
```

### 2. Cross-attention Models with Protein Language

```bash
# Step 1: Extract ESM2 embeddings (one-time, reusable)
python src/extract_esm2_embeddings.py --output_dir /path/to/esm2_embeddings

# Step 2: Train FFT model (frequency-domain conditioning)
python src/train_mae3d_cross_attention_fft_opencell.py --config configs/opencell/opencell_3d_cross_attention_fft.yaml

# Step 3: Fine-tune with CLIP (contrastive learning on top of FFT)
# Update config: set resume to FFT checkpoint
python src/train_mae3d_cross_attention_clip_opencell.py --config configs/opencell/opencell_3d_cross_attention_clip.yaml
```

### 3. K-fold Cross-validation

For robust evaluation on WTC-11:
```bash
# Step 1: Create k-fold splits
python src/create_kfold_splits.py --kfold_dir /path/to/kfold5 --n_splits 5

# Step 2: Create fold-specific ESM2 embeddings
python src/create_kfold_esm2_embeddings.py --kfold_dir /path/to/kfold5

# Step 3: Train on each fold (array job)
# Typically done with SLURM array jobs, but can run individually:
for fold in {0..4}; do
    python src/train_with_trainer.py --config configs/wtc/wtc_3d_cross_attention_fft_kfold.yaml --fold $fold
done
```

## Important Implementation Details

### Checkpoint Format

Checkpoints saved as `.pth.tar` files contain:
```python
{
    'epoch': epoch,
    'state_dict': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'args': args,  # Full config/args namespace
}
```

When loading checkpoints:
- MAE pretraining saves encoder+decoder
- Fine-tuning typically loads only encoder weights
- Check `resume` field in config to auto-resume training

### Distributed Training

The codebase uses PyTorch DistributedDataParallel (DDP):
- Launch via `torchrun --nproc_per_node=N`
- BaseTrainer handles all DDP setup
- Batch size is per-GPU; effective batch = batch_size × num_gpus
- Learning rate is automatically scaled: `lr_scaled = lr × batch_size × world_size / 256`

### Mixed Precision Training

Automatic Mixed Precision (AMP) is used throughout:
- GradScaler handles loss scaling
- Reduces memory usage by ~30-40%
- Enables larger batch sizes
- Already configured in all trainers

### Logging

WandB (Weights & Biases) is used for experiment tracking:
- Set `proj_name` and `run_name` in config
- Automatic logging of loss, learning rate, GPU stats
- MAE reconstructions visualized every `vis_freq` steps
- To disable: set environment variable `WANDB_MODE=disabled`

## Common Issues and Solutions

### Import Errors After Restructuring

The codebase was reorganized. Old imports may fail:
```python
# OLD (incorrect)
from data.opencell_dataset import OpenCellDataset

# NEW (correct)
from data.opencell.dataset import OpenCellDataset
```

### Config Path Changes

Configs moved to subdirectories:
```bash
# OLD: configs/opencell_3d.yaml
# NEW: configs/opencell/opencell_3d.yaml
```

### CUDA Out of Memory

Solutions in order of preference:
1. Reduce `batch_size` in config
2. Use 2D instead of 3D models
3. Reduce model size (encoder_depth, encoder_embed_dim)
4. Use gradient accumulation: set `gradient_accumulation_steps: 2`
5. Enable gradient checkpointing (not yet implemented, would need to add)

### Data Path Issues

The codebase has hardcoded paths for a specific cluster:
- Default data root: `/ictstr01/groups/labs/marr/qscd01/datasets/SingleCellImagesDataset/`
- Update `data_path`, `csv_path`, and `output_dir` in configs for your environment
- Or override via command line: `--data_path /your/path`

## Dataset-Specific Notes

### OpenCell
- 1,310 proteins with GFP tags
- 2 channels: nucleus (Hoechst), protein (GFP)
- Pre-cropped single cells: (100, 2, 176, 176)
- Full FOV images: (51, 2, 600, 600)
- Localization: 17 classes (cytoplasmic, nuclear, mitochondria, etc.)

### WTC-11
- 25 proteins (including AAVS1 control)
- 15 localization categories
- Smaller dataset → use k-fold cross-validation (typically 5-fold)
- AAVS1 has no protein sequence → zero embedding

## File Naming Conventions

Training scripts follow a naming pattern:
- `train_mae{2d|3d}_{variant}_{dataset}.py`
  - Examples: `train_mae3d_opencell.py`, `train_mae3d_cross_attention_fft_opencell.py`
- Evaluation: `evaluate_{task}_{variant}.py`
  - Examples: `evaluate_localization.py`, `evaluate_ppi.py`
- Extraction: `extract_{what}_{variant}.py`
  - Examples: `extract_mae3d_embeddings.py`, `extract_esm2_embeddings.py`

Trainer classes mirror model names:
- Model: `MAE3DChannelCrossAttentionFFT` → Trainer: `MAE3DCrossAttentionFFTTrainer`

## Python Environment

```bash
conda create -n sc_project python=3.11 -y
conda activate sc_project

# PyTorch with CUDA
conda install -y -c pytorch -c nvidia -c conda-forge \
  pytorch=2.1.* torchvision=0.16.* pytorch-cuda=11.8

# Dependencies
pip install -r requirements.txt
```

Key dependencies:
- PyTorch 2.1.x with CUDA 11.8
- MONAI 1.3.x (medical imaging transforms)
- timm 0.4.12 (vision models)
- transformers 4.44.2 (for ESM2, VLM)
- wandb 0.23.1 (experiment tracking)

## Vision-Language Model (VLM) Pipeline

The codebase includes experimental VLM support for training models that answer questions about cell images:

```bash
# 1. Create VLM dataset (generates QA pairs from metadata)
python src/data/opencell/create_vlm_dataset.py \
    --output_dir /path/to/vlm_dataset \
    --expanded_format

# 2. Train VLM (Stage 1: projection layer only)
python src/train_vlm.py --config configs/opencell/vlm_3d.yaml --stage 1

# 3. Train VLM (Stage 2: fine-tune with LoRA)
python src/train_vlm.py --config configs/opencell/vlm_3d.yaml --stage 2 --resume /path/to/stage1/checkpoint.pt
```

VLM combines:
- Pretrained MAE encoder (vision)
- MLP projection layer
- Large language model (Mistral/LLaMA 7B)

QA types: identification, localization, abundance, protein interactions, descriptive.

## Additional Resources

- Main README: `/README.md` - User-facing guide with examples
- Trainer documentation: `src/TRAINER_README.md` - Deep dive on trainer pattern
- Normalization test: `test/test_normalization.py` - Verify data pipeline works
