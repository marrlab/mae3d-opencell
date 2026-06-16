# 3D Masked Autoencoders are Robust Learners of Volumetric and Multimodal Cellular Representations for Microscopy

2D / 3D Masked Autoencoders (MAE) for single-cell microscopy, with an InfoNCE
loss that aligns the learned image representation to the protein modality
(ESM2 protein-language-model embeddings).

**Dataset:** OpenCell (nucleus + protein channels)
**Base code:** Adapted from [SelfMedMAE](https://github.com/cvlab-stonybrook/SelfMedMAE)

---

## Models and the MAE2D\* / MAE3D\* lineage

The "starred" models (**MAE2D\*** / **MAE3D\***) are the final, protein-aligned
models: a channel cross-attention MAE pretrained with an additional **InfoNCE
(CLIP-style) contrastive loss** between the pooled image embedding and the
protein's ESM2 embedding. They are built on top of two intermediate stages,
which are **kept in this repository because the final models depend on them**:

```
2D:  MAE2D  ──►  + channel cross-attention + FFT loss  ──►  MAE2D* (+ InfoNCE)
3D:  MAE3D  ──►  + channel cross-attention + FFT loss  ──►  MAE3D* (+ InfoNCE)
```

- **Baseline MAE2D / MAE3D** — plain masked autoencoder (75% masking,
  channel-wise normalization).
- **Cross-attention** — dual-stream encoder/decoder where the nucleus and
  protein channels exchange information via position-wise cross-attention
  (`src/lib/networks/cross_attention.py`).
- **FFT** — adds a frequency-domain reconstruction loss
  (`src/lib/losses/fft2d_loss.py`, `fft3d_loss.py`).
- **MAE2D\* / MAE3D\* (InfoNCE)** — adds the protein-alignment loss. **The 3D
  CLIP recipe resumes from an FFT-pretrained checkpoint** (`resume:` in the
  CLIP config), so the FFT stage must be trained first. The 2D CLIP recipe
  likewise resumes from the 2D FFT checkpoint.

### InfoNCE protein-alignment loss

Defined as `info_nce_loss()` in `src/lib/models/mae3d_cross_attention_clip.py`
(and `mae2d_cross_attention_clip.py`). It L2-normalizes the pooled image
embedding and the projected ESM2 embedding, computes a temperature-scaled
cosine-similarity matrix over the batch, and applies a symmetric
(image→protein and protein→image) cross-entropy. The temperature is a learnable
parameter. The total loss during pretraining is:

```
loss = reconstruction_loss [+ fft_loss] + clip_weight * info_nce_loss
```

`clip_weight` is ramped up linearly over `clip_rampup_epochs`. ESM2 embeddings
are precomputed (see `extract_esm2_embeddings.py`) and supplied per-cell; they
are used **only in the loss** (`use_esm2_conditioning: false`), not as encoder
input.

---

## Project structure

```
src/
├── data/
│   └── opencell/                 # OpenCell datasets + transforms
│       ├── dataset.py            # OpenCellDataset (single-cell 2D/3D)
│       ├── localization_dataset.py
│       ├── ppi_dataset.py
│       └── transforms.py         # channel-wise normalization, augmentation
│
├── lib/
│   ├── models/                   # MAE2D/3D, cross-attn, FFT, CLIP, ViT/PPI heads
│   ├── networks/                 # ViT encoder/decoder, cross-attention, patch embed
│   ├── losses/                   # FFT2D / FFT3D losses
│   └── trainers/                 # one trainer per model (BaseTrainer subclasses)
│
├── train/opencell/               # training entry points
├── extract/opencell/             # embedding extraction (image + ESM2)
├── evaluate/opencell/            # localization & PPI evaluation
├── tools/opencell/               # k-fold split / k-fold ESM2 helpers
└── utils/                        # get_conf, set_seed, logging

configs/opencell/                 # YAML configs (see table below)
```

Models, trainers, datasets and losses are reusable modules; the scripts under
`train/`, `extract/`, `evaluate/` are thin entry points that load a YAML config
and call the relevant trainer/utility.

---

## Installation

### Reproducibility environment

The reported experiments were run in the conda environment **`sc_project`** with:

| Component | Version |
|-----------|---------|
| Python | 3.11.9 |
| CUDA (build) | 11.8 |
| cuDNN | 8.7 |
| PyTorch | 2.1.2 (`+cu118`) |
| torchvision | 0.16.2 |
| NumPy | 1.26.4 (torch 2.1.x requires `numpy < 2`) |

All dependencies and their exact pins live in `pyproject.toml`.

### Install with `uv`

[`uv`](https://docs.astral.sh/uv/) reads `pyproject.toml` and resolves the CUDA 11.8
PyTorch wheels automatically (configured via `[tool.uv.sources]`).

```bash
# 1. Install uv (if needed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Create the environment and install all pinned dependencies
uv venv --python 3.11        # creates .venv with Python 3.11
uv sync                      # installs torch 2.1.2+cu118, monai, transformers, ...

# 3. Run any script through the resulting environment
uv run python src/train/opencell/train_mae3d_opencell.py --config configs/opencell/opencell_3d.yaml
```

Optional ESM2 native backend (`fair-esm`); the HuggingFace `transformers` backend
is installed by default and needs nothing extra:

```bash
uv sync --extra esm
```

> Already using the `sc_project` conda env? It satisfies all of the above —
> just activate it (`conda activate sc_project`) and run the scripts directly.

The OpenCell single-cell crops are 3D TIFFs of shape `(100, 2, 176, 176)`
(Z, C, H, W); the 2D variants use the Z-max-projection (`176×176`, 2 channels).
Dataset/embedding locations are passed via the configs and CLI flags — search
the configs for the `/path/to/...` placeholders and point them at your data.

---

## Workflow

All commands are run from the repository root. Add
`torchrun --nproc_per_node=N` before the `python` call for multi-GPU training.

### 1. Pretraining

```bash
# Baseline MAE
python src/train/opencell/train_mae2d_opencell.py --config configs/opencell/opencell_2d.yaml
python src/train/opencell/train_mae3d_opencell.py            # uses opencell_3d.yaml

# 3D cross-attention (intermediate)
python src/train/opencell/train_mae3d_cross_attention_opencell.py \
    --config configs/opencell/opencell_3d_cross_attention.yaml

# FFT stage (prerequisite for the CLIP models)
python src/train/opencell/train_mae2d_cross_attention_fft_opencell.py \
    --config configs/opencell/opencell_2d_cross_attention_fft_kfold.yaml
python src/train/opencell/train_mae3d_cross_attention_fft_opencell.py \
    --config configs/opencell/opencell_3d_cross_attention_fft.yaml

# Final InfoNCE-aligned models (MAE2D* / MAE3D*) — resume from the FFT checkpoint
python src/train/opencell/train_mae2d_cross_attention_clip_opencell.py \
    --config configs/opencell/opencell_2d_cross_attention_clip_kfold.yaml
python src/train/opencell/train_mae3d_cross_attention_clip_opencell.py \
    --config configs/opencell/opencell_3d_cross_attention_clip.yaml
```

ESM2 protein embeddings (needed for the InfoNCE loss) are precomputed once:

```bash
python src/extract/opencell/extract_esm2_embeddings.py --csv_dir <dataset1/> --output_dir <esm2/>
# For k-fold runs, re-align them to each fold:
python src/tools/opencell/create_kfold_splits.py --data_dir <dataset1/> --output_dir <kfold5/>
python src/tools/opencell/create_kfold_esm2_embeddings.py \
    --global_csv_dir <dataset1/> --global_esm2_dir <esm2/> \
    --kfold_dir <kfold5/> --output_dir <esm2_kfold5/>
```

### 2. Embedding extraction

```bash
python src/extract/opencell/extract_embeddings_2d.py            --config <cfg> --checkpoint <ckpt>
python src/extract/opencell/extract_embeddings_3d.py            --config <cfg> --checkpoint <ckpt>
python src/extract/opencell/extract_embeddings_3d_cross_attention.py --config <cfg> --checkpoint <ckpt> --output_dir <out>
```

### 3. Downstream evaluation

**Protein subcellular localization** (17-class multi-label):

```bash
# Train a classifier on a (frozen / linear-probe) encoder or on extracted embeddings
python src/train/opencell/train_localization.py configs/opencell/opencell_localization_3d.yaml \
    [--mae_embedding_path <emb/> --mae_embedding_csv_path <dataset1/>]

# Evaluate (mAP, macro/micro AUC, macro/micro F1, per-class)
python src/evaluate/opencell/evaluate_localization.py \
    --config configs/opencell/opencell_localization_3d.yaml \
    --checkpoint <ckpt> --output results --split test
```

**Protein–protein interaction (PPI)**:

```bash
python src/train/opencell/train_ppi.py configs/opencell/opencell_ppi_3d.yaml \
    [--mae_embedding_path <emb/> --mae_embedding_csv_path <dataset1/>]

python src/evaluate/opencell/evaluate_ppi.py            --config configs/opencell/opencell_ppi_3d.yaml --checkpoint <ckpt>
python src/evaluate/opencell/evaluate_ppi_bootstrap.py  --output_dir <out>   # AUROC with bootstrap CIs
```

Use the 2D configs (`*_2d*.yaml`) for the 2D pipeline. `train_with_trainer.py`
is a generic entry point that dispatches to the correct trainer based on the
config's `trainer_name`.

---

## Configs

| Group | Files |
|-------|-------|
| Baseline MAE | `opencell_2d.yaml`, `opencell_3d.yaml` |
| Cross-attention | `opencell_3d_cross_attention.yaml` |
| FFT (prerequisite) | `opencell_2d_cross_attention_fft_kfold.yaml`, `opencell_3d_cross_attention_fft.yaml`, `opencell_3d_cross_attention_fft_kfold.yaml` |
| **InfoNCE final (MAE2D\*/MAE3D\*)** | `opencell_2d_cross_attention_clip_kfold.yaml`, `opencell_3d_cross_attention_clip.yaml`, `opencell_3d_cross_attention_clip_kfold.yaml` |
| Localization | `opencell_localization_2d.yaml`, `opencell_localization_3d.yaml`, `opencell_localization_3d_cross_attention.yaml`, `opencell_localization_emb_{2d,3d}_fft_kfold.yaml` |
| PPI | `opencell_ppi_2d.yaml`, `opencell_ppi_3d.yaml`, `opencell_ppi_emb_{2d,3d}_fft_kfold.yaml` |

Key fields: `arch`, `enc_arch` / `dec_arch`, `trainer_name`, `mask_ratio`
(0.75), `channel_wise_norm: true`, `intensity_augmentation: true`, and — for the
final models — `use_clip_loss`, `clip_weight`, `clip_embed_dim`,
`clip_rampup_epochs`, `clip_temperature_init`, `esm2_embedding_path`,
`esm2_embed_dim`.

---

## Citation

This work builds on the SelfMedMAE codebase and the OpenCell dataset, and uses the
Masked Autoencoder framework and ESM2 protein language model. Please cite the
relevant works:

**SelfMedMAE** — base code
([github.com/cvlab-stonybrook/SelfMedMAE](https://github.com/cvlab-stonybrook/SelfMedMAE)):

```bibtex
@inproceedings{zhou2023selfmedmae,
  title     = {Self Pre-training with Masked Autoencoders for Medical Image
               Classification and Segmentation},
  author    = {Zhou, Lei and Liu, Huidong and Bae, Joseph and He, Junjun and
               Samaras, Dimitris and Prasanna, Prateek},
  booktitle = {2023 IEEE 20th International Symposium on Biomedical Imaging (ISBI)},
  year      = {2023},
  doi       = {10.1109/ISBI53787.2023.10230477},
  note      = {arXiv:2203.05573}
}
```

**OpenCell** — dataset ([opencell.czbiohub.org](https://opencell.czbiohub.org)):

```bibtex
@article{cho2022opencell,
  title   = {OpenCell: Endogenous tagging for the cartography of human cellular
             organization},
  author  = {Cho, Nathan H. and Cheveralls, Keith C. and Brunner, Andr{\'e}-Denis G.
             and Kim, Kibeom and Michaelis, Andr{\'e} C. and Raghavan, Preethi and
             Kobayashi, Hirofumi and Savy, Laura and Li, Jason Y. and Canaj, Hera
             and others},
  journal = {Science},
  volume  = {375},
  number  = {6585},
  pages   = {eabi6983},
  year    = {2022},
  doi     = {10.1126/science.abi6983}
}
```

**Masked Autoencoders (MAE)** — method:

```bibtex
@inproceedings{he2022masked,
  title     = {Masked Autoencoders Are Scalable Vision Learners},
  author    = {He, Kaiming and Chen, Xinlei and Xie, Saining and Li, Yanghao and
               Doll{\'a}r, Piotr and Girshick, Ross},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and
               Pattern Recognition (CVPR)},
  pages     = {16000--16009},
  year      = {2022}
}
```

**ESM2** — protein language model used for the InfoNCE protein alignment:

```bibtex
@article{lin2023esm2,
  title   = {Evolutionary-scale prediction of atomic-level protein structure
             with a language model},
  author  = {Lin, Zeming and Akin, Halil and Rao, Roshan and Hie, Brian and
             Zhu, Zhongkai and Lu, Wenting and Smetanin, Nikita and Verkuil, Robert
             and Kabeli, Ori and Shmueli, Yaniv and others},
  journal = {Science},
  volume  = {379},
  number  = {6637},
  pages   = {1123--1130},
  year    = {2023},
  doi     = {10.1126/science.ade2574}
}
```
