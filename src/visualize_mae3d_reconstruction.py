#!/usr/bin/env python3
"""
Visualise MAE masking and reconstruction on OpenCell z-stack images.

Layout per sample (no text annotations):
  Row 0         — Max projection  → uses MAE-2D model (trained on max-projections)
  Rows 1..n_z   — Z-slices        → uses MAE-3D model (trained on full volumes)

Each row has 6 columns:
  Cols 0-1 : Original       (Nucleus blue | Protein gray)
  Cols 2-3 : Masked Input   (Nucleus blue | Protein gray)
  Cols 4-5 : Reconstruction (Nucleus blue | Protein gray)

Usage
-----
    python src/visualize_mae3d_reconstruction.py \\
        --config_2d     configs/opencell/opencell_2d_cross_attention_clip_kfold.yaml \\
        --ckpt_2d       /path/to/mae2d/fold0/ckpts/checkpoint.pth.tar \\
        --model_type_2d mae2d_clip \\
        --config_3d     configs/opencell/opencell_3d_cross_attention_clip_kfold.yaml \\
        --ckpt_3d       /path/to/mae3d/fold0/ckpts/checkpoint.pth.tar \\
        --model_type_3d mae3d_clip \\
        --csv_path      /path/to/fold0/train.csv \\
        --esm2_emb_path /path/to/esm2_embeddings_kfold5/fold0/train.npy \\
        --sample_indices 100 \\
        --n_z_slices    4 \\
        --output_dir    /path/to/output/
"""

import argparse
import importlib
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import tifffile

sys.path.insert(0, os.path.dirname(__file__))

from utils.utils import get_conf
from data.opencell.transforms import get_opencell_val_transforms, get_opencell_2d_val_transforms

# ── Colormaps ──────────────────────────────────────────────────────────────────
NUC_CMAP  = mcolors.LinearSegmentedColormap.from_list('nuc_blue', ['black', '#0088ff'])
PROT_CMAP = 'gray'

# ── Model registries ───────────────────────────────────────────────────────────
_MODEL_MAP_2D = {
    'mae2d_clip': ('lib.models.mae2d_cross_attention_clip', 'MAE2DChannelCrossAttentionCLIP'),
    'mae2d_fft':  ('lib.models.mae2d_cross_attention_fft',  'MAE2DChannelCrossAttentionFFT'),
}
_MODEL_MAP_3D = {
    'mae3d_base': ('lib.models.mae3d_cross_attention',      'MAE3DChannelCrossAttention'),
    'mae3d_fft':  ('lib.models.mae3d_cross_attention_fft',  'MAE3DChannelCrossAttentionFFT'),
    'mae3d_clip': ('lib.models.mae3d_cross_attention_clip', 'MAE3DChannelCrossAttentionCLIP'),
}


def build_and_load(config_path, ckpt_path, model_map, model_type, device):
    args = get_conf(config_path)
    mod_name, cls_name = model_map[model_type]
    ModelCls = getattr(importlib.import_module(mod_name), cls_name)
    model = ModelCls(args)

    ckpt = torch.load(ckpt_path, map_location='cpu')
    sd   = ckpt.get('state_dict', ckpt)
    sd   = {k[7:] if k.startswith('module.') else k: v for k, v in sd.items()}

    model_sd = model.state_dict()
    filtered, skipped = {}, []
    for k, v in sd.items():
        if k in model_sd and v.shape != model_sd[k].shape:
            skipped.append(k)
        else:
            filtered[k] = v
    if skipped:
        print(f"  Skipped {len(skipped)} shape-mismatched key(s): {skipped[:3]}"
              f"{'...' if len(skipped) > 3 else ''}")

    msg = model.load_state_dict(filtered, strict=False)
    print(f"  Missing: {len(msg.missing_keys)}  |  Unexpected: {len(msg.unexpected_keys)}")
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, args


# ── Patch → volume helpers ─────────────────────────────────────────────────────

def patches_to_image_2d(patches_cat, patch_size, grid_size):
    """
    [1, gY*gX, patch_area*2] → numpy (2, Y, X)
    """
    from lib.models.mae2d_cross_attention_fft import unpatchify_channelwise_2d

    patch_area = int(np.prod(patch_size))
    ch0 = patches_cat[:, :, :patch_area]
    ch1 = patches_cat[:, :, patch_area:]
    vol = unpatchify_channelwise_2d([ch0, ch1], patch_size, grid_size)
    return vol[0].cpu().numpy()   # (2, Y, X)


def patches_to_volume_3d(patches_cat, patch_size, grid_size):
    """
    [1, gZ*gY*gX, patch_vol*2] → numpy (2, Z, Y, X)
    """
    from lib.models.mae3d_cross_attention import unpatchify_image_channelwise

    patch_vol = int(np.prod(patch_size))
    ch0 = patches_cat[:, :, :patch_vol]
    ch1 = patches_cat[:, :, patch_vol:]
    vol = unpatchify_image_channelwise([ch0, ch1], patch_size, grid_size)
    return vol[0].cpu().numpy()   # (2, Z, Y, X)


# ── Forward pass helpers ───────────────────────────────────────────────────────

def run_forward(model, model_type, img_batch, esm2_emb):
    """Run forward with return_image=True; return (orig, recon, masked) patch tensors."""
    is_clip = model_type.endswith('clip')
    with torch.no_grad():
        if is_clip and esm2_emb is not None:
            out = model(img_batch, esm2_emb=esm2_emb,
                        return_image=True, return_clip_loss=True)
            _, _, orig_p, recon_p, masked_p = out
        else:
            out = model(img_batch, return_image=True)
            _, orig_p, recon_p, masked_p = out
    return orig_p, recon_p, masked_p


# ── Display helpers ────────────────────────────────────────────────────────────

def norm_img(img, lo_pct=1, hi_pct=99):
    lo, hi = np.percentile(img, lo_pct), np.percentile(img, hi_pct)
    return np.clip((img - lo) / (hi - lo + 1e-8), 0, 1)


def show_panel(ax, img, cmap):
    ax.imshow(img, cmap=cmap, vmin=0, vmax=1, interpolation='bilinear')
    ax.axis('off')


# ── Figure builder ─────────────────────────────────────────────────────────────

def make_reconstruction_figure(
    maxproj_orig,    # (2, Y, X) — from 2D model
    maxproj_masked,
    maxproj_recon,
    slices_orig,     # list of (2, Y, X) — from 3D model, one per z_index
    slices_masked,
    slices_recon,
    output_path,
):
    """
    No text. Layout: (1 + n_z) rows × 6 cols.
      Cols 0-1 : Original  (Nucleus | Protein)
      Cols 2-3 : Masked    (Nucleus | Protein)
      Cols 4-5 : Recon     (Nucleus | Protein)
    """
    n_z    = len(slices_orig)
    n_rows = 1 + n_z
    n_cols = 6

    col_w = 1.7
    fig = plt.figure(figsize=(n_cols * col_w, n_rows * col_w), facecolor='black')
    fig.patch.set_facecolor('black')

    gs = gridspec.GridSpec(
        n_rows, n_cols, figure=fig,
        top=1.0, bottom=0.0, left=0.0, right=1.0,
        hspace=0.02, wspace=0.02,
    )

    cmaps = [NUC_CMAP, PROT_CMAP, NUC_CMAP, PROT_CMAP, NUC_CMAP, PROT_CMAP]

    def fill_row(row_idx, orig, masked, recon):
        panels = [orig[0], orig[1], masked[0], masked[1], recon[0], recon[1]]
        for col, (img, cmap) in enumerate(zip(panels, cmaps)):
            ax = fig.add_subplot(gs[row_idx, col])
            ax.set_facecolor('black')
            show_panel(ax, norm_img(img), cmap)

    # Row 0: max projection (2D model)
    fill_row(0, maxproj_orig, maxproj_masked, maxproj_recon)

    # Rows 1..n_z: Z-slices (3D model)
    for r in range(n_z):
        fill_row(r + 1, slices_orig[r], slices_masked[r], slices_recon[r])

    fig.savefig(output_path, dpi=200, bbox_inches='tight',
                facecolor='black', edgecolor='none')
    fig.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight',
                facecolor='black', edgecolor='none')
    plt.close(fig)
    print(f'  → {output_path}')


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='MAE2D/3D masking & reconstruction visualisation on OpenCell'
    )
    # 2D model (used for max-projection row)
    parser.add_argument('--config_2d',     type=str, required=True)
    parser.add_argument('--ckpt_2d',       type=str, required=True)
    parser.add_argument('--model_type_2d', type=str, default='mae2d_clip',
                        choices=list(_MODEL_MAP_2D))
    # 3D model (used for Z-slice rows)
    parser.add_argument('--config_3d',     type=str, required=True)
    parser.add_argument('--ckpt_3d',       type=str, required=True)
    parser.add_argument('--model_type_3d', type=str, default='mae3d_clip',
                        choices=list(_MODEL_MAP_3D))
    # Data
    parser.add_argument('--csv_path',       type=str, required=True)
    parser.add_argument('--esm2_emb_path',  type=str, default=None)
    parser.add_argument('--sample_indices', type=str, default='0',
                        help='Comma-separated row indices into the CSV.')
    parser.add_argument('--n_z_slices',    type=int, default=4)
    parser.add_argument('--output_dir',    type=str, required=True)
    parser.add_argument('--cpu',           action='store_true')
    cmd = parser.parse_args()

    os.makedirs(cmd.output_dir, exist_ok=True)
    device = torch.device('cpu' if cmd.cpu or not torch.cuda.is_available() else 'cuda')
    print(f'Device: {device}')

    code_base = '/path/to/repository/'

    def absp(p):
        return p if os.path.isabs(p) else os.path.join(code_base, p)

    # ── Load models ───────────────────────────────────────────────────────────
    print('\n[1/2] Loading MAE-2D ...')
    model_2d, args_2d = build_and_load(
        absp(cmd.config_2d), cmd.ckpt_2d, _MODEL_MAP_2D, cmd.model_type_2d, device)
    pat_2d   = list(args_2d.patch_size)                        # [pY, pX]
    inp_2d   = list(args_2d.input_size)                        # [Y, X]
    grid_2d  = [inp_2d[i] // pat_2d[i] for i in range(2)]     # [gY, gX]

    print('\n[2/2] Loading MAE-3D ...')
    model_3d, args_3d = build_and_load(
        absp(cmd.config_3d), cmd.ckpt_3d, _MODEL_MAP_3D, cmd.model_type_3d, device)
    pat_3d   = list(args_3d.patch_size)                        # [pZ, pY, pX]
    inp_3d   = list(args_3d.input_size)                        # [Z, Y, X]
    grid_3d  = [inp_3d[i] // pat_3d[i] for i in range(3)]     # [gZ, gY, gX]
    Z        = inp_3d[0]

    # ── Load ESM2 embeddings (optional) ──────────────────────────────────────
    esm2_all = None
    if cmd.esm2_emb_path and os.path.isfile(cmd.esm2_emb_path):
        esm2_all = np.load(cmd.esm2_emb_path)   # (N, D)
        print(f'\nESM2 embeddings: {esm2_all.shape}')

    # ── Load CSV ──────────────────────────────────────────────────────────────
    df = pd.read_csv(cmd.csv_path)
    protein_col = 'folder_protein' if 'folder_protein' in df.columns else None
    indices = [int(i) for i in cmd.sample_indices.split(',')]
    print(f'\nProcessing {len(indices)} sample(s): {indices}')

    transform_3d = get_opencell_val_transforms()
    transform_2d = get_opencell_2d_val_transforms()

    # ── Process each sample ───────────────────────────────────────────────────
    for idx in indices:
        row      = df.iloc[idx]
        img_path = row['image_path']
        gene     = row[protein_col] if protein_col else f'sample_{idx}'
        print(f'\n═══  {gene}  (row {idx})  ═══')

        # ── Load raw image ────────────────────────────────────────────────────
        raw = tifffile.imread(img_path)   # (Z, C, Y, X) or (C, Z, Y, X)

        # ── 3D input: (1, 2, Z, Y, X) ─────────────────────────────────────────
        img_3d = transform_3d({'image': raw})['image']
        if not isinstance(img_3d, torch.Tensor):
            img_3d = torch.from_numpy(img_3d)
        img_3d_batch = img_3d.float().unsqueeze(0).to(device)

        # ── 2D input: max-project along Z → (1, 2, Y, X) ─────────────────────
        raw_2d = raw.max(axis=0)   # (C, Y, X)
        img_2d = transform_2d({'image': raw_2d})['image']
        if not isinstance(img_2d, torch.Tensor):
            img_2d = torch.from_numpy(img_2d)
        img_2d_batch = img_2d.float().unsqueeze(0).to(device)

        # ── ESM2 embedding ────────────────────────────────────────────────────
        esm2_emb = None
        if esm2_all is not None:
            esm2_emb = torch.from_numpy(
                esm2_all[idx].astype(np.float32)
            ).unsqueeze(0).to(device)

        # ── Run 2D forward (max-projection row) ───────────────────────────────
        print('  MAE-2D forward ...')
        orig_2d_p, recon_2d_p, masked_2d_p = run_forward(
            model_2d, cmd.model_type_2d, img_2d_batch, esm2_emb)
        orig_2d   = patches_to_image_2d(orig_2d_p,   pat_2d, grid_2d)   # (2, Y, X)
        recon_2d  = patches_to_image_2d(recon_2d_p,  pat_2d, grid_2d)
        masked_2d = patches_to_image_2d(masked_2d_p, pat_2d, grid_2d)

        # ── Run 3D forward (Z-slice rows) ─────────────────────────────────────
        print('  MAE-3D forward ...')
        orig_3d_p, recon_3d_p, masked_3d_p = run_forward(
            model_3d, cmd.model_type_3d, img_3d_batch, esm2_emb)
        orig_3d   = patches_to_volume_3d(orig_3d_p,   pat_3d, grid_3d)  # (2, Z, Y, X)
        recon_3d  = patches_to_volume_3d(recon_3d_p,  pat_3d, grid_3d)
        masked_3d = patches_to_volume_3d(masked_3d_p, pat_3d, grid_3d)

        # ── Select Z-slice positions ──────────────────────────────────────────
        z_margin  = max(1, Z // 8)
        z_indices = np.linspace(z_margin, Z - z_margin - 1,
                                cmd.n_z_slices, dtype=int).tolist()

        slices_orig    = [orig_3d[:, z]   for z in z_indices]
        slices_masked  = [masked_3d[:, z] for z in z_indices]
        slices_recon   = [recon_3d[:, z]  for z in z_indices]

        # ── Build figure ──────────────────────────────────────────────────────
        out_path = os.path.join(cmd.output_dir, f'recon_{gene}_row{idx}.png')
        make_reconstruction_figure(
            maxproj_orig=orig_2d,
            maxproj_masked=masked_2d,
            maxproj_recon=recon_2d,
            slices_orig=slices_orig,
            slices_masked=slices_masked,
            slices_recon=slices_recon,
            output_path=out_path,
        )

    print(f'\nAll outputs saved to: {cmd.output_dir}')


if __name__ == '__main__':
    main()
