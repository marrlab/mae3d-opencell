#!/usr/bin/env python3
"""
Attention map comparison: MAE2D vs MAE3D (no ESM2) vs MAE3D+CLIP/ESM2
on OpenCell sample images.

Two complementary attention types are visualised:
  1. Self-attention  — CLS token attends to all patch tokens in each channel
                       (hooks on encoder_blocks[-1].block_ch{0,1}.self_attn)
  2. Cross-channel gate — position-wise nucleus↔protein gate (sigmoid)
                          (hooks on encoder_blocks[-1].block_ch1.cross_attn.attn_drop)

Key visual argument
-------------------
  • 2D model: attention is over (gY, gX) = (22, 22) patches → flat map
    repeated at every Z slice.  The model is blind to depth.
  • 3D model: attention is over (gZ, gY, gX) = (10, 22, 22) patches →
    varies along the Z axis, showing WHERE in 3D space the protein lives.
  • 3D + ESM2/CLIP: ESM2 sequence embedding guides attention to focus on
    the biologically correct compartment more sharply.

Usage
-----
    python src/visualize_attention_comparison.py \\
        --config_2d      configs/opencell/opencell_2d_cross_attention_clip_kfold.yaml \\
        --ckpt_2d        /path/to/mae2d_clip/fold0/ckpts/checkpoint.pth.tar \\
        --model_type_2d  mae2d_clip \\
        --config_3d      configs/opencell/opencell_3d_cross_attention_fft_kfold.yaml \\
        --ckpt_3d        /path/to/mae3d_fft/fold0/ckpts/checkpoint.pth.tar \\
        --model_type_3d  mae3d_fft \\
        --config_3d_esm2 configs/opencell/opencell_3d_cross_attention_clip_kfold.yaml \\
        --ckpt_3d_esm2   /path/to/mae3d_clip/run3/fold0/ckpts/checkpoint_0004.pth.tar \\
        --model_type_3d_esm2 mae3d_clip \\
        --csv_path       /path/to/opencell/kfold5/fold0/val.csv \\
        --n_samples      4 \\
        --output_dir     /path/to/attention_viz/
"""

import argparse
import importlib
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

# Black-to-blue colormap for the nucleus (DAPI) channel
NUC_CMAP = mcolors.LinearSegmentedColormap.from_list('nuc_blue', ['black', '#0088ff'])
import pandas as pd
import torch
import torch.nn.functional as F
import tifffile
from scipy.ndimage import zoom

sys.path.insert(0, os.path.dirname(__file__))

from utils.utils import get_conf
from data.opencell.dataset import OpenCellDataset
from data.opencell.transforms import get_opencell_val_transforms, get_opencell_2d_val_transforms


# ── Model registry ─────────────────────────────────────────────────────────────

_MODEL_MAP = {
    'mae2d_clip': ('lib.models.mae2d_cross_attention_clip',  'MAE2DChannelCrossAttentionCLIP'),
    'mae2d_fft':  ('lib.models.mae2d_cross_attention_fft',   'MAE2DChannelCrossAttentionFFT'),
    'mae3d_base': ('lib.models.mae3d_cross_attention',       'MAE3DChannelCrossAttention'),
    'mae3d_fft':  ('lib.models.mae3d_cross_attention_fft',   'MAE3DChannelCrossAttentionFFT'),
    'mae3d_clip': ('lib.models.mae3d_cross_attention_clip',  'MAE3DChannelCrossAttentionCLIP'),
}


def build_and_load(config_path, ckpt_path, model_type, device):
    """
    Build a model from config and load checkpoint.
    Shape-mismatched keys (e.g. pos-embed across different grid sizes) are
    silently skipped so the model uses freshly-computed sincos embeddings.
    """
    args = get_conf(config_path)
    mod_name, cls_name = _MODEL_MAP[model_type]
    ModelCls = getattr(importlib.import_module(mod_name), cls_name)
    model = ModelCls(args)

    ckpt = torch.load(ckpt_path, map_location='cpu')
    sd = ckpt.get('state_dict', ckpt)
    sd = {k[7:] if k.startswith('module.') else k: v for k, v in sd.items()}

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


# ── Attention capture hooks ────────────────────────────────────────────────────

class MHAWeightCapture:
    """
    Captures attention weight tensor from nn.MultiheadAttention via forward hook.
    nn.MultiheadAttention.forward() returns (output, attn_weights) by default
    (need_weights=True).  The ChannelCrossAttentionBlock discards weights with
    `attn_out, _ = self.self_attn(...)`, but the hook sees the full tuple.
    """
    def __init__(self):
        self.weights = None   # [B, N+1, N+1] head-averaged
        self._handle = None

    def register(self, mha_module):
        self._handle = mha_module.register_forward_hook(self._hook)
        return self

    def _hook(self, module, inp, out):
        if isinstance(out, (tuple, list)) and len(out) == 2 and out[1] is not None:
            self.weights = out[1].detach().cpu()

    def remove(self):
        if self._handle is not None:
            self._handle.remove()


class CrossGateCapture:
    """
    Captures PositionWiseCrossAttention gate weights via a hook on attn_drop.
    In eval mode Dropout is a no-op, so the hook input equals the gate tensor.
    Gates have shape [B, num_heads, N] (N includes CLS token).
    """
    def __init__(self):
        self.gates = None
        self._handle = None

    def register(self, dropout_module):
        self._handle = dropout_module.register_forward_hook(self._hook)
        return self

    def _hook(self, module, inp, out):
        # inp[0] == out == attn_weights [B, num_heads, N] in eval mode
        if inp and inp[0] is not None:
            self.gates = inp[0].detach().cpu()

    def remove(self):
        if self._handle is not None:
            self._handle.remove()


# ── Feature-similarity fallback ───────────────────────────────────────────────

def _feature_similarity_map(x_ch):
    """
    Cosine similarity of each patch token to the CLS token.
    Used when MHA attention weights are not available (e.g. flash-attn path).

    Args:
        x_ch: [B, N+1, D]  (token 0 = CLS)
    Returns:
        sim: [B, N]  normalised to [0, 1]
    """
    cls    = F.normalize(x_ch[:, 0:1, :], dim=-1)   # [B, 1, D]
    patches = F.normalize(x_ch[:, 1:, :],  dim=-1)   # [B, N, D]
    sim = (cls * patches).sum(dim=-1)                 # [B, N]
    sim = sim - sim.min(dim=1, keepdim=True).values
    sim = sim / (sim.max(dim=1, keepdim=True).values + 1e-8)
    return sim.cpu().numpy()


# ── Attention extraction ───────────────────────────────────────────────────────

def extract_attn_3d(model, img_3d, grid_size, device, layer_idx=-1):
    """
    Extract self-attention and cross-channel gate maps for a 3D model.

    Args:
        img_3d:    [1, 2, Z, Y, X]  (already preprocessed)
        grid_size: (gZ, gY, gX)

    Returns:
        self_attn_ch0: (gZ, gY, gX)  — nucleus self-attention
        self_attn_ch1: (gZ, gY, gX)  — protein self-attention
        cross_gate:    (gZ, gY, gX) or None — nucleus→protein cross gate
        methods:       dict of extraction methods used
    """
    gZ, gY, gX = grid_size
    block = model.encoder_blocks[layer_idx]

    cap_sa0 = MHAWeightCapture().register(block.block_ch0.self_attn)
    cap_sa1 = MHAWeightCapture().register(block.block_ch1.self_attn)
    cap_cg1 = CrossGateCapture()
    if hasattr(block.block_ch1, 'cross_attn') and hasattr(block.block_ch1.cross_attn, 'attn_drop'):
        cap_cg1.register(block.block_ch1.cross_attn.attn_drop)

    with torch.no_grad():
        x_ch0, x_ch1 = model.forward_encoder_no_mask(img_3d.to(device))

    cap_sa0.remove(); cap_sa1.remove(); cap_cg1.remove()

    methods = {}

    def _process_mha(cap, x_ch, name):
        if cap.weights is not None:
            # CLS-to-patches attention: row 0, columns 1..N
            raw = cap.weights[0, 0, 1:].numpy()   # [N]
            methods[name] = 'mha_weights'
        else:
            raw = _feature_similarity_map(x_ch)[0]  # [N]
            methods[name] = 'feature_similarity'
        raw = (raw - raw.min()) / (raw.max() - raw.min() + 1e-8)
        return raw.reshape(gZ, gY, gX)

    sa_ch0 = _process_mha(cap_sa0, x_ch0, 'self_ch0')
    sa_ch1 = _process_mha(cap_sa1, x_ch1, 'self_ch1')

    cross_gate = None
    if cap_cg1.gates is not None:
        # gates: [B, num_heads, N+1]  — average over heads, drop CLS position
        raw = cap_cg1.gates[0, :, 1:].mean(0).numpy()   # [N]
        raw = (raw - raw.min()) / (raw.max() - raw.min() + 1e-8)
        cross_gate = raw.reshape(gZ, gY, gX)
        methods['cross_gate'] = 'position_wise_sigmoid'

    return sa_ch0, sa_ch1, cross_gate, methods


def extract_attn_2d(model, img_2d, grid_size, device, layer_idx=-1):
    """
    Extract self-attention map for a 2D model.

    Args:
        img_2d:    [1, 2, Y, X]
        grid_size: (gY, gX)

    Returns:
        self_attn_ch0: (gY, gX)  — nucleus
        self_attn_ch1: (gY, gX)  — protein
        method:        str
    """
    gY, gX = grid_size
    block = model.encoder_blocks[layer_idx]

    cap0 = MHAWeightCapture().register(block.block_ch0.self_attn)
    cap1 = MHAWeightCapture().register(block.block_ch1.self_attn)

    with torch.no_grad():
        x_ch0, x_ch1 = model.forward_encoder_no_mask(img_2d.to(device))

    cap0.remove(); cap1.remove()

    def _proc(cap, x_ch):
        if cap.weights is not None:
            raw = cap.weights[0, 0, 1:].numpy()
            method = 'mha_weights'
        else:
            raw = _feature_similarity_map(x_ch)[0]
            method = 'feature_similarity'
        raw = (raw - raw.min()) / (raw.max() - raw.min() + 1e-8)
        return raw.reshape(gY, gX), method

    a0, m0 = _proc(cap0, x_ch0)
    a1, m1 = _proc(cap1, x_ch1)
    return a0, a1, {'self_ch0': m0, 'self_ch1': m1}


# ── Upsampling ─────────────────────────────────────────────────────────────────

def upsample_3d(attn, tZ, tY, tX):
    gZ, gY, gX = attn.shape
    return zoom(attn, (tZ / gZ, tY / gY, tX / gX), order=1)


def upsample_2d(attn, tY, tX):
    gY, gX = attn.shape
    return zoom(attn, (tY / gY, tX / gX), order=1)


# ── Display helpers ────────────────────────────────────────────────────────────

def norm_img(img, lo_pct=1, hi_pct=99):
    lo, hi = np.percentile(img, lo_pct), np.percentile(img, hi_pct)
    return np.clip((img - lo) / (hi - lo + 1e-8), 0, 1)


def show_attn_overlay(ax, image, attn, cmap='inferno', alpha=0.55, title='', fontsize=7):
    ax.imshow(image, cmap='gray', vmin=0, vmax=1, interpolation='bilinear')
    ax.imshow(attn,  cmap=cmap,   vmin=0, vmax=1, alpha=alpha, interpolation='bilinear')
    ax.set_title(title, fontsize=fontsize, pad=2)
    ax.axis('off')


def show_image(ax, image, cmap='gray', title='', fontsize=7):
    ax.imshow(image, cmap=cmap, vmin=0, vmax=1, interpolation='bilinear')
    ax.set_title(title, fontsize=fontsize, pad=2)
    ax.axis('off')


# ── Quantitative depth metric ─────────────────────────────────────────────────

def compute_per_z_spatial_correlation(prot_vol, attn_vol):
    """
    For each Z slice, compute the Pearson r between the spatial attention
    map and the protein intensity image.

    A 3D model that is depth-aware should show HIGH correlation at Z levels
    where the protein is concentrated, and low correlation elsewhere.
    A 2D model shows roughly uniform correlation (no depth discrimination).

    Args:
        prot_vol: (Z, Y, X) float, normalised protein channel
        attn_vol: (Z, Y, X) float, normalised attention map (or repeated 2D map)

    Returns:
        corr: (Z,) Pearson r per Z-slice (NaN → 0)
    """
    Z = prot_vol.shape[0]
    corr = np.zeros(Z)
    for z in range(Z):
        p = prot_vol[z].flatten().astype(np.float64)
        a = attn_vol[z].flatten().astype(np.float64)
        if p.std() > 1e-8 and a.std() > 1e-8:
            corr[z] = float(np.corrcoef(p, a)[0, 1])
    return corr


# ── Main figure ────────────────────────────────────────────────────────────────

def make_sample_figure(
    sample_name,
    nuc_3d, prot_3d,          # (Z, Y, X) normalised float
    # 2D attention (Y, X)
    attn_2d_nuc, attn_2d_prot,
    # 3D attention (Z, Y, X)
    attn_3d_nuc,     attn_3d_prot,     attn_3d_cross,
    attn_esm2_nuc,   attn_esm2_prot,   attn_esm2_cross,
    z_indices,        # list of Z positions to show
    output_path,
):
    """
    Publication-quality layout (MICCAI style).

    Image rows × 5 cols, then an optional cross-gate row, then a plot row
    with Z-profile (left) and per-Z Pearson r (right).
    Uses constrained_layout to avoid overlap; height_ratios gives the plot
    row more vertical space than the image rows.
    """
    Z, Y, X  = prot_3d.shape
    n_z      = len(z_indices)
    has_gate = (attn_3d_cross is not None) or (attn_esm2_cross is not None)
    n_img    = 1 + n_z + 1 + int(has_gate)   # image rows (incl. optional gate)
    n_rows   = n_img + 1                      # +1 for plot row
    n_cols   = 5

    # Image rows ≈ square (width/n_cols), plot row gets 3.2× that height
    col_w    = 2.0          # inches per column
    img_h    = col_w        # keep images square
    plot_h   = col_w * 3.2  # taller row for the two graphs (extra space for bigger fonts)
    fig_w    = n_cols * col_w
    fig_h    = n_img * img_h + plot_h + 0.45   # 0.45 for suptitle

    height_ratios = [1.0] * n_img + [plot_h / img_h]

    # constrained_layout handles all padding automatically
    fig = plt.figure(figsize=(fig_w, fig_h), constrained_layout=True, facecolor='white')
    fig.suptitle(sample_name, fontsize=12, fontweight='bold')

    gs = gridspec.GridSpec(n_rows, n_cols, figure=fig, height_ratios=height_ratios)

    FS  = 9    # panel label font size
    FSP = 11   # plot axis font size

    col_titles = [
        'Nucleus\n(input)', 'Protein\n(input)',
        'MAE-2D\n(attention)', 'MAE-3D\n(no ESM2)', 'MAE-3D+ESM2\n(ours)',
    ]

    # ── Row 0: Max projection ────────────────────────────────────────────────
    row = 0
    nuc_max   = nuc_3d.max(0)
    prot_max  = prot_3d.max(0)
    panels = [
        (nuc_max,  None,                  col_titles[0]),
        (prot_max, None,                  col_titles[1]),
        (prot_max, attn_2d_prot,          col_titles[2]),
        (prot_max, attn_3d_prot.max(0),   col_titles[3]),
        (prot_max, attn_esm2_prot.max(0), col_titles[4]),
    ]
    for col, (img, attn, title) in enumerate(panels):
        ax = fig.add_subplot(gs[row, col])
        img_cmap = NUC_CMAP if col == 0 else 'gray'
        if attn is None:
            show_image(ax, img, cmap=img_cmap, title=title, fontsize=FS)
        else:
            show_attn_overlay(ax, img, attn, title=title, fontsize=FS)
        if col == 0:
            ax.set_ylabel('Max proj.', fontsize=FS, labelpad=2)
            ax.axis('on'); ax.set_xticks([]); ax.set_yticks([])

    # ── Rows 1..n_z: Selected Z-slices ──────────────────────────────────────
    for r, z in enumerate(z_indices):
        row = r + 1
        prot_z = prot_3d[z]
        nuc_z  = nuc_3d[z]
        panels = [
            (nuc_z,  None,              ''),
            (prot_z, None,              ''),
            (prot_z, attn_2d_prot,      ''),   # same at every Z — intentionally flat
            (prot_z, attn_3d_prot[z],   ''),
            (prot_z, attn_esm2_prot[z], ''),
        ]
        for col, (img, attn, _) in enumerate(panels):
            ax = fig.add_subplot(gs[row, col])
            img_cmap = NUC_CMAP if col == 0 else 'gray'
            if attn is None:
                show_image(ax, img, cmap=img_cmap, title='', fontsize=FS)
            else:
                show_attn_overlay(ax, img, attn, title='', fontsize=FS)
            # Annotate 2D column once to flag depth-blindness
            if col == 2 and r == 0:
                ax.text(0.5, 0.03, '\u2605 identical at every Z',
                        transform=ax.transAxes, ha='center', va='bottom',
                        fontsize=8, color='white',
                        bbox=dict(facecolor='steelblue', alpha=0.80,
                                  pad=1.5, edgecolor='none'))
            if col == 0:
                ax.set_ylabel(f'Z\u202f=\u202f{z}', fontsize=FS, labelpad=2)
                ax.axis('on'); ax.set_xticks([]); ax.set_yticks([])

    # ── Row n_z+1: XZ cross-section (depth view) ────────────────────────────
    row = n_z + 1
    cy      = Y // 2
    nuc_xz  = nuc_3d[:, cy, :]
    prot_xz = prot_3d[:, cy, :]
    a2d_xz  = np.repeat(attn_2d_prot[np.newaxis, cy, :], Z, axis=0)
    a3d_xz  = attn_3d_prot[:, cy, :]
    ae2_xz  = attn_esm2_prot[:, cy, :]
    panels = [
        (nuc_xz,  None,    'Nucleus XZ'),
        (prot_xz, None,    'Protein XZ'),
        (prot_xz, a2d_xz,  'MAE-2D XZ\n(uniform depth)'),
        (prot_xz, a3d_xz,  'MAE-3D XZ'),
        (prot_xz, ae2_xz,  'MAE-3D+ESM2 XZ'),
    ]
    for col, (img, attn, title) in enumerate(panels):
        ax = fig.add_subplot(gs[row, col])
        img_cmap = NUC_CMAP if col == 0 else 'gray'
        if attn is None:
            show_image(ax, img, cmap=img_cmap, title=title, fontsize=FS)
        else:
            show_attn_overlay(ax, img, attn, title=title, fontsize=FS)
        if col == 0:
            ax.set_ylabel('XZ\u202f(depth)', fontsize=FS, labelpad=2)
            ax.axis('on'); ax.set_xticks([]); ax.set_yticks([])

    # ── Optional row: Cross-channel gates ────────────────────────────────────
    if has_gate:
        row = n_z + 2
        for col in range(n_cols):
            ax = fig.add_subplot(gs[row, col])
            if col == 0:
                ax.set_ylabel('Cross-gate\n(nuc\u2192prot)', fontsize=FS, labelpad=2)
                ax.axis('on'); ax.set_xticks([]); ax.set_yticks([])
                show_image(ax, nuc_3d.max(0), cmap=NUC_CMAP, title='Nucleus ref.', fontsize=FS)
            elif col == 1:
                show_image(ax, prot_3d.max(0), title='Protein ref.', fontsize=FS)
            elif col == 2:
                ax.text(0.5, 0.5, 'N/A\n(2D: no depth gates)',
                        transform=ax.transAxes, ha='center', va='center',
                        fontsize=FS, color='gray', style='italic')
                ax.set_title('MAE-2D', fontsize=FS, pad=2)
                ax.axis('off')
            elif col == 3:
                if attn_3d_cross is not None:
                    show_attn_overlay(ax, prot_3d.max(0), attn_3d_cross.max(0),
                                      cmap='plasma', title='MAE-3D gate', fontsize=FS)
                else:
                    ax.text(0.5, 0.5, 'no gate', transform=ax.transAxes,
                            ha='center', va='center', fontsize=FS, color='gray')
                    ax.axis('off')
            elif col == 4:
                if attn_esm2_cross is not None:
                    show_attn_overlay(ax, prot_3d.max(0), attn_esm2_cross.max(0),
                                      cmap='plasma', title='MAE-3D+ESM2 gate', fontsize=FS)
                else:
                    ax.text(0.5, 0.5, 'no gate', transform=ax.transAxes,
                            ha='center', va='center', fontsize=FS, color='gray')
                    ax.axis('off')

    # ── Plot row: two side-by-side axes via subgridspec ───────────────────────
    # subgridspec gives each plot its own proper bounding box with controlled gap
    row     = n_z + 2 + int(has_gate)
    gs_bot  = gs[row, :].subgridspec(1, 2, wspace=0.38)
    ax_l    = fig.add_subplot(gs_bot[0, 0])
    ax_r    = fig.add_subplot(gs_bot[0, 1])

    z_ax         = np.arange(Z)
    prot_profile = prot_3d.mean(axis=(1, 2))
    prot_profile = (prot_profile - prot_profile.min()) / (prot_profile.max() - prot_profile.min() + 1e-8)
    a2d_profile  = np.full(Z, attn_2d_prot.mean())
    a3d_profile  = attn_3d_prot.mean(axis=(1, 2))
    ae2_profile  = attn_esm2_prot.mean(axis=(1, 2))

    # — Left: Z-profile —
    ax_l.fill_between(z_ax, prot_profile, alpha=0.12, color='forestgreen')
    ax_l.plot(z_ax, prot_profile, color='forestgreen', lw=2.0, alpha=0.8,
              label='Protein intensity')
    ax_l.plot(z_ax, a2d_profile,  color='steelblue',  lw=2.5, ls='--',
              label='MAE-2D (flat)')
    ax_l.plot(z_ax, a3d_profile,  color='orangered',  lw=2.5,
              label='MAE-3D')
    ax_l.plot(z_ax, ae2_profile,  color='black',      lw=2.5,
              label='MAE-3D+ESM2')
    for z in z_indices:
        ax_l.axvline(z, color='gray', ls=':', lw=0.9, alpha=0.5)
    ax_l.set_xlabel('Z-slice index', fontsize=FSP)
    ax_l.set_ylabel('Mean attention (norm.)', fontsize=FSP)
    ax_l.set_title('Z-profile: 3D attention tracks depth; 2D is flat',
                   fontsize=FSP, pad=6)
    ax_l.legend(fontsize=FSP, loc='upper right',
                framealpha=0.85, handlelength=2.0, borderpad=0.6)
    ax_l.set_xlim(0, Z - 1)
    ax_l.set_ylim(-0.02, 1.10)
    ax_l.grid(True, alpha=0.25, lw=0.6)
    ax_l.tick_params(labelsize=FSP)
    ax_l.spines[['top', 'right']].set_visible(False)

    # — Right: per-Z Pearson r —
    corr_2d = compute_per_z_spatial_correlation(
        prot_3d, np.stack([attn_2d_prot] * Z, axis=0))
    corr_3d = compute_per_z_spatial_correlation(prot_3d, attn_3d_prot)
    corr_e2 = compute_per_z_spatial_correlation(prot_3d, attn_esm2_prot)

    ax_r.plot(z_ax, corr_2d, color='steelblue', lw=2.5, ls='--', label='MAE-2D')
    ax_r.plot(z_ax, corr_3d, color='orangered', lw=2.5,           label='MAE-3D')
    ax_r.plot(z_ax, corr_e2, color='black',     lw=2.5,           label='MAE-3D+ESM2')
    ax_r.axhline(0, color='gray', lw=1.0, ls=':')
    for z in z_indices:
        ax_r.axvline(z, color='gray', ls=':', lw=0.9, alpha=0.5)
    ax_r.set_xlabel('Z-slice index', fontsize=FSP)
    ax_r.set_ylabel('Pearson\u202fr (attention \u2194 protein)', fontsize=FSP)
    ax_r.set_title('Per-Z spatial correlation (\u2191 = better depth alignment)',
                   fontsize=FSP, pad=6)
    ax_r.legend(fontsize=FSP, loc='upper right',
                framealpha=0.85, handlelength=2.0, borderpad=0.6)
    ax_r.set_xlim(0, Z - 1)
    ax_r.grid(True, alpha=0.25, lw=0.6)
    ax_r.tick_params(labelsize=FSP)
    ax_r.spines[['top', 'right']].set_visible(False)

    fig.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    fig.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  → {output_path}')


# ── Summary figure (all samples, protein-channel max-proj only) ───────────────

def make_summary_figure(records, output_path):
    """
    One row per sample, columns: nucleus | protein | 2D | 3D-FFT | 3D+ESM2.
    """
    n = len(records)
    fig, axes = plt.subplots(n, 5, figsize=(5 * 2.3, n * 2.2), facecolor='white')
    if n == 1:
        axes = axes[np.newaxis, :]

    col_headers = ['Nucleus (max proj)', 'Protein (max proj)',
                   'MAE2D attention', 'MAE3D FFT attention', 'MAE3D+ESM2 attention']
    for col, h in enumerate(col_headers):
        axes[0, col].set_title(h, fontsize=8, fontweight='bold')

    for row, rec in enumerate(records):
        nuc_max  = rec['nuc_3d'].max(0)
        prot_max = rec['prot_3d'].max(0)
        a2d   = rec['attn_2d_prot']
        a3d   = rec['attn_3d_prot'].max(0)
        ae2   = rec['attn_esm2_prot'].max(0)

        panels = [nuc_max, prot_max, (prot_max, a2d), (prot_max, a3d), (prot_max, ae2)]
        for col, panel in enumerate(panels):
            ax = axes[row, col]
            if isinstance(panel, tuple):
                show_attn_overlay(ax, *panel)
            else:
                show_image(ax, panel, cmap=NUC_CMAP if col == 0 else 'gray')
        axes[row, 0].set_ylabel(rec['name'], fontsize=7, labelpad=2)

    plt.tight_layout(pad=0.5)
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'Summary → {output_path}')


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Attention visualisation: MAE2D vs MAE3D vs MAE3D+ESM2 on OpenCell'
    )
    # --- 2D model ---
    parser.add_argument('--config_2d',       type=str, required=True)
    parser.add_argument('--ckpt_2d',         type=str, required=True)
    parser.add_argument('--model_type_2d',   type=str, default='mae2d_clip',
                        choices=list(_MODEL_MAP))
    # --- 3D model (no ESM2) ---
    parser.add_argument('--config_3d',       type=str, required=True)
    parser.add_argument('--ckpt_3d',         type=str, required=True)
    parser.add_argument('--model_type_3d',   type=str, default='mae3d_fft',
                        choices=list(_MODEL_MAP))
    # --- 3D + ESM2 model (best) ---
    parser.add_argument('--config_3d_esm2',      type=str, required=True)
    parser.add_argument('--ckpt_3d_esm2',        type=str, required=True)
    parser.add_argument('--model_type_3d_esm2',  type=str, default='mae3d_clip',
                        choices=list(_MODEL_MAP))
    # --- data ---
    parser.add_argument('--csv_path',     type=str, required=True,
                        help='val.csv (or any fold CSV) from OpenCell kfold.')
    parser.add_argument('--n_samples',    type=int, default=4)
    parser.add_argument('--sample_indices', type=str, default=None,
                        help='Comma-separated row indices (overrides --n_samples).')
    parser.add_argument('--n_z_slices',   type=int, default=4,
                        help='Number of Z slices to show per sample.')
    parser.add_argument('--attn_layer',   type=int, default=-1,
                        help='Encoder layer index to extract attention from (-1 = last).')
    parser.add_argument('--output_dir',   type=str, required=True)
    parser.add_argument('--cpu',          action='store_true',
                        help='Force CPU (useful on login nodes).')
    cmd = parser.parse_args()

    os.makedirs(cmd.output_dir, exist_ok=True)
    device = torch.device('cpu' if cmd.cpu or not torch.cuda.is_available() else 'cuda')
    print(f'Device: {device}')

    code_base = '/path/to/repository/'

    def absp(p):
        return p if os.path.isabs(p) else os.path.join(code_base, p)

    # ── Load models ───────────────────────────────────────────────────────────
    print('\n[1/3] Loading MAE2D ...')
    model_2d,   args_2d   = build_and_load(absp(cmd.config_2d),
                                            cmd.ckpt_2d, cmd.model_type_2d, device)

    print('\n[2/3] Loading MAE3D FFT (no ESM2) ...')
    model_3d,   args_3d   = build_and_load(absp(cmd.config_3d),
                                            cmd.ckpt_3d, cmd.model_type_3d, device)

    print('\n[3/3] Loading MAE3D+ESM2/CLIP ...')
    model_esm2, args_esm2 = build_and_load(absp(cmd.config_3d_esm2),
                                            cmd.ckpt_3d_esm2, cmd.model_type_3d_esm2, device)

    # Compute grid sizes from configs
    inp_3d   = list(args_3d.input_size)    # [100, 176, 176]
    pat_3d   = list(args_3d.patch_size)    # [10, 8, 8]
    grid_3d  = [inp_3d[i] // pat_3d[i] for i in range(3)]   # [10, 22, 22]
    Z, Y, X  = inp_3d

    inp_2d   = list(args_2d.input_size)    # [176, 176]
    pat_2d   = list(args_2d.patch_size)    # [8, 8]
    grid_2d  = [inp_2d[i] // pat_2d[i] for i in range(2)]   # [22, 22]

    print(f'\n3D grid: {grid_3d}  |  2D grid: {grid_2d}')

    # ── Load sample list ──────────────────────────────────────────────────────
    df = pd.read_csv(cmd.csv_path)
    protein_col = 'folder_protein' if 'folder_protein' in df.columns else None

    if cmd.sample_indices is not None:
        indices = [int(i) for i in cmd.sample_indices.split(',')]
    elif protein_col is not None:
        proteins = df[protein_col].unique()
        step = max(1, len(proteins) // cmd.n_samples)
        sel_proteins = proteins[::step][:cmd.n_samples]
        indices = [df[df[protein_col] == p].index[0] for p in sel_proteins]
    else:
        indices = np.linspace(0, len(df) - 1, cmd.n_samples, dtype=int).tolist()

    print(f'\nSelected {len(indices)} sample row(s): {indices}')

    transform_3d = get_opencell_val_transforms()
    transform_2d = get_opencell_2d_val_transforms()

    # ── Process each sample ───────────────────────────────────────────────────
    summary_records = []

    for idx in indices:
        row = df.iloc[idx]
        img_path = row['image_path']
        gene = row[protein_col] if protein_col else f'sample_{idx}'
        sample_name = f'{gene}  (row {idx})'
        print(f'\n═══  {sample_name}  ═══')

        # Load raw image and apply transforms
        raw = tifffile.imread(img_path)

        # 3D input: (1, 2, Z, Y, X)
        d3 = transform_3d({'image': raw})
        img_3d = d3['image']
        if not isinstance(img_3d, torch.Tensor):
            img_3d = torch.from_numpy(img_3d)
        if img_3d.dim() == 4:     # (2, Z, Y, X) → already correct
            img_3d_batch = img_3d.unsqueeze(0).float()
        else:
            img_3d_batch = img_3d.float().unsqueeze(0)

        # 2D input: max-project raw (Z, C, Y, X) along Z → (C, Y, X)
        # transform_2d expects (C, Y, X), not the full 3D volume
        raw_2d = raw.max(axis=0)                      # (C, Y, X)
        d2 = transform_2d({'image': raw_2d})
        img_2d = d2['image']
        if not isinstance(img_2d, torch.Tensor):
            img_2d = torch.from_numpy(img_2d)
        img_2d_batch = img_2d.float().unsqueeze(0)   # (1, 2, Y, X)

        # Display arrays: protein ch1, nucleus ch0 — shape (Z, Y, X)
        prot_3d_raw = img_3d_batch[0, 1].numpy()    # (Z, Y, X)
        nuc_3d_raw  = img_3d_batch[0, 0].numpy()    # (Z, Y, X)
        prot_3d_d   = norm_img(prot_3d_raw)
        nuc_3d_d    = norm_img(nuc_3d_raw)

        # ── Extract attention maps ────────────────────────────────────────────
        print('  Extracting MAE2D attention ...')
        a2d_nuc_g, a2d_prot_g, m2d = extract_attn_2d(
            model_2d, img_2d_batch, grid_2d, device, layer_idx=cmd.attn_layer)
        print(f"    methods: {m2d}")

        print('  Extracting MAE3D FFT attention ...')
        a3d_nuc_g, a3d_prot_g, a3d_cross_g, m3d = extract_attn_3d(
            model_3d, img_3d_batch, grid_3d, device, layer_idx=cmd.attn_layer)
        print(f"    methods: {m3d}")

        print('  Extracting MAE3D+ESM2 attention ...')
        ae2_nuc_g, ae2_prot_g, ae2_cross_g, me2 = extract_attn_3d(
            model_esm2, img_3d_batch, grid_3d, device, layer_idx=cmd.attn_layer)
        print(f"    methods: {me2}")

        # Upsample to full image resolution
        a2d_nuc  = upsample_2d(a2d_nuc_g,   Y, X)
        a2d_prot = upsample_2d(a2d_prot_g,  Y, X)

        a3d_nuc  = upsample_3d(a3d_nuc_g,   Z, Y, X)
        a3d_prot = upsample_3d(a3d_prot_g,  Z, Y, X)
        a3d_cross = upsample_3d(a3d_cross_g, Z, Y, X) if a3d_cross_g is not None else None

        ae2_nuc  = upsample_3d(ae2_nuc_g,   Z, Y, X)
        ae2_prot = upsample_3d(ae2_prot_g,  Z, Y, X)
        ae2_cross = upsample_3d(ae2_cross_g, Z, Y, X) if ae2_cross_g is not None else None

        # Choose Z-slice positions spaced evenly in [10%, 90%] of Z range
        z_margin  = max(1, Z // 8)
        z_indices = np.linspace(z_margin, Z - z_margin - 1, cmd.n_z_slices, dtype=int).tolist()

        # ── Per-sample figure ─────────────────────────────────────────────────
        out_path = os.path.join(cmd.output_dir, f'attn_{gene}_row{idx}.png')
        make_sample_figure(
            sample_name=sample_name,
            nuc_3d=nuc_3d_d,   prot_3d=prot_3d_d,
            attn_2d_nuc=a2d_nuc,  attn_2d_prot=a2d_prot,
            attn_3d_nuc=a3d_nuc,  attn_3d_prot=a3d_prot,  attn_3d_cross=a3d_cross,
            attn_esm2_nuc=ae2_nuc, attn_esm2_prot=ae2_prot, attn_esm2_cross=ae2_cross,
            z_indices=z_indices,
            output_path=out_path,
        )

        # Collect for summary figure
        summary_records.append({
            'name':          gene,
            'nuc_3d':        nuc_3d_d,
            'prot_3d':       prot_3d_d,
            'attn_2d_prot':  a2d_prot,
            'attn_3d_prot':  a3d_prot,
            'attn_esm2_prot': ae2_prot,
        })

        # Save attention grid data for further analysis
        np_out = os.path.join(cmd.output_dir, f'attn_data_{gene}_row{idx}.npz')
        np.savez_compressed(
            np_out,
            prot_3d=prot_3d_d, nuc_3d=nuc_3d_d,
            attn_2d_prot=a2d_prot,
            attn_3d_prot=a3d_prot,  attn_3d_nuc=a3d_nuc,
            attn_esm2_prot=ae2_prot, attn_esm2_nuc=ae2_nuc,
        )
        print(f'  Data  → {np_out}')

    # ── Summary figure ─────────────────────────────────────────────────────────
    if len(summary_records) > 1:
        make_summary_figure(
            summary_records,
            os.path.join(cmd.output_dir, 'attention_summary.png'),
        )

    # Dump config
    meta = {
        'config_2d': cmd.config_2d, 'ckpt_2d': cmd.ckpt_2d, 'model_type_2d': cmd.model_type_2d,
        'config_3d': cmd.config_3d, 'ckpt_3d': cmd.ckpt_3d, 'model_type_3d': cmd.model_type_3d,
        'config_3d_esm2': cmd.config_3d_esm2, 'ckpt_3d_esm2': cmd.ckpt_3d_esm2,
        'model_type_3d_esm2': cmd.model_type_3d_esm2,
        'csv_path': cmd.csv_path, 'indices': [int(i) for i in indices],
        'grid_3d': grid_3d, 'grid_2d': grid_2d, 'attn_layer': cmd.attn_layer,
    }
    with open(os.path.join(cmd.output_dir, 'run_config.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    print(f'\nAll outputs saved to: {cmd.output_dir}')


if __name__ == '__main__':
    main()
