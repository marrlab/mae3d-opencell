#!/usr/bin/env python3
"""
MICCAI-quality PPI comparison figures.

Reads outputs saved by evaluate_ppi_bootstrap.py and generates
two publication-quality figures:

  violin_bootstrap_miccai.{png,pdf}
      Bootstrap distributions of ROC-AUC and Average Precision,
      MAE-2D vs MAE-3D.  Violin shape + subsampled dots.
      Bootstrap p-value significance brackets.

  violin_similarities_miccai.{png,pdf}
      Per-protein-pair cosine similarities (positive / negative sets),
      MAE-2D vs MAE-3D.  Violin shape + every individual data point as dot.
      Wilcoxon signed-rank p-value significance brackets.

The per-pair similarities are recomputed on-the-fly from the cached
protein embeddings (embeddings_cache.npz) and the same PPI filtering
that was used during evaluation, so no re-extraction is needed.

Usage
-----
    python src/plot_ppi_miccai.py \\
        --results_dir /path/to/bootstrap_comparison \\
        [--ppi_path   /path/to/opencell-protein-interactions.csv] \\
        [--abundance_path /path/to/opencell-protein-abundance.csv] \\
        [--n_dot_subsample 200] \\
        [--output_dir /path/to/output]   # default: same as results_dir
"""

import argparse
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── MICCAI style constants ──────────────────────────────────────────────────
# LNCS text width ≈ 12.2 cm = 4.8"; a 2-panel figure at 5" fits comfortably.
FIG_W   = 5.0   # inches (full text-column width)
FIG_H   = 2.9   # inches per figure

FS_TITLE  = 8
FS_LABEL  = 8
FS_TICK   = 7
FS_ANNOT  = 7

C_2D = '#a0a0a0'   # light grey — MAE-2D
C_3D = '#303030'   # dark grey  — MAE-3D (ours / best model)

ALPHA_V = 0.55     # violin body
ALPHA_D = 0.65     # dot

# ── Significance helpers ─────────────────────────────────────────────────────

def _stars(p):
    if p < 0.001: return '***'
    if p < 0.01:  return '**'
    if p < 0.05:  return '*'
    return 'ns'


def _bracket(ax, x0, x1, y_bot, height, label, fs=FS_ANNOT):
    """Draw a significance bracket between x0 and x1 at y_bot."""
    ax.plot([x0, x0, x1, x1],
            [y_bot, y_bot + height, y_bot + height, y_bot],
            lw=0.9, c='black', solid_capstyle='round')
    ax.text((x0 + x1) / 2, y_bot + height * 1.15, label,
            ha='center', va='bottom', fontsize=fs, fontweight='bold')


# ── Core violin-with-dots primitive ─────────────────────────────────────────

MARKERS = ['o', 's']   # index 0 → 2D (circle), index 1 → 3D (square)


def _violin_dots(ax, datasets, positions, colors,
                 dot_n=None, dot_size=8, jitter=0.10):
    """
    Draw a clean violin (shape only) with jittered dots.

    * showmeans / showmedians / showextrema all False  →  shape only.
    * dots are the sole data summary shown inside.
    * First dataset uses circles (2D), second uses squares (3D).
    * dot_n: if set, randomly subsample each dataset to at most dot_n points.
    """
    rng = np.random.default_rng(42)

    # violin shape
    parts = ax.violinplot(
        datasets, positions=positions,
        showmeans=False, showmedians=False, showextrema=False,
        widths=0.52,
    )
    for pc, c in zip(parts['bodies'], colors):
        pc.set_facecolor(c)
        pc.set_edgecolor('none')
        pc.set_alpha(ALPHA_V)

    # dots — circle for 2D (index 0), square for 3D (index 1)
    for i, (data, pos, c) in enumerate(zip(datasets, positions, colors)):
        pts = np.asarray(data)
        if dot_n is not None and len(pts) > dot_n:
            pts = pts[rng.choice(len(pts), dot_n, replace=False)]
        xs = pos + rng.uniform(-jitter, jitter, len(pts))
        ax.scatter(xs, pts, s=dot_size, color=c, alpha=ALPHA_D,
                   marker=MARKERS[i % len(MARKERS)],
                   linewidths=0, zorder=3)


# ── Figure 1: Bootstrap metric distributions ─────────────────────────────────

def plot_bootstrap_violin(bootstrap_results, bootstrap_npz,
                          output_dir, n_dot_subsample=200):
    """
    2-panel figure: ROC-AUC | Average Precision.
    Violin = bootstrap distribution.  Dots = subsampled bootstrap samples.
    Horizontal tick = observed (non-bootstrap) value.
    """
    boots = np.load(bootstrap_npz)

    metrics = [
        ('roc_auc',           boots['roc_auc_2d'], boots['roc_auc_3d'],
         'ROC-AUC'),
        ('average_precision', boots['ap_2d'],       boots['ap_3d'],
         'Avg. Precision (AP)'),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(FIG_W, FIG_H),
                             constrained_layout=True, facecolor='white')

    for ax, (key, b2d, b3d, ylabel) in zip(axes, metrics):
        _violin_dots(ax, [b2d, b3d], [1, 2], [C_2D, C_3D],
                     dot_n=n_dot_subsample, dot_size=7)

        # Observed values as short horizontal ticks
        obs_2d = bootstrap_results[key]['mae2d']['value']
        obs_3d = bootstrap_results[key]['mae3d']['value']
        hw = 0.18   # half-width of tick
        ax.hlines([obs_2d, obs_3d],
                  [1 - hw, 2 - hw], [1 + hw, 2 + hw],
                  colors='black', lw=1.6, zorder=5)

        # Significance bracket (stars only)
        pval  = bootstrap_results[key]['difference']['pvalue']
        label = _stars(pval)
        y_top = max(b2d.max(), b3d.max())
        span  = max(b2d.max() - b2d.min(), b3d.max() - b3d.min())
        _bracket(ax, 1, 2, y_top + span * 0.03, span * 0.05, label)

        ax.set_xticks([1, 2])
        ax.set_xticklabels(['2D', '3D'], fontsize=FS_TICK)
        ax.set_ylabel(ylabel, fontsize=FS_LABEL)
        ax.tick_params(labelsize=FS_TICK)
        ax.spines[['top', 'right']].set_visible(False)
        ax.set_xlim(0.45, 2.55)
        ax.grid(axis='y', lw=0.4, alpha=0.35, color='gray')

    # Legend patches (shared)
    import matplotlib.lines as mlines
    h2 = mlines.Line2D([], [], color=C_2D, marker='o', linestyle='None',
                       markersize=4, alpha=0.85, label='2D')
    h3 = mlines.Line2D([], [], color=C_3D, marker='s', linestyle='None',
                       markersize=4, alpha=0.85, label='3D')
    axes[0].legend(handles=[h2, h3], fontsize=FS_ANNOT,
                   framealpha=0.85, loc='lower right',
                   handlelength=1.0, handletextpad=0.5)

    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(output_dir, f'violin_bootstrap_miccai.{ext}'),
                    dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('  → violin_bootstrap_miccai.{png,pdf}')


# ── Figure 2: Per-pair cosine similarity distributions ───────────────────────

def plot_similarity_violin(pos_sims_2d, pos_sims_3d,
                           neg_sims_2d, neg_sims_3d,
                           wilcoxon_results, output_dir):
    """
    2-panel figure: Positive pairs | Negative pairs.
    Violin = similarity distribution.
    Dots = every individual protein pair (n ≤ ~200 typically).
    """
    n_pos = len(pos_sims_2d)
    n_neg = len(neg_sims_2d)

    panels = [
        (pos_sims_2d, pos_sims_3d,
         f'Positive pairs  (n\u2009=\u2009{n_pos})',
         wilcoxon_results['positive_pairs']['pvalue_twosided']),
        (neg_sims_2d, neg_sims_3d,
         f'Negative pairs  (n\u2009=\u2009{n_neg})',
         wilcoxon_results['negative_pairs']['pvalue_twosided']),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(FIG_W, FIG_H),
                             constrained_layout=True, facecolor='white')

    for ax, (d2d, d3d, title, pval) in zip(axes, panels):
        # Show all data points (n ≤ 106 here — no subsampling needed)
        _violin_dots(ax, [d2d, d3d], [1, 2], [C_2D, C_3D],
                     dot_n=None, dot_size=12, jitter=0.13)

        # Median horizontal ticks
        hw = 0.18
        ax.hlines([np.median(d2d), np.median(d3d)],
                  [1 - hw, 2 - hw], [1 + hw, 2 + hw],
                  colors='black', lw=1.8, zorder=5,
                  label='Median')

        # Significance bracket (stars only)
        label = _stars(pval)
        y_top = max(d2d.max(), d3d.max())
        span  = max(d2d.max() - d2d.min(), d3d.max() - d3d.min())
        _bracket(ax, 1, 2, y_top + span * 0.04, span * 0.06, label)

        ax.set_xticks([1, 2])
        ax.set_xticklabels(['MAE-2D', 'MAE-3D'], fontsize=FS_TICK)
        ax.set_ylabel('Cosine similarity', fontsize=FS_LABEL)
        ax.tick_params(labelsize=FS_TICK)
        ax.spines[['top', 'right']].set_visible(False)
        ax.set_xlim(0.45, 2.55)
        ax.grid(axis='y', lw=0.4, alpha=0.35, color='gray')

    import matplotlib.lines as mlines
    h2 = mlines.Line2D([], [], color=C_2D, marker='o', linestyle='None',
                       markersize=4, alpha=0.85, label='2D')
    h3 = mlines.Line2D([], [], color=C_3D, marker='s', linestyle='None',
                       markersize=4, alpha=0.85, label='3D')
    hm = mlines.Line2D([], [], color='black', lw=1.8, label='Median')
    axes[0].legend(handles=[h2, h3, hm], fontsize=FS_ANNOT,
                   framealpha=0.85, loc='lower right',
                   handlelength=1.0, handletextpad=0.5)

    for ext in ('png', 'pdf'):
        fig.savefig(
            os.path.join(output_dir, f'violin_similarities_miccai.{ext}'),
            dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('  → violin_similarities_miccai.{png,pdf}')


# ── Pair building (mirrors evaluate_ppi_bootstrap.py, no model loading) ──────

def _load_ppi(ppi_path, pval_thr=5.0, enr_thr=2.5, stoi_thr=0.05):
    df = pd.read_csv(ppi_path)
    return df[(df['pval'] > pval_thr) &
              (df['enrichment'] > enr_thr) &
              (df['interaction_stoichiometry'] > stoi_thr)].copy()


def _load_abundance(abundance_path):
    df = pd.read_csv(abundance_path)
    d = {}
    for _, row in df.iterrows():
        g = row['gene_name']
        if pd.notna(row.get('hek_protein_conc_nm')):
            d[g] = row['hek_protein_conc_nm']
        elif pd.notna(row.get('hek_rna_tpm')):
            d[g] = row['hek_rna_tpm']
    return d


def _abundance_buckets(proteins, abundance, n=10):
    with_a = [(p, abundance[p]) for p in proteins if p in abundance]
    with_a.sort(key=lambda x: x[1])
    bp = defaultdict(list)
    ba = {}
    bsz = len(with_a) / n if with_a else 1
    for i, (p, _) in enumerate(with_a):
        b = min(int(i / bsz), n - 1)
        ba[p] = b; bp[b].append(p)
    for p in proteins:
        if p not in ba:
            ba[p] = -1; bp[-1].append(p)
    return ba, dict(bp)


def _positive_pairs(ppi_df, available):
    avail = set(available)
    pairs = set()
    for _, row in ppi_df.iterrows():
        t, i = row['target_gene_name'], row['interactor_gene_name']
        if t in avail and i in avail:
            pairs.add(tuple(sorted([t, i])))
    return list(pairs)


def _negative_pairs(pos_pairs, bucket_a, bucket_p,
                    n_per_pos=1, seed=42):
    rng = np.random.default_rng(seed)
    pos_set = set(pos_pairs)
    negs = []
    all_p = list(bucket_a.keys())
    for p1, p2 in pos_pairs:
        c1 = bucket_p.get(bucket_a.get(p1, -1), all_p)
        c2 = bucket_p.get(bucket_a.get(p2, -1), all_p)
        for _ in range(n_per_pos * 10):
            n1 = rng.choice(c1); n2 = rng.choice(c2)
            if n1 == n2: continue
            np_ = tuple(sorted([n1, n2]))
            if np_ not in pos_set and np_ not in negs:
                negs.append(np_); break
    while len(negs) < len(pos_pairs) * n_per_pos:
        n1, n2 = rng.choice(all_p, 2, replace=False)
        np_ = tuple(sorted([n1, n2]))
        if np_ not in pos_set and np_ not in negs:
            negs.append(np_)
    return negs[:len(pos_pairs) * n_per_pos]


def _cosine_sims(pairs, emb):
    return np.array([np.dot(emb[p1], emb[p2]) for p1, p2 in pairs])


def _rebuild_similarities(results_dir, ppi_path, abundance_path, seed=42):
    """
    Reload cached protein embeddings and rebuild per-pair similarities
    using the same filtering / pairing logic used during evaluation.
    """
    print('  Loading cached protein embeddings ...')
    cache = np.load(os.path.join(results_dir, 'embeddings_cache.npz'),
                    allow_pickle=True)
    emb2d = cache['embeddings_2d'].item()
    emb3d = cache['embeddings_3d'].item()
    available = list(set(emb2d) & set(emb3d))
    print(f'    {len(available)} proteins available')

    print('  Loading PPI data ...')
    ppi_df    = _load_ppi(ppi_path)
    abundance = _load_abundance(abundance_path)
    ba, bp    = _abundance_buckets(available, abundance)

    pos_pairs = _positive_pairs(ppi_df, available)
    neg_pairs = _negative_pairs(pos_pairs, ba, bp, seed=seed)
    print(f'    {len(pos_pairs)} positive pairs, {len(neg_pairs)} negative pairs')

    pos2d = _cosine_sims(pos_pairs, emb2d)
    pos3d = _cosine_sims(pos_pairs, emb3d)
    neg2d = _cosine_sims(neg_pairs, emb2d)
    neg3d = _cosine_sims(neg_pairs, emb3d)
    return pos2d, pos3d, neg2d, neg3d


# ── Main ─────────────────────────────────────────────────────────────────────

_DEFAULT_PPI_PATH = (
    '/ictstr01/groups/labs/marr/qscd01/datasets/SingleCellImagesDataset'
    '/opencell/opencell_metadata_raw/protein-protein-interactions'
    '/opencell-protein-interactions.csv'
)
_DEFAULT_ABUNDANCE_PATH = (
    '/ictstr01/groups/labs/marr/qscd01/datasets/SingleCellImagesDataset'
    '/opencell/opencell_metadata_raw/protein-abundance'
    '/opencell-protein-abundance.csv'
)


def main():
    parser = argparse.ArgumentParser(
        description='MICCAI-quality PPI violin plots from saved bootstrap results')
    parser.add_argument('--results_dir', type=str, required=True,
                        help='Directory with bootstrap_results.json, '
                             'bootstrap_distributions.npz, embeddings_cache.npz')
    parser.add_argument('--ppi_path', type=str, default=_DEFAULT_PPI_PATH)
    parser.add_argument('--abundance_path', type=str, default=_DEFAULT_ABUNDANCE_PATH)
    parser.add_argument('--n_dot_subsample', type=int, default=200,
                        help='Max bootstrap dots shown per violin (default 200)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Seed for negative-pair sampling (must match evaluation)')
    parser.add_argument('--output_dir', type=str, default=None)
    args = parser.parse_args()

    out_dir = args.output_dir or args.results_dir
    os.makedirs(out_dir, exist_ok=True)

    print(f'Results dir : {args.results_dir}')
    print(f'Output dir  : {out_dir}')

    # ── Load JSON summary ──────────────────────────────────────────────────
    with open(os.path.join(args.results_dir, 'bootstrap_results.json')) as f:
        summary = json.load(f)
    bootstrap_results = summary['bootstrap_results']
    wilcoxon_results  = summary['wilcoxon_results']

    # ── Figure 1: bootstrap distributions ─────────────────────────────────
    boots_npz = os.path.join(args.results_dir, 'bootstrap_distributions.npz')
    print('\nFigure 1: bootstrap violin ...')
    plot_bootstrap_violin(bootstrap_results, boots_npz, out_dir,
                          n_dot_subsample=args.n_dot_subsample)

    # ── Figure 2: per-pair similarities ───────────────────────────────────
    print('\nFigure 2: similarity violin ...')
    pos2d, pos3d, neg2d, neg3d = _rebuild_similarities(
        args.results_dir, args.ppi_path, args.abundance_path, seed=args.seed)
    plot_similarity_violin(pos2d, pos3d, neg2d, neg3d, wilcoxon_results, out_dir)

    print(f'\nDone. Figures saved to: {out_dir}')


if __name__ == '__main__':
    main()
