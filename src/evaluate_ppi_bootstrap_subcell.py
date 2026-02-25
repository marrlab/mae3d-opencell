#!/usr/bin/env python3
"""
Evaluate PPI metric learning models with Bootstrap significance testing.
Compares SubCell embeddings vs MAE3D.

Uses validation + test splits for more PPI pairs.
Bootstrap resampling provides reliable confidence intervals and p-values.

Usage:
    python src/evaluate_ppi_bootstrap_subcell.py \
        --config_subcell configs/opencell/opencell_ppi_subcell.yaml \
        --config_3d configs/opencell/opencell_ppi_3d.yaml \
        --checkpoint_subcell /path/to/subcell_checkpoint.pth.tar \
        --checkpoint_3d /path/to/3d_checkpoint.pth.tar \
        --output_dir /path/to/output \
        --n_bootstrap 10000
"""

import argparse
import os
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.stats import wilcoxon, mannwhitneyu
import matplotlib.pyplot as plt
import json

sys.path.insert(0, str(Path(__file__).parent))

from omegaconf import OmegaConf
from data.opencell.ppi_dataset import OpenCellPPITestDataset
from data.opencell.subcell_ppi_dataset import SubCellPPIDataset
from data.opencell.transforms import get_opencell_val_transforms
from lib.models.ppi_metric import PPIMetric3D, PPIMetric3DCrossAttention
from lib.models.ppi_metric_subcell import PPIMetricSubCell


def load_config(config_path):
    config = OmegaConf.load(config_path)
    OmegaConf.resolve(config)
    return config


def build_model_3d(config):
    model_params = {
        'input_size': tuple(config.input_size) if hasattr(config, 'input_size') else (100, 176, 176),
        'patch_size': tuple(config.patch_size) if hasattr(config, 'patch_size') else (10, 8, 8),
        'in_chans': config.in_chans,
        'embed_dim': config.encoder_embed_dim,
        'depth': config.encoder_depth,
        'num_heads': config.encoder_num_heads,
        'drop_path_rate': getattr(config, 'drop_path', 0.0),
        'pos_embed_type': getattr(config, 'pos_embed_type', 'sincos'),
        'use_global_pool': getattr(config, 'use_global_pool', True),
        'proj_hidden_dim': getattr(config, 'proj_hidden_dim', 512),
        'proj_output_dim': getattr(config, 'proj_output_dim', 128),
        'proj_num_layers': getattr(config, 'proj_num_layers', 2),
    }

    arch = getattr(config, 'arch', 'PPIMetric3D')

    if arch == 'PPIMetric3DCrossAttention':
        model_params['cross_attention_type'] = getattr(config, 'cross_attention_type', 'position_wise')
        model_params['pool_mode'] = getattr(config, 'pool_mode', 'concat')
        model = PPIMetric3DCrossAttention(**model_params)
    else:
        model = PPIMetric3D(**model_params)

    return model


def build_model_subcell(config):
    model = PPIMetricSubCell(
        embed_dim=config.embed_dim,
        proj_hidden_dim=getattr(config, 'proj_hidden_dim', 512),
        proj_output_dim=getattr(config, 'proj_output_dim', 128),
        proj_num_layers=getattr(config, 'proj_num_layers', 2),
    )
    return model


def load_checkpoint(model, checkpoint_path, device):
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict, strict=True)
    print(f"Loaded checkpoint (epoch: {checkpoint.get('epoch', 'unknown')})")
    return model


def extract_embeddings_3d(model, dataloader, device):
    """Extract embeddings from MAE3D model."""
    model.eval()
    all_embeddings = []
    all_protein_names = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting 3D embeddings"):
            images = batch['image'].to(device)
            protein_names = batch['protein_name']
            embeddings = model.forward_embedding(images)
            all_embeddings.append(embeddings.cpu().numpy())
            all_protein_names.extend(protein_names)

    embeddings = np.concatenate(all_embeddings, axis=0)
    return embeddings, all_protein_names


def extract_embeddings_subcell(model, embedding_dict, device):
    """Extract projected embeddings from SubCell model."""
    model.eval()
    protein_embeddings = {}

    with torch.no_grad():
        for protein_name, emb in tqdm(embedding_dict.items(), desc="Extracting SubCell embeddings"):
            emb_tensor = emb.unsqueeze(0).to(device)
            projected = model.forward_embedding(emb_tensor)
            protein_embeddings[protein_name] = projected.cpu().numpy().squeeze()

    return protein_embeddings


def aggregate_protein_embeddings(embeddings, protein_names):
    protein_to_embeddings = defaultdict(list)

    for emb, prot in zip(embeddings, protein_names):
        protein_to_embeddings[prot].append(emb)

    protein_embeddings = {}
    for prot, embs in protein_to_embeddings.items():
        mean_emb = np.mean(embs, axis=0)
        mean_emb = mean_emb / (np.linalg.norm(mean_emb) + 1e-8)
        protein_embeddings[prot] = mean_emb

    return protein_embeddings


def load_ppi_data(ppi_path, pval_threshold=5, enrichment_threshold=2.5,
                  stoichiometry_threshold=0.05):
    ppi_df = pd.read_csv(ppi_path)
    filtered = ppi_df[
        (ppi_df['pval'] > pval_threshold) &
        (ppi_df['enrichment'] > enrichment_threshold) &
        (ppi_df['interaction_stoichiometry'] > stoichiometry_threshold)
    ].copy()
    return filtered


def load_abundance_data(abundance_path):
    abundance_df = pd.read_csv(abundance_path)
    abundance_dict = {}

    for _, row in abundance_df.iterrows():
        gene = row['gene_name']
        if pd.notna(row.get('hek_protein_conc_nm')):
            abundance_dict[gene] = row['hek_protein_conc_nm']
        elif pd.notna(row.get('hek_rna_tpm')):
            abundance_dict[gene] = row['hek_rna_tpm']

    return abundance_dict


def assign_abundance_buckets(proteins, abundance_dict, n_buckets=10):
    protein_abundance = []
    for prot in proteins:
        if prot in abundance_dict:
            protein_abundance.append((prot, abundance_dict[prot]))
        else:
            protein_abundance.append((prot, None))

    with_abundance = [(p, a) for p, a in protein_abundance if a is not None]
    without_abundance = [p for p, a in protein_abundance if a is None]

    bucket_assignments = {}
    bucket_proteins = defaultdict(list)

    if with_abundance:
        with_abundance.sort(key=lambda x: x[1])
        bucket_size = len(with_abundance) / n_buckets

        for i, (prot, _) in enumerate(with_abundance):
            bucket_id = min(int(i / bucket_size), n_buckets - 1)
            bucket_assignments[prot] = bucket_id
            bucket_proteins[bucket_id].append(prot)

    for prot in without_abundance:
        bucket_assignments[prot] = -1
        bucket_proteins[-1].append(prot)

    return bucket_assignments, dict(bucket_proteins)


def build_positive_pairs(ppi_df, available_proteins):
    available_set = set(available_proteins)
    positive_pairs = []

    for _, row in ppi_df.iterrows():
        target = row['target_gene_name']
        interactor = row['interactor_gene_name']

        if target in available_set and interactor in available_set:
            pair = tuple(sorted([target, interactor]))
            positive_pairs.append(pair)

    positive_pairs = list(set(positive_pairs))
    return positive_pairs


def build_negative_pairs(positive_pairs, bucket_assignments, bucket_proteins,
                         n_negatives_per_positive=1, seed=42):
    np.random.seed(seed)

    positive_set = set(positive_pairs)
    negative_pairs = []
    all_proteins = list(bucket_assignments.keys())

    for prot1, prot2 in positive_pairs:
        bucket1 = bucket_assignments.get(prot1, -1)
        bucket2 = bucket_assignments.get(prot2, -1)

        candidates1 = bucket_proteins.get(bucket1, all_proteins)
        candidates2 = bucket_proteins.get(bucket2, all_proteins)

        for _ in range(n_negatives_per_positive * 10):
            neg1 = np.random.choice(candidates1)
            neg2 = np.random.choice(candidates2)

            if neg1 == neg2:
                continue

            neg_pair = tuple(sorted([neg1, neg2]))

            if neg_pair not in positive_set and neg_pair not in negative_pairs:
                negative_pairs.append(neg_pair)
                break

    while len(negative_pairs) < len(positive_pairs) * n_negatives_per_positive:
        neg1, neg2 = np.random.choice(all_proteins, 2, replace=False)
        neg_pair = tuple(sorted([neg1, neg2]))
        if neg_pair not in positive_set and neg_pair not in negative_pairs:
            negative_pairs.append(neg_pair)

    negative_pairs = negative_pairs[:len(positive_pairs) * n_negatives_per_positive]
    return negative_pairs


def compute_similarities(pairs, protein_embeddings):
    similarities = []
    for prot1, prot2 in pairs:
        emb1 = protein_embeddings[prot1]
        emb2 = protein_embeddings[prot2]
        sim = np.dot(emb1, emb2)
        similarities.append(sim)
    return np.array(similarities)


def compute_metrics(positive_sims, negative_sims):
    labels = np.concatenate([np.ones(len(positive_sims)), np.zeros(len(negative_sims))])
    scores = np.concatenate([positive_sims, negative_sims])
    roc_auc = roc_auc_score(labels, scores)
    avg_precision = average_precision_score(labels, scores)
    return roc_auc, avg_precision


def bootstrap_metrics(positive_sims_subcell, negative_sims_subcell,
                      positive_sims_3d, negative_sims_3d,
                      n_bootstrap=10000, seed=42):
    """Compute bootstrap confidence intervals and paired difference test."""
    np.random.seed(seed)

    n_pos = len(positive_sims_subcell)
    n_neg = len(negative_sims_subcell)

    auc_subcell_boots = []
    auc_3d_boots = []
    ap_subcell_boots = []
    ap_3d_boots = []
    auc_diff_boots = []
    ap_diff_boots = []

    for _ in tqdm(range(n_bootstrap), desc="Bootstrap"):
        pos_idx = np.random.choice(n_pos, n_pos, replace=True)
        neg_idx = np.random.choice(n_neg, n_neg, replace=True)

        pos_subcell = positive_sims_subcell[pos_idx]
        neg_subcell = negative_sims_subcell[neg_idx]
        pos_3d = positive_sims_3d[pos_idx]
        neg_3d = negative_sims_3d[neg_idx]

        auc_subcell, ap_subcell = compute_metrics(pos_subcell, neg_subcell)
        auc_3d, ap_3d = compute_metrics(pos_3d, neg_3d)

        auc_subcell_boots.append(auc_subcell)
        auc_3d_boots.append(auc_3d)
        ap_subcell_boots.append(ap_subcell)
        ap_3d_boots.append(ap_3d)
        auc_diff_boots.append(auc_3d - auc_subcell)
        ap_diff_boots.append(ap_3d - ap_subcell)

    auc_subcell_boots = np.array(auc_subcell_boots)
    auc_3d_boots = np.array(auc_3d_boots)
    ap_subcell_boots = np.array(ap_subcell_boots)
    ap_3d_boots = np.array(ap_3d_boots)
    auc_diff_boots = np.array(auc_diff_boots)
    ap_diff_boots = np.array(ap_diff_boots)

    # Original metrics
    auc_subcell_orig, ap_subcell_orig = compute_metrics(positive_sims_subcell, negative_sims_subcell)
    auc_3d_orig, ap_3d_orig = compute_metrics(positive_sims_3d, negative_sims_3d)

    def ci(arr, alpha=0.05):
        return np.percentile(arr, [100 * alpha / 2, 100 * (1 - alpha / 2)])

    # P-value calculation
    auc_diff_orig = auc_3d_orig - auc_subcell_orig
    ap_diff_orig = ap_3d_orig - ap_subcell_orig

    if auc_diff_orig > 0:
        auc_pval_onesided = np.mean(auc_diff_boots <= 0)
    else:
        auc_pval_onesided = np.mean(auc_diff_boots >= 0)
    auc_pval = 2 * min(auc_pval_onesided, 1 - auc_pval_onesided)

    if ap_diff_orig > 0:
        ap_pval_onesided = np.mean(ap_diff_boots <= 0)
    else:
        ap_pval_onesided = np.mean(ap_diff_boots >= 0)
    ap_pval = 2 * min(ap_pval_onesided, 1 - ap_pval_onesided)

    results = {
        'roc_auc': {
            'subcell': {'value': auc_subcell_orig, 'ci_95': ci(auc_subcell_boots).tolist(), 'std': np.std(auc_subcell_boots)},
            'mae3d': {'value': auc_3d_orig, 'ci_95': ci(auc_3d_boots).tolist(), 'std': np.std(auc_3d_boots)},
            'difference': {'value': auc_diff_orig, 'ci_95': ci(auc_diff_boots).tolist(), 'pvalue': auc_pval},
            'bootstrap_values': {'subcell': auc_subcell_boots, 'mae3d': auc_3d_boots, 'diff': auc_diff_boots}
        },
        'average_precision': {
            'subcell': {'value': ap_subcell_orig, 'ci_95': ci(ap_subcell_boots).tolist(), 'std': np.std(ap_subcell_boots)},
            'mae3d': {'value': ap_3d_orig, 'ci_95': ci(ap_3d_boots).tolist(), 'std': np.std(ap_3d_boots)},
            'difference': {'value': ap_diff_orig, 'ci_95': ci(ap_diff_boots).tolist(), 'pvalue': ap_pval},
            'bootstrap_values': {'subcell': ap_subcell_boots, 'mae3d': ap_3d_boots, 'diff': ap_diff_boots}
        }
    }

    return results


def wilcoxon_test(positive_sims_subcell, negative_sims_subcell,
                  positive_sims_3d, negative_sims_3d):
    """Compute Wilcoxon signed-rank test for paired comparisons."""
    results = {}

    # Test on positive pairs
    diff_positive = positive_sims_3d - positive_sims_subcell
    stat_pos, pval_pos_twosided = wilcoxon(positive_sims_3d, positive_sims_subcell, alternative='two-sided')
    _, pval_pos_greater = wilcoxon(positive_sims_3d, positive_sims_subcell, alternative='greater')

    results['positive_pairs'] = {
        'statistic': float(stat_pos),
        'pvalue_twosided': float(pval_pos_twosided),
        'pvalue_3d_greater': float(pval_pos_greater),
        'mean_diff': float(np.mean(diff_positive)),
        'median_diff': float(np.median(diff_positive)),
        'n_3d_higher': int(np.sum(diff_positive > 0)),
        'n_subcell_higher': int(np.sum(diff_positive < 0)),
        'n_equal': int(np.sum(diff_positive == 0)),
    }

    # Test on negative pairs
    diff_negative = negative_sims_3d - negative_sims_subcell
    stat_neg, pval_neg_twosided = wilcoxon(negative_sims_3d, negative_sims_subcell, alternative='two-sided')
    _, pval_neg_less = wilcoxon(negative_sims_3d, negative_sims_subcell, alternative='less')

    results['negative_pairs'] = {
        'statistic': float(stat_neg),
        'pvalue_twosided': float(pval_neg_twosided),
        'pvalue_3d_less': float(pval_neg_less),
        'mean_diff': float(np.mean(diff_negative)),
        'median_diff': float(np.median(diff_negative)),
        'n_3d_higher': int(np.sum(diff_negative > 0)),
        'n_subcell_higher': int(np.sum(diff_negative < 0)),
        'n_equal': int(np.sum(diff_negative == 0)),
    }

    # Discriminability test
    gap_subcell = positive_sims_subcell - np.mean(negative_sims_subcell)
    gap_3d = positive_sims_3d - np.mean(negative_sims_3d)

    stat_gap, pval_gap = wilcoxon(gap_3d, gap_subcell, alternative='greater')

    results['discriminability'] = {
        'description': 'Tests if 3D has better separation between positive and negative pairs',
        'statistic': float(stat_gap),
        'pvalue_3d_better': float(pval_gap),
        'mean_gap_subcell': float(np.mean(gap_subcell)),
        'mean_gap_3d': float(np.mean(gap_3d)),
    }

    # Mann-Whitney U test
    stat_subcell, pval_subcell = mannwhitneyu(positive_sims_subcell, negative_sims_subcell, alternative='greater')
    stat_3d, pval_3d = mannwhitneyu(positive_sims_3d, negative_sims_3d, alternative='greater')

    n1, n2 = len(positive_sims_subcell), len(negative_sims_subcell)
    effect_subcell = 1 - (2 * stat_subcell) / (n1 * n2)
    effect_3d = 1 - (2 * stat_3d) / (n1 * n2)

    results['mann_whitney'] = {
        'description': 'Tests if positive similarities > negative similarities (per model)',
        'subcell': {
            'statistic': float(stat_subcell),
            'pvalue': float(pval_subcell),
            'effect_size_r': float(effect_subcell),
        },
        'mae3d': {
            'statistic': float(stat_3d),
            'pvalue': float(pval_3d),
            'effect_size_r': float(effect_3d),
        }
    }

    return results


def print_results(results, n_positive, n_negative, wilcoxon_results=None):
    print("\n" + "=" * 80)
    print("BOOTSTRAP SIGNIFICANCE TEST RESULTS")
    print("=" * 80)
    print(f"\nDataset: {n_positive} positive pairs, {n_negative} negative pairs")

    for metric_name in ['roc_auc', 'average_precision']:
        metric = results[metric_name]
        print(f"\n{metric_name.upper().replace('_', ' ')}")
        print("-" * 50)

        m_subcell = metric['subcell']
        m3d = metric['mae3d']
        diff = metric['difference']

        print(f"  SubCell: {m_subcell['value']:.4f} (95% CI: [{m_subcell['ci_95'][0]:.4f}, {m_subcell['ci_95'][1]:.4f}])")
        print(f"  MAE3D:   {m3d['value']:.4f} (95% CI: [{m3d['ci_95'][0]:.4f}, {m3d['ci_95'][1]:.4f}])")
        print(f"  Difference (3D - SubCell): {diff['value']:.4f} (95% CI: [{diff['ci_95'][0]:.4f}, {diff['ci_95'][1]:.4f}])")
        print(f"  P-value (bootstrap): {diff['pvalue']:.4f}")

        if diff['pvalue'] < 0.001:
            sig = "*** (p < 0.001)"
        elif diff['pvalue'] < 0.01:
            sig = "** (p < 0.01)"
        elif diff['pvalue'] < 0.05:
            sig = "* (p < 0.05)"
        else:
            sig = "not significant (p >= 0.05)"

        print(f"  Significance: {sig}")

    # Print Wilcoxon results
    if wilcoxon_results:
        print("\n" + "=" * 80)
        print("WILCOXON SIGNED-RANK TEST RESULTS")
        print("=" * 80)

        pos = wilcoxon_results['positive_pairs']
        print(f"\nPOSITIVE PAIRS (Interacting proteins)")
        print("-" * 50)
        print(f"  Mean difference (3D - SubCell): {pos['mean_diff']:.4f}")
        print(f"  P-value (two-sided): {pos['pvalue_twosided']:.4e}")
        print(f"  P-value (3D > SubCell): {pos['pvalue_3d_greater']:.4e}")

        neg = wilcoxon_results['negative_pairs']
        print(f"\nNEGATIVE PAIRS (Non-interacting proteins)")
        print("-" * 50)
        print(f"  Mean difference (3D - SubCell): {neg['mean_diff']:.4f}")
        print(f"  P-value (two-sided): {neg['pvalue_twosided']:.4e}")

        disc = wilcoxon_results['discriminability']
        print(f"\nDISCRIMINABILITY")
        print("-" * 50)
        print(f"  Mean gap SubCell: {disc['mean_gap_subcell']:.4f}")
        print(f"  Mean gap 3D: {disc['mean_gap_3d']:.4f}")
        print(f"  P-value (3D better): {disc['pvalue_3d_better']:.4e}")


def plot_violin(results, output_path):
    """Plot violin plot comparing SubCell and MAE3D bootstrap distributions."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Grayscale colors
    gray_light = '#a0a0a0'
    gray_dark = '#404040'

    for idx, (metric_name, title) in enumerate([('roc_auc', 'ROC AUC'), ('average_precision', 'Average Precision')]):
        ax = axes[idx]
        metric = results[metric_name]
        boots = metric['bootstrap_values']

        parts = ax.violinplot([boots['subcell'], boots['mae3d']], positions=[1, 2],
                              showmeans=True, showmedians=True)

        for pc in parts['bodies']:
            pc.set_facecolor(gray_light)
            pc.set_edgecolor(gray_dark)
            pc.set_alpha(0.7)

        for partname in ['cbars', 'cmins', 'cmaxes', 'cmeans', 'cmedians']:
            if partname in parts:
                parts[partname].set_edgecolor(gray_dark)
                parts[partname].set_linewidth(1.5)

        # Add observed values
        obs_subcell = metric['subcell']['value']
        obs_3d = metric['mae3d']['value']
        ax.scatter([1, 2], [obs_subcell, obs_3d], color='black', s=150, zorder=5,
                   marker='D', label='Observed', edgecolors='white', linewidths=1.5)

        pval = metric['difference']['pvalue']
        sig_str = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else 'ns'

        ax.set_xticks([1, 2])
        ax.set_xticklabels(['SubCell', 'MAE3D'], fontsize=16, fontweight='bold')
        ax.set_ylabel(title, fontsize=16, fontweight='bold')
        ax.set_title(f'{title}\nBootstrap p = {pval:.4f}', fontsize=18, fontweight='bold')
        ax.legend(loc='lower right', fontsize=12)
        ax.grid(axis='y', alpha=0.3, color=gray_light)
        ax.tick_params(axis='both', labelsize=14)

        # Significance bar
        y_max = max(boots['subcell'].max(), boots['mae3d'].max()) + 0.02
        ax.plot([1, 1, 2, 2], [y_max, y_max + 0.01, y_max + 0.01, y_max], 'k-', linewidth=2)
        ax.text(1.5, y_max + 0.015, sig_str, ha='center', fontsize=18, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight')
    plt.close()
    print(f"Saved violin plot to {output_path}")


def plot_similarity_differences(positive_sims_subcell, positive_sims_3d,
                                negative_sims_subcell, negative_sims_3d,
                                wilcoxon_results, output_path):
    """Plot violin/box plots showing per-pair similarity scores."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    gray_light = '#b0b0b0'
    gray_dark = '#303030'
    gray_medium = '#606060'

    # Positive pairs
    ax = axes[0]
    parts = ax.violinplot([positive_sims_subcell, positive_sims_3d], positions=[1, 2],
                          showmeans=False, showmedians=False, widths=0.7)

    for pc in parts['bodies']:
        pc.set_facecolor(gray_light)
        pc.set_edgecolor(gray_dark)
        pc.set_alpha(0.5)

    np.random.seed(42)
    n_points = len(positive_sims_subcell)
    jitter_subcell = 1 + np.random.uniform(-0.2, 0.2, n_points)
    jitter_3d = 2 + np.random.uniform(-0.2, 0.2, n_points)

    ax.scatter(jitter_subcell, positive_sims_subcell, color=gray_dark, s=60, zorder=4,
               alpha=0.8, marker='o', edgecolors='white', linewidths=0.5)
    ax.scatter(jitter_3d, positive_sims_3d, color=gray_dark, s=60, zorder=4,
               alpha=0.8, marker='o', edgecolors='white', linewidths=0.5)

    for i in range(n_points):
        ax.plot([jitter_subcell[i], jitter_3d[i]], [positive_sims_subcell[i], positive_sims_3d[i]],
                color=gray_medium, alpha=0.3, linewidth=0.8, zorder=3)

    mean_subcell = np.mean(positive_sims_subcell)
    mean_3d = np.mean(positive_sims_3d)
    ax.scatter([1, 2], [mean_subcell, mean_3d], color='black', s=200, zorder=5, marker='_', linewidths=3)

    pval = wilcoxon_results['positive_pairs']['pvalue_twosided']
    sig_str = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else 'ns'

    ax.set_xticks([1, 2])
    ax.set_xticklabels(['SubCell', 'MAE3D'], fontsize=18, fontweight='bold')
    ax.set_ylabel('Cosine Similarity', fontsize=18, fontweight='bold')
    ax.set_title(f'Positive Pairs (n={n_points})\nWilcoxon p = {pval:.4e}', fontsize=20, fontweight='bold')
    ax.tick_params(axis='both', labelsize=16)
    ax.grid(axis='y', alpha=0.3, color=gray_light)

    y_max = max(positive_sims_subcell.max(), positive_sims_3d.max()) + 0.05
    ax.plot([1, 1, 2, 2], [y_max, y_max + 0.02, y_max + 0.02, y_max], 'k-', linewidth=2)
    ax.text(1.5, y_max + 0.03, sig_str, ha='center', fontsize=20, fontweight='bold')

    # Negative pairs
    ax = axes[1]
    parts = ax.violinplot([negative_sims_subcell, negative_sims_3d], positions=[1, 2],
                          showmeans=False, showmedians=False, widths=0.7)

    for pc in parts['bodies']:
        pc.set_facecolor(gray_light)
        pc.set_edgecolor(gray_dark)
        pc.set_alpha(0.5)

    n_neg = len(negative_sims_subcell)
    jitter_subcell_neg = 1 + np.random.uniform(-0.2, 0.2, n_neg)
    jitter_3d_neg = 2 + np.random.uniform(-0.2, 0.2, n_neg)

    ax.scatter(jitter_subcell_neg, negative_sims_subcell, color=gray_dark, s=60, zorder=4,
               alpha=0.8, marker='o', edgecolors='white', linewidths=0.5)
    ax.scatter(jitter_3d_neg, negative_sims_3d, color=gray_dark, s=60, zorder=4,
               alpha=0.8, marker='o', edgecolors='white', linewidths=0.5)

    for i in range(n_neg):
        ax.plot([jitter_subcell_neg[i], jitter_3d_neg[i]], [negative_sims_subcell[i], negative_sims_3d[i]],
                color=gray_medium, alpha=0.3, linewidth=0.8, zorder=3)

    mean_subcell_neg = np.mean(negative_sims_subcell)
    mean_3d_neg = np.mean(negative_sims_3d)
    ax.scatter([1, 2], [mean_subcell_neg, mean_3d_neg], color='black', s=200, zorder=5, marker='_', linewidths=3)

    pval_neg = wilcoxon_results['negative_pairs']['pvalue_twosided']
    sig_str_neg = '***' if pval_neg < 0.001 else '**' if pval_neg < 0.01 else '*' if pval_neg < 0.05 else 'ns'

    ax.set_xticks([1, 2])
    ax.set_xticklabels(['SubCell', 'MAE3D'], fontsize=18, fontweight='bold')
    ax.set_ylabel('Cosine Similarity', fontsize=18, fontweight='bold')
    ax.set_title(f'Negative Pairs (n={n_neg})\nWilcoxon p = {pval_neg:.4e}', fontsize=20, fontweight='bold')
    ax.tick_params(axis='both', labelsize=16)
    ax.grid(axis='y', alpha=0.3, color=gray_light)

    y_max_neg = max(negative_sims_subcell.max(), negative_sims_3d.max()) + 0.05
    ax.plot([1, 1, 2, 2], [y_max_neg, y_max_neg + 0.02, y_max_neg + 0.02, y_max_neg], 'k-', linewidth=2)
    ax.text(1.5, y_max_neg + 0.03, sig_str_neg, ha='center', fontsize=20, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight')
    plt.close()
    print(f"Saved similarity differences plot to {output_path}")


def plot_summary(results, output_path):
    """Plot summary bar chart with error bars."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    gray_light = '#a0a0a0'
    gray_dark = '#404040'

    for idx, (metric_name, title) in enumerate([('roc_auc', 'ROC-AUC'), ('average_precision', 'Average Precision')]):
        ax = axes[idx]
        metric = results[metric_name]

        val_subcell = metric['subcell']['value']
        val_3d = metric['mae3d']['value']
        ci_subcell = metric['subcell']['ci_95']
        ci_3d = metric['mae3d']['ci_95']

        yerr = np.array([
            [val_subcell - ci_subcell[0], val_3d - ci_3d[0]],
            [ci_subcell[1] - val_subcell, ci_3d[1] - val_3d]
        ])

        bars = ax.bar([0, 1], [val_subcell, val_3d], yerr=yerr,
                      capsize=10, color=[gray_light, gray_dark], alpha=0.8,
                      edgecolor='black', linewidth=1.5,
                      error_kw={'linewidth': 2, 'capthick': 2})

        ax.set_xticks([0, 1])
        ax.set_xticklabels(['SubCell', 'MAE3D'], fontsize=18, fontweight='bold')
        ax.set_ylabel(title, fontsize=18, fontweight='bold')
        ax.tick_params(axis='both', labelsize=16)

        pval = metric['difference']['pvalue']
        sig_str = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else 'ns'

        ax.set_title(f'{title}\np = {pval:.4f} ({sig_str})', fontsize=20, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y', color=gray_light)

        max_y = max(ci_subcell[1], ci_3d[1])
        y_line = max_y + 0.03
        ax.plot([0, 1], [y_line, y_line], 'k-', linewidth=2)
        ax.text(0.5, y_line + 0.01, sig_str, ha='center', fontsize=20, fontweight='bold')
        ax.set_ylim([0, min(1.0, y_line + 0.1)])

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight')
    plt.close()
    print(f"Saved summary to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate PPI: SubCell vs MAE3D with Bootstrap')

    parser.add_argument('--config_subcell', type=str, required=True)
    parser.add_argument('--config_3d', type=str, required=True)
    parser.add_argument('--checkpoint_subcell', type=str, required=True)
    parser.add_argument('--checkpoint_3d', type=str, required=True)

    parser.add_argument('--csv_path', type=str,
                        default='/ictstr01/groups/labs/marr/qscd01/datasets/SingleCellImagesDataset/opencell/opencell_dataset/single_cells/metadata/dataset1/')
    parser.add_argument('--embedding_dir', type=str,
                        default='/ictstr01/groups/labs/marr/qscd01/datasets/SingleCellImagesDataset/opencell/subcell_pca384')
    parser.add_argument('--ppi_path', type=str,
                        default='/ictstr01/groups/labs/marr/qscd01/datasets/SingleCellImagesDataset/opencell/opencell_metadata_raw/protein-protein-interactions/opencell-protein-interactions.csv')
    parser.add_argument('--abundance_path', type=str,
                        default='/ictstr01/groups/labs/marr/qscd01/datasets/SingleCellImagesDataset/opencell/opencell_metadata_raw/protein-abundance/opencell-protein-abundance.csv')

    parser.add_argument('--pval_threshold', type=float, default=5.0)
    parser.add_argument('--enrichment_threshold', type=float, default=2.5)
    parser.add_argument('--stoichiometry_threshold', type=float, default=0.05)
    parser.add_argument('--n_abundance_buckets', type=int, default=10)
    parser.add_argument('--n_negatives_per_positive', type=int, default=1)

    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--splits', type=str, nargs='+', default=['val', 'test'])
    parser.add_argument('--n_bootstrap', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cache_embeddings', action='store_true', default=True)
    parser.add_argument('--force_recompute', action='store_true', default=False)

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Using splits: {args.splits}")

    # ========================================
    # Load or Extract Embeddings
    # ========================================
    embeddings_cache_path = os.path.join(args.output_dir, 'embeddings_cache.npz')

    if args.cache_embeddings and os.path.exists(embeddings_cache_path) and not args.force_recompute:
        print("\n" + "=" * 60)
        print("Loading cached embeddings")
        print("=" * 60)

        cache = np.load(embeddings_cache_path, allow_pickle=True)
        all_protein_embeddings_subcell = cache['embeddings_subcell'].item()
        all_protein_embeddings_3d = cache['embeddings_3d'].item()

        print(f"Loaded: SubCell={len(all_protein_embeddings_subcell)}, MAE3D={len(all_protein_embeddings_3d)}")
    else:
        print("\n" + "=" * 60)
        print("Extracting embeddings")
        print("=" * 60)

        # Load SubCell model
        config_subcell = load_config(args.config_subcell)
        model_subcell = build_model_subcell(config_subcell)
        model_subcell = load_checkpoint(model_subcell, args.checkpoint_subcell, device)
        model_subcell = model_subcell.to(device)
        model_subcell.eval()

        # Load 3D model
        config_3d = load_config(args.config_3d)
        model_3d = build_model_3d(config_3d)
        model_3d = load_checkpoint(model_3d, args.checkpoint_3d, device)
        model_3d = model_3d.to(device)
        model_3d.eval()

        all_protein_embeddings_subcell = {}
        all_protein_embeddings_3d = {}

        for split in args.splits:
            print(f"\n{'=' * 60}")
            print(f"Processing split: {split}")
            print(f"{'=' * 60}")

            # Load SubCell embeddings for this split
            embedding_path = os.path.join(args.embedding_dir, f'{split}.npy')
            csv_path = os.path.join(args.csv_path, f'{split}.csv')

            # Load raw SubCell embeddings
            embeddings_raw = np.load(embedding_path)
            df = pd.read_csv(csv_path)

            if 'file_gene_symbol' in df.columns:
                df['protein_name'] = df['file_gene_symbol']
            elif 'folder_protein' in df.columns:
                df['protein_name'] = df['folder_protein']

            # Build protein to embedding mapping
            protein_to_embeddings = defaultdict(list)
            for idx, row in df.iterrows():
                protein_to_embeddings[row['protein_name']].append(embeddings_raw[idx])

            # Average and normalize per protein
            protein_embeddings_raw = {}
            for prot, embs in protein_to_embeddings.items():
                mean_emb = np.mean(embs, axis=0)
                protein_embeddings_raw[prot] = torch.from_numpy(mean_emb).float()

            # Project through SubCell model
            subcell_projected = extract_embeddings_subcell(model_subcell, protein_embeddings_raw, device)

            # Normalize
            for prot in subcell_projected:
                emb = subcell_projected[prot]
                subcell_projected[prot] = emb / (np.linalg.norm(emb) + 1e-8)

            all_protein_embeddings_subcell.update(subcell_projected)
            print(f"SubCell {split}: {len(subcell_projected)} proteins")

            # Extract 3D embeddings
            transform_3d = get_opencell_val_transforms()
            dataset_3d = OpenCellPPITestDataset(csv_path=csv_path, transform=transform_3d, use_max_projection=False)
            dataloader_3d = DataLoader(dataset_3d, batch_size=args.batch_size, shuffle=False,
                                       num_workers=args.num_workers, pin_memory=True)

            embeddings_3d, protein_names_3d = extract_embeddings_3d(model_3d, dataloader_3d, device)
            protein_emb_3d = aggregate_protein_embeddings(embeddings_3d, protein_names_3d)
            all_protein_embeddings_3d.update(protein_emb_3d)
            print(f"MAE3D {split}: {len(protein_emb_3d)} proteins")

        del model_subcell, model_3d
        torch.cuda.empty_cache()

        # Cache embeddings
        if args.cache_embeddings:
            print(f"\nCaching embeddings to {embeddings_cache_path}")
            np.savez(embeddings_cache_path,
                     embeddings_subcell=all_protein_embeddings_subcell,
                     embeddings_3d=all_protein_embeddings_3d)

    print(f"\nTotal proteins: SubCell={len(all_protein_embeddings_subcell)}, MAE3D={len(all_protein_embeddings_3d)}")

    # ========================================
    # Load PPI Data and Build Pairs
    # ========================================
    print("\n" + "=" * 60)
    print("Loading PPI Data")
    print("=" * 60)

    ppi_df = load_ppi_data(args.ppi_path, pval_threshold=args.pval_threshold,
                           enrichment_threshold=args.enrichment_threshold,
                           stoichiometry_threshold=args.stoichiometry_threshold)
    print(f"Filtered PPI: {len(ppi_df)} interactions")

    abundance_dict = load_abundance_data(args.abundance_path)

    available_proteins = list(set(all_protein_embeddings_subcell.keys()) & set(all_protein_embeddings_3d.keys()))
    print(f"Common proteins: {len(available_proteins)}")

    bucket_assignments, bucket_proteins = assign_abundance_buckets(
        available_proteins, abundance_dict, n_buckets=args.n_abundance_buckets)

    positive_pairs = build_positive_pairs(ppi_df, available_proteins)
    print(f"Positive pairs: {len(positive_pairs)}")

    negative_pairs = build_negative_pairs(positive_pairs, bucket_assignments, bucket_proteins,
                                          n_negatives_per_positive=args.n_negatives_per_positive, seed=args.seed)
    print(f"Negative pairs: {len(negative_pairs)}")

    # ========================================
    # Compute Similarities
    # ========================================
    print("\n" + "=" * 60)
    print("Computing Similarities")
    print("=" * 60)

    positive_sims_subcell = compute_similarities(positive_pairs, all_protein_embeddings_subcell)
    negative_sims_subcell = compute_similarities(negative_pairs, all_protein_embeddings_subcell)
    positive_sims_3d = compute_similarities(positive_pairs, all_protein_embeddings_3d)
    negative_sims_3d = compute_similarities(negative_pairs, all_protein_embeddings_3d)

    print(f"SubCell - Positive mean: {np.mean(positive_sims_subcell):.4f}, Negative mean: {np.mean(negative_sims_subcell):.4f}")
    print(f"MAE3D   - Positive mean: {np.mean(positive_sims_3d):.4f}, Negative mean: {np.mean(negative_sims_3d):.4f}")

    # ========================================
    # Bootstrap Significance Testing
    # ========================================
    print("\n" + "=" * 60)
    print(f"Running Bootstrap ({args.n_bootstrap} iterations)")
    print("=" * 60)

    results = bootstrap_metrics(positive_sims_subcell, negative_sims_subcell,
                                positive_sims_3d, negative_sims_3d,
                                n_bootstrap=args.n_bootstrap, seed=args.seed)

    # ========================================
    # Wilcoxon Signed-Rank Test
    # ========================================
    print("\n" + "=" * 60)
    print("Running Wilcoxon Signed-Rank Test")
    print("=" * 60)

    wilcoxon_results = wilcoxon_test(positive_sims_subcell, negative_sims_subcell,
                                      positive_sims_3d, negative_sims_3d)

    print_results(results, len(positive_pairs), len(negative_pairs), wilcoxon_results)

    # ========================================
    # Save Results
    # ========================================
    results_summary = {
        'n_bootstrap': args.n_bootstrap,
        'seed': args.seed,
        'splits': args.splits,
        'n_positive_pairs': len(positive_pairs),
        'n_negative_pairs': len(negative_pairs),
        'n_proteins': len(available_proteins),
        'bootstrap_results': {
            metric: {
                'subcell': data['subcell'],
                'mae3d': data['mae3d'],
                'difference': data['difference'],
            }
            for metric, data in results.items()
        },
        'wilcoxon_results': wilcoxon_results,
    }

    with open(os.path.join(args.output_dir, 'bootstrap_results.json'), 'w') as f:
        json.dump(results_summary, f, indent=2)

    # Save bootstrap distributions
    np.savez(os.path.join(args.output_dir, 'bootstrap_distributions.npz'),
             roc_auc_subcell=results['roc_auc']['bootstrap_values']['subcell'],
             roc_auc_3d=results['roc_auc']['bootstrap_values']['mae3d'],
             roc_auc_diff=results['roc_auc']['bootstrap_values']['diff'],
             ap_subcell=results['average_precision']['bootstrap_values']['subcell'],
             ap_3d=results['average_precision']['bootstrap_values']['mae3d'],
             ap_diff=results['average_precision']['bootstrap_values']['diff'])

    # Plot results
    plot_violin(results, os.path.join(args.output_dir, 'violin_comparison.png'))
    plot_similarity_differences(positive_sims_subcell, positive_sims_3d,
                                negative_sims_subcell, negative_sims_3d,
                                wilcoxon_results,
                                os.path.join(args.output_dir, 'similarity_differences.png'))
    plot_summary(results, os.path.join(args.output_dir, 'bootstrap_comparison.png'))

    print("\n" + "=" * 60)
    print("Bootstrap Significance Testing Complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
