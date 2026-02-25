#!/usr/bin/env python3
"""
Evaluate PPI metric learning models with 5-fold cross-validation.

This script compares MAE2D vs MAE3D for PPI prediction using:
1. 5-fold stratified cross-validation on PPI pairs
2. Statistical significance testing (paired t-test and Wilcoxon signed-rank test)

Usage:
    python src/evaluate_ppi_crossval.py \
        --config_2d configs/opencell/opencell_ppi_2d.yaml \
        --config_3d configs/opencell/opencell_ppi_3d.yaml \
        --checkpoint_2d /path/to/2d_checkpoint.pth.tar \
        --checkpoint_3d /path/to/3d_checkpoint.pth.tar \
        --output_dir /path/to/output \
        --n_folds 5
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
from sklearn.model_selection import StratifiedKFold
from scipy import stats
import matplotlib.pyplot as plt
import json

sys.path.insert(0, str(Path(__file__).parent))

from omegaconf import OmegaConf
from data.opencell.ppi_dataset import OpenCellPPITestDataset
from data.opencell.transforms import (
    get_opencell_val_transforms,
    get_opencell_2d_val_transforms
)
from lib.models.ppi_metric import PPIMetric3D, PPIMetric2D, PPIMetric3DCrossAttention


def load_config(config_path):
    """Load config from yaml file."""
    config = OmegaConf.load(config_path)
    OmegaConf.resolve(config)
    return config


def build_model(config, model_type='3d'):
    """Build PPI metric model from config."""
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
    elif model_type == '2d':
        model_params['input_size'] = model_params['input_size'][1:] if len(model_params['input_size']) == 3 else model_params['input_size']
        model_params['patch_size'] = model_params['patch_size'][1:] if len(model_params['patch_size']) == 3 else model_params['patch_size']
        model = PPIMetric2D(**model_params)
    else:
        model = PPIMetric3D(**model_params)

    return model


def load_checkpoint(model, checkpoint_path, device):
    """Load model weights from checkpoint."""
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    # Remove 'module.' prefix if present
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict, strict=True)
    print(f"Loaded checkpoint (epoch: {checkpoint.get('epoch', 'unknown')})")
    return model


def extract_embeddings(model, dataloader, device):
    """Extract embeddings for all cells."""
    model.eval()
    all_embeddings = []
    all_protein_names = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting embeddings"):
            images = batch['image'].to(device)
            protein_names = batch['protein_name']
            embeddings = model.forward_embedding(images)
            all_embeddings.append(embeddings.cpu().numpy())
            all_protein_names.extend(protein_names)

    embeddings = np.concatenate(all_embeddings, axis=0)
    return embeddings, all_protein_names


def aggregate_protein_embeddings(embeddings, protein_names):
    """Compute mean embedding per protein."""
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
    """Load and filter PPI data."""
    ppi_df = pd.read_csv(ppi_path)
    filtered = ppi_df[
        (ppi_df['pval'] > pval_threshold) &
        (ppi_df['enrichment'] > enrichment_threshold) &
        (ppi_df['interaction_stoichiometry'] > stoichiometry_threshold)
    ].copy()
    return filtered


def load_abundance_data(abundance_path):
    """Load protein abundance data."""
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
    """Assign proteins to abundance buckets."""
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
    """Build positive pairs from filtered PPI data."""
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
    """Build abundance-matched negative pairs."""
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
    """Compute cosine similarity for pairs."""
    similarities = []
    for prot1, prot2 in pairs:
        emb1 = protein_embeddings[prot1]
        emb2 = protein_embeddings[prot2]
        sim = np.dot(emb1, emb2)
        similarities.append(sim)
    return np.array(similarities)


def evaluate_fold(positive_pairs, negative_pairs, protein_embeddings):
    """Evaluate PPI prediction on a fold."""
    positive_similarities = compute_similarities(positive_pairs, protein_embeddings)
    negative_similarities = compute_similarities(negative_pairs, protein_embeddings)

    labels = np.concatenate([
        np.ones(len(positive_similarities)),
        np.zeros(len(negative_similarities))
    ])
    scores = np.concatenate([positive_similarities, negative_similarities])

    roc_auc = roc_auc_score(labels, scores)
    avg_precision = average_precision_score(labels, scores)

    predictions = (scores > 0).astype(float)
    accuracy = (predictions == labels).mean()

    return {
        'roc_auc': roc_auc,
        'average_precision': avg_precision,
        'accuracy': accuracy,
        'n_positive': len(positive_pairs),
        'n_negative': len(negative_pairs),
    }


def run_crossval(positive_pairs, negative_pairs, protein_embeddings_2d,
                 protein_embeddings_3d, n_folds=5, seed=42):
    """Run cross-validation and return per-fold metrics."""

    # Create combined dataset for stratified splitting
    all_pairs = positive_pairs + negative_pairs
    all_labels = [1] * len(positive_pairs) + [0] * len(negative_pairs)

    # Convert to numpy arrays for indexing
    all_pairs = np.array(all_pairs)
    all_labels = np.array(all_labels)

    # Stratified K-Fold
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    results_2d = []
    results_3d = []

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(all_pairs, all_labels)):
        print(f"\n{'='*50}")
        print(f"Fold {fold_idx + 1}/{n_folds}")
        print(f"{'='*50}")

        # Get test pairs
        test_pairs = all_pairs[test_idx]
        test_labels = all_labels[test_idx]

        # Separate positive and negative for this fold
        test_positive = [tuple(p) for p, l in zip(test_pairs, test_labels) if l == 1]
        test_negative = [tuple(p) for p, l in zip(test_pairs, test_labels) if l == 0]

        print(f"Test set: {len(test_positive)} positive, {len(test_negative)} negative pairs")

        # Evaluate 2D model
        metrics_2d = evaluate_fold(test_positive, test_negative, protein_embeddings_2d)
        results_2d.append(metrics_2d)
        print(f"2D - ROC-AUC: {metrics_2d['roc_auc']:.4f}, AP: {metrics_2d['average_precision']:.4f}")

        # Evaluate 3D model
        metrics_3d = evaluate_fold(test_positive, test_negative, protein_embeddings_3d)
        results_3d.append(metrics_3d)
        print(f"3D - ROC-AUC: {metrics_3d['roc_auc']:.4f}, AP: {metrics_3d['average_precision']:.4f}")

    return results_2d, results_3d


def compute_statistics(results_2d, results_3d):
    """Compute statistical significance tests."""

    metrics = ['roc_auc', 'average_precision', 'accuracy']
    stats_results = {}

    for metric in metrics:
        values_2d = np.array([r[metric] for r in results_2d])
        values_3d = np.array([r[metric] for r in results_3d])

        # Mean and std
        mean_2d = np.mean(values_2d)
        std_2d = np.std(values_2d, ddof=1)
        mean_3d = np.mean(values_3d)
        std_3d = np.std(values_3d, ddof=1)

        # Paired t-test (two-sided)
        t_stat, t_pval = stats.ttest_rel(values_3d, values_2d)

        # Wilcoxon signed-rank test (non-parametric)
        # Only use if differences are not all zero
        diff = values_3d - values_2d
        if np.all(diff == 0):
            w_stat, w_pval = np.nan, 1.0
        else:
            try:
                w_stat, w_pval = stats.wilcoxon(values_3d, values_2d, alternative='two-sided')
            except ValueError:
                w_stat, w_pval = np.nan, np.nan

        stats_results[metric] = {
            'mean_2d': mean_2d,
            'std_2d': std_2d,
            'mean_3d': mean_3d,
            'std_3d': std_3d,
            'difference': mean_3d - mean_2d,
            'improvement_pct': ((mean_3d - mean_2d) / mean_2d) * 100 if mean_2d > 0 else 0,
            't_statistic': t_stat,
            't_pvalue': t_pval,
            'wilcoxon_statistic': w_stat,
            'wilcoxon_pvalue': w_pval,
            'values_2d': values_2d.tolist(),
            'values_3d': values_3d.tolist(),
        }

    return stats_results


def print_results(stats_results):
    """Print formatted results."""

    print("\n" + "=" * 80)
    print("CROSS-VALIDATION RESULTS SUMMARY")
    print("=" * 80)

    for metric, results in stats_results.items():
        print(f"\n{metric.upper()}")
        print("-" * 40)
        print(f"  MAE2D: {results['mean_2d']:.4f} +/- {results['std_2d']:.4f}")
        print(f"  MAE3D: {results['mean_3d']:.4f} +/- {results['std_3d']:.4f}")
        print(f"  Difference (3D - 2D): {results['difference']:.4f} ({results['improvement_pct']:.2f}%)")
        print(f"  Paired t-test: t = {results['t_statistic']:.4f}, p = {results['t_pvalue']:.6f}")
        if not np.isnan(results['wilcoxon_pvalue']):
            print(f"  Wilcoxon test: W = {results['wilcoxon_statistic']:.4f}, p = {results['wilcoxon_pvalue']:.6f}")

        # Significance interpretation
        if results['t_pvalue'] < 0.001:
            sig_level = "*** (p < 0.001)"
        elif results['t_pvalue'] < 0.01:
            sig_level = "** (p < 0.01)"
        elif results['t_pvalue'] < 0.05:
            sig_level = "* (p < 0.05)"
        else:
            sig_level = "not significant (p >= 0.05)"

        print(f"  Significance: {sig_level}")


def plot_crossval_results(stats_results, output_path):
    """Plot cross-validation results with error bars."""

    metrics = list(stats_results.keys())
    n_metrics = len(metrics)

    fig, axes = plt.subplots(1, n_metrics + 1, figsize=(4 * (n_metrics + 1), 5))

    # Bar plots for each metric
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        results = stats_results[metric]

        x = [0, 1]
        means = [results['mean_2d'], results['mean_3d']]
        stds = [results['std_2d'], results['std_3d']]
        colors = ['#1f77b4', '#ff7f0e']

        bars = ax.bar(x, means, yerr=stds, capsize=5, color=colors, alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(['MAE2D', 'MAE3D'])
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_title(f'{metric.replace("_", " ").title()}\np = {results["t_pvalue"]:.4f}')
        ax.grid(True, alpha=0.3, axis='y')

        # Add significance stars
        if results['t_pvalue'] < 0.001:
            sig_stars = '***'
        elif results['t_pvalue'] < 0.01:
            sig_stars = '**'
        elif results['t_pvalue'] < 0.05:
            sig_stars = '*'
        else:
            sig_stars = 'ns'

        # Draw significance bar
        max_val = max(means[0] + stds[0], means[1] + stds[1])
        y_line = max_val + 0.02
        ax.plot([0, 1], [y_line, y_line], 'k-', linewidth=1)
        ax.text(0.5, y_line + 0.01, sig_stars, ha='center', fontsize=12)

    # Box plot comparing all folds
    ax = axes[-1]
    all_values_2d = []
    all_values_3d = []
    for metric in metrics:
        all_values_2d.extend(stats_results[metric]['values_2d'])
        all_values_3d.extend(stats_results[metric]['values_3d'])

    # Just plot ROC-AUC fold values as box plot
    roc_values_2d = stats_results['roc_auc']['values_2d']
    roc_values_3d = stats_results['roc_auc']['values_3d']

    bp = ax.boxplot([roc_values_2d, roc_values_3d], labels=['MAE2D', 'MAE3D'])
    ax.scatter([1] * len(roc_values_2d), roc_values_2d, alpha=0.6, color='#1f77b4', s=50, label='MAE2D folds')
    ax.scatter([2] * len(roc_values_3d), roc_values_3d, alpha=0.6, color='#ff7f0e', s=50, label='MAE3D folds')
    ax.set_ylabel('ROC-AUC')
    ax.set_title('ROC-AUC per Fold')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nSaved plot to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate PPI with Cross-Validation')

    parser.add_argument('--config_2d', type=str, required=True)
    parser.add_argument('--config_3d', type=str, required=True)
    parser.add_argument('--checkpoint_2d', type=str, required=True)
    parser.add_argument('--checkpoint_3d', type=str, required=True)

    parser.add_argument('--csv_path', type=str,
                        default='/path/to/datasets/opencell/opencell_dataset/single_cells/metadata/dataset1/')
    parser.add_argument('--ppi_path', type=str,
                        default='/path/to/datasets/opencell/opencell_metadata_raw/protein-protein-interactions/opencell-protein-interactions.csv')
    parser.add_argument('--abundance_path', type=str,
                        default='/path/to/datasets/opencell/opencell_metadata_raw/protein-abundance/opencell-protein-abundance.csv')

    parser.add_argument('--pval_threshold', type=float, default=5.0)
    parser.add_argument('--enrichment_threshold', type=float, default=2.5)
    parser.add_argument('--stoichiometry_threshold', type=float, default=0.05)
    parser.add_argument('--n_abundance_buckets', type=int, default=10)
    parser.add_argument('--n_negatives_per_positive', type=int, default=1)

    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--n_folds', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # ========================================
    # Load 2D Model and Extract Embeddings
    # ========================================
    print("\n" + "=" * 60)
    print("Loading MAE2D Model")
    print("=" * 60)

    config_2d = load_config(args.config_2d)
    model_2d = build_model(config_2d, '2d')
    model_2d = load_checkpoint(model_2d, args.checkpoint_2d, device)
    model_2d = model_2d.to(device)
    model_2d.eval()

    transform_2d = get_opencell_2d_val_transforms()
    csv_path = os.path.join(args.csv_path, f'{args.split}.csv')

    dataset_2d = OpenCellPPITestDataset(
        csv_path=csv_path,
        transform=transform_2d,
        use_max_projection=True
    )

    dataloader_2d = DataLoader(
        dataset_2d,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    print(f"Extracting 2D embeddings from {len(dataset_2d)} cells...")
    embeddings_2d, protein_names_2d = extract_embeddings(model_2d, dataloader_2d, device)
    protein_embeddings_2d = aggregate_protein_embeddings(embeddings_2d, protein_names_2d)
    print(f"2D: {len(protein_embeddings_2d)} protein embeddings")

    # Free GPU memory
    del model_2d
    torch.cuda.empty_cache()

    # ========================================
    # Load 3D Model and Extract Embeddings
    # ========================================
    print("\n" + "=" * 60)
    print("Loading MAE3D Model")
    print("=" * 60)

    config_3d = load_config(args.config_3d)
    model_3d = build_model(config_3d, '3d')
    model_3d = load_checkpoint(model_3d, args.checkpoint_3d, device)
    model_3d = model_3d.to(device)
    model_3d.eval()

    transform_3d = get_opencell_val_transforms()

    dataset_3d = OpenCellPPITestDataset(
        csv_path=csv_path,
        transform=transform_3d,
        use_max_projection=False
    )

    dataloader_3d = DataLoader(
        dataset_3d,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    print(f"Extracting 3D embeddings from {len(dataset_3d)} cells...")
    embeddings_3d, protein_names_3d = extract_embeddings(model_3d, dataloader_3d, device)
    protein_embeddings_3d = aggregate_protein_embeddings(embeddings_3d, protein_names_3d)
    print(f"3D: {len(protein_embeddings_3d)} protein embeddings")

    # Free GPU memory
    del model_3d
    torch.cuda.empty_cache()

    # ========================================
    # Load PPI Data and Build Pairs
    # ========================================
    print("\n" + "=" * 60)
    print("Loading PPI Data")
    print("=" * 60)

    ppi_df = load_ppi_data(
        args.ppi_path,
        pval_threshold=args.pval_threshold,
        enrichment_threshold=args.enrichment_threshold,
        stoichiometry_threshold=args.stoichiometry_threshold
    )
    print(f"Filtered PPI: {len(ppi_df)} interactions")

    abundance_dict = load_abundance_data(args.abundance_path)

    # Use proteins available in both 2D and 3D
    available_proteins = list(set(protein_embeddings_2d.keys()) & set(protein_embeddings_3d.keys()))
    print(f"Common proteins: {len(available_proteins)}")

    bucket_assignments, bucket_proteins = assign_abundance_buckets(
        available_proteins, abundance_dict, n_buckets=args.n_abundance_buckets
    )

    positive_pairs = build_positive_pairs(ppi_df, available_proteins)
    print(f"Positive pairs: {len(positive_pairs)}")

    negative_pairs = build_negative_pairs(
        positive_pairs, bucket_assignments, bucket_proteins,
        n_negatives_per_positive=args.n_negatives_per_positive,
        seed=args.seed
    )
    print(f"Negative pairs: {len(negative_pairs)}")

    # ========================================
    # Run Cross-Validation
    # ========================================
    print("\n" + "=" * 60)
    print(f"Running {args.n_folds}-Fold Cross-Validation")
    print("=" * 60)

    results_2d, results_3d = run_crossval(
        positive_pairs, negative_pairs,
        protein_embeddings_2d, protein_embeddings_3d,
        n_folds=args.n_folds, seed=args.seed
    )

    # ========================================
    # Compute Statistics
    # ========================================
    stats_results = compute_statistics(results_2d, results_3d)
    print_results(stats_results)

    # ========================================
    # Save Results
    # ========================================

    # Save detailed results
    output = {
        'n_folds': args.n_folds,
        'seed': args.seed,
        'n_positive_pairs': len(positive_pairs),
        'n_negative_pairs': len(negative_pairs),
        'n_proteins': len(available_proteins),
        'statistics': {},
        'fold_results_2d': results_2d,
        'fold_results_3d': results_3d,
    }

    for metric, results in stats_results.items():
        output['statistics'][metric] = {
            'mean_2d': results['mean_2d'],
            'std_2d': results['std_2d'],
            'mean_3d': results['mean_3d'],
            'std_3d': results['std_3d'],
            'difference': results['difference'],
            'improvement_pct': results['improvement_pct'],
            't_statistic': float(results['t_statistic']),
            't_pvalue': float(results['t_pvalue']),
            'wilcoxon_pvalue': float(results['wilcoxon_pvalue']) if not np.isnan(results['wilcoxon_pvalue']) else None,
        }

    output_path = os.path.join(args.output_dir, 'crossval_results.json')
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved results to {output_path}")

    # Plot results
    plot_path = os.path.join(args.output_dir, 'crossval_comparison.png')
    plot_crossval_results(stats_results, plot_path)

    print("\n" + "=" * 60)
    print("Cross-Validation Complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
