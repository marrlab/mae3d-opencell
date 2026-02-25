#!/usr/bin/env python3
"""
Evaluate trained PPI metric learning model on test set.

This script:
1. Loads a trained PPI metric model
2. Extracts embeddings for all cells in the test set
3. Aggregates embeddings per protein (mean pooling)
4. Evaluates PPI prediction using cosine similarity on known pairs

Usage:
    python src/evaluate_ppi_metric.py \
        --config configs/opencell/opencell_ppi_3d.yaml \
        --checkpoint /path/to/checkpoint.pth.tar \
        --output_dir /path/to/output \
        --model_type 3d
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
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
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
    """
    Extract embeddings for all cells.

    Returns:
        embeddings: numpy array [N, embed_dim]
        protein_names: list of protein names [N]
    """
    model.eval()
    all_embeddings = []
    all_protein_names = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting embeddings"):
            images = batch['image'].to(device)
            protein_names = batch['protein_name']

            # Get normalized embeddings from projection head
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
    protein_cell_counts = {}

    for prot, embs in protein_to_embeddings.items():
        # Mean pooling and re-normalize
        mean_emb = np.mean(embs, axis=0)
        mean_emb = mean_emb / (np.linalg.norm(mean_emb) + 1e-8)
        protein_embeddings[prot] = mean_emb
        protein_cell_counts[prot] = len(embs)

    print(f"Aggregated {len(embeddings)} cells into {len(protein_embeddings)} protein embeddings")
    return protein_embeddings, protein_cell_counts


def load_ppi_data(ppi_path, pval_threshold=5, enrichment_threshold=2.5,
                  stoichiometry_threshold=0.05):
    """Load and filter PPI data."""
    ppi_df = pd.read_csv(ppi_path)
    print(f"Loaded {len(ppi_df)} total PPI records")

    filtered = ppi_df[
        (ppi_df['pval'] > pval_threshold) &
        (ppi_df['enrichment'] > enrichment_threshold) &
        (ppi_df['interaction_stoichiometry'] > stoichiometry_threshold)
    ].copy()

    print(f"After filtering: {len(filtered)} interactions")
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

    print(f"Loaded abundance data for {len(abundance_dict)} proteins")
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
    print(f"Built {len(positive_pairs)} positive pairs")
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
    print(f"Built {len(negative_pairs)} negative pairs")
    return negative_pairs


def compute_similarities(pairs, protein_embeddings):
    """Compute cosine similarity for pairs (embeddings are already normalized)."""
    similarities = []

    for prot1, prot2 in pairs:
        emb1 = protein_embeddings[prot1]
        emb2 = protein_embeddings[prot2]
        sim = np.dot(emb1, emb2)
        similarities.append(sim)

    return np.array(similarities)


def evaluate_ppi_prediction(positive_similarities, negative_similarities):
    """Evaluate PPI prediction."""
    labels = np.concatenate([
        np.ones(len(positive_similarities)),
        np.zeros(len(negative_similarities))
    ])

    scores = np.concatenate([positive_similarities, negative_similarities])

    roc_auc = roc_auc_score(labels, scores)
    avg_precision = average_precision_score(labels, scores)

    fpr, tpr, thresholds = roc_curve(labels, scores)

    # Accuracy at threshold 0
    predictions = (scores > 0).astype(float)
    accuracy = (predictions == labels).mean()

    metrics = {
        'roc_auc': float(roc_auc),
        'average_precision': float(avg_precision),
        'accuracy': float(accuracy),
        'n_positive_pairs': int(len(positive_similarities)),
        'n_negative_pairs': int(len(negative_similarities)),
        'mean_positive_similarity': float(np.mean(positive_similarities)),
        'mean_negative_similarity': float(np.mean(negative_similarities)),
        'std_positive_similarity': float(np.std(positive_similarities)),
        'std_negative_similarity': float(np.std(negative_similarities)),
        'fpr': fpr.tolist(),
        'tpr': tpr.tolist(),
    }

    return metrics


def plot_results(metrics, output_path):
    """Plot ROC curve and similarity distributions."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ROC Curve
    ax1 = axes[0]
    ax1.plot(metrics['fpr'], metrics['tpr'], 'b-', linewidth=2,
             label=f"ROC (AUC = {metrics['roc_auc']:.3f})")
    ax1.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    ax1.set_xlabel('False Positive Rate', fontsize=12)
    ax1.set_ylabel('True Positive Rate', fontsize=12)
    ax1.set_title('ROC Curve for PPI Prediction (Metric Learning)', fontsize=14)
    ax1.legend(loc='lower right', fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Similarity distribution
    ax2 = axes[1]
    ax2.bar(['Positive Pairs\n(Known PPIs)', 'Negative Pairs\n(Random)'],
            [metrics['mean_positive_similarity'], metrics['mean_negative_similarity']],
            yerr=[metrics['std_positive_similarity'], metrics['std_negative_similarity']],
            color=['green', 'red'], alpha=0.7, capsize=5)
    ax2.set_ylabel('Cosine Similarity', fontsize=12)
    ax2.set_title('Mean Cosine Similarity by Pair Type', fontsize=14)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)

    ax2.text(0, metrics['mean_positive_similarity'] + metrics['std_positive_similarity'] + 0.05,
             f"n={metrics['n_positive_pairs']}", ha='center', fontsize=10)
    ax2.text(1, metrics['mean_negative_similarity'] + metrics['std_negative_similarity'] + 0.05,
             f"n={metrics['n_negative_pairs']}", ha='center', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved plot to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate PPI Metric Learning Model')

    parser.add_argument('--config', type=str, required=True,
                        help='Path to config.yaml from the trained model')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--model_type', type=str, default='3d', choices=['2d', '3d'],
                        help='Model type')

    parser.add_argument('--csv_path', type=str,
                        default='/ictstr01/groups/labs/marr/qscd01/datasets/SingleCellImagesDataset/opencell/opencell_dataset/single_cells/metadata/dataset1/',
                        help='Path to directory containing test.csv')
    parser.add_argument('--ppi_path', type=str,
                        default='/ictstr01/groups/labs/marr/qscd01/datasets/SingleCellImagesDataset/opencell/opencell_metadata_raw/protein-protein-interactions/opencell-protein-interactions.csv',
                        help='Path to PPI data CSV')
    parser.add_argument('--abundance_path', type=str,
                        default='/ictstr01/groups/labs/marr/qscd01/datasets/SingleCellImagesDataset/opencell/opencell_metadata_raw/protein-abundance/opencell-protein-abundance.csv',
                        help='Path to protein abundance CSV')

    parser.add_argument('--pval_threshold', type=float, default=5.0)
    parser.add_argument('--enrichment_threshold', type=float, default=2.5)
    parser.add_argument('--stoichiometry_threshold', type=float, default=0.05)
    parser.add_argument('--n_abundance_buckets', type=int, default=10)
    parser.add_argument('--n_negatives_per_positive', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--split', type=str, default='test')

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load config and model
    config = load_config(args.config)
    model = build_model(config, args.model_type)
    model = load_checkpoint(model, args.checkpoint, device)
    model = model.to(device)
    model.eval()

    # Load transforms
    use_max_projection = args.model_type == '2d'
    if args.model_type == '2d':
        transform = get_opencell_2d_val_transforms()
    else:
        transform = get_opencell_val_transforms()

    # Create dataset
    csv_path = os.path.join(args.csv_path, f'{args.split}.csv')
    print(f"Loading dataset from {csv_path}")

    dataset = OpenCellPPITestDataset(
        csv_path=csv_path,
        transform=transform,
        use_max_projection=use_max_projection
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    print(f"Dataset size: {len(dataset)} cells")

    # Extract embeddings
    print("\n" + "=" * 60)
    print("Step 1: Extracting cell embeddings")
    print("=" * 60)
    embeddings, protein_names = extract_embeddings(model, dataloader, device)
    print(f"Extracted embeddings shape: {embeddings.shape}")

    # Aggregate to protein level
    print("\n" + "=" * 60)
    print("Step 2: Aggregating to protein-level embeddings")
    print("=" * 60)
    protein_embeddings, protein_cell_counts = aggregate_protein_embeddings(
        embeddings, protein_names
    )

    # Save embeddings
    emb_path = os.path.join(args.output_dir, 'protein_embeddings.npz')
    np.savez(
        emb_path,
        embeddings=np.array(list(protein_embeddings.values())),
        protein_names=list(protein_embeddings.keys()),
        cell_counts=np.array(list(protein_cell_counts.values()))
    )
    print(f"Saved protein embeddings to {emb_path}")

    # Load PPI data
    print("\n" + "=" * 60)
    print("Step 3: Loading and filtering PPI data")
    print("=" * 60)
    ppi_df = load_ppi_data(
        args.ppi_path,
        pval_threshold=args.pval_threshold,
        enrichment_threshold=args.enrichment_threshold,
        stoichiometry_threshold=args.stoichiometry_threshold
    )

    # Load abundance data
    print("\n" + "=" * 60)
    print("Step 4: Loading abundance data")
    print("=" * 60)
    abundance_dict = load_abundance_data(args.abundance_path)

    available_proteins = list(protein_embeddings.keys())
    bucket_assignments, bucket_proteins = assign_abundance_buckets(
        available_proteins, abundance_dict, n_buckets=args.n_abundance_buckets
    )

    # Build pairs
    print("\n" + "=" * 60)
    print("Step 5: Building positive and negative pairs")
    print("=" * 60)
    positive_pairs = build_positive_pairs(ppi_df, available_proteins)

    if len(positive_pairs) == 0:
        print("ERROR: No positive pairs found!")
        return

    negative_pairs = build_negative_pairs(
        positive_pairs, bucket_assignments, bucket_proteins,
        n_negatives_per_positive=args.n_negatives_per_positive,
        seed=args.seed
    )

    # Compute similarities
    print("\n" + "=" * 60)
    print("Step 6: Computing cosine similarities")
    print("=" * 60)
    positive_similarities = compute_similarities(positive_pairs, protein_embeddings)
    negative_similarities = compute_similarities(negative_pairs, protein_embeddings)

    print(f"Positive pairs mean similarity: {np.mean(positive_similarities):.4f} +/- {np.std(positive_similarities):.4f}")
    print(f"Negative pairs mean similarity: {np.mean(negative_similarities):.4f} +/- {np.std(negative_similarities):.4f}")

    # Evaluate
    print("\n" + "=" * 60)
    print("Step 7: Evaluating PPI prediction")
    print("=" * 60)
    metrics = evaluate_ppi_prediction(positive_similarities, negative_similarities)

    print(f"\n{'=' * 40}")
    print("RESULTS")
    print(f"{'=' * 40}")
    print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"Average Precision: {metrics['average_precision']:.4f}")
    print(f"Accuracy (threshold=0): {metrics['accuracy']:.4f}")
    print(f"Positive pairs: {metrics['n_positive_pairs']}")
    print(f"Negative pairs: {metrics['n_negative_pairs']}")

    # Save metrics
    metrics_path = os.path.join(args.output_dir, 'ppi_metrics.json')
    metrics_to_save = {k: v for k, v in metrics.items() if k not in ['fpr', 'tpr']}
    metrics_to_save['config'] = {
        'model_type': args.model_type,
        'pval_threshold': args.pval_threshold,
        'enrichment_threshold': args.enrichment_threshold,
        'stoichiometry_threshold': args.stoichiometry_threshold,
        'checkpoint': args.checkpoint,
    }

    with open(metrics_path, 'w') as f:
        json.dump(metrics_to_save, f, indent=2)
    print(f"\nSaved metrics to {metrics_path}")

    # Plot results
    plot_path = os.path.join(args.output_dir, 'ppi_metric_evaluation.png')
    plot_results(metrics, plot_path)

    # Save pairs
    pairs_path = os.path.join(args.output_dir, 'pairs.npz')
    np.savez(
        pairs_path,
        positive_pairs=np.array(positive_pairs),
        negative_pairs=np.array(negative_pairs),
        positive_similarities=positive_similarities,
        negative_similarities=negative_similarities
    )
    print(f"Saved pairs to {pairs_path}")

    print("\n" + "=" * 60)
    print("PPI Metric Learning Evaluation Complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
