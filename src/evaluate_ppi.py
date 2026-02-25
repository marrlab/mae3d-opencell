"""
Evaluate MAE embeddings on Protein-Protein Interaction (PPI) prediction.

This script:
1. Loads a trained MAE model (2D or 3D)
2. Extracts embeddings for all cells in the test set
3. Computes mean embedding per protein
4. Loads PPI data and filters by significance thresholds
5. Creates positive pairs (known PPIs) and negative pairs (random, abundance-matched)
6. Evaluates using ROC-AUC on cosine/Euclidean distances

Usage:
    python src/evaluate_ppi.py \
        --config configs/opencell/opencell_3d.yaml \
        --checkpoint /path/to/checkpoint.pth.tar \
        --output_dir /path/to/output \
        --model_type 3d \
        --pval_threshold 5 \
        --enrichment_threshold 2.5 \
        --stoichiometry_threshold 0.05
"""

import argparse
import os
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
import matplotlib.pyplot as plt
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from omegaconf import OmegaConf


# ============================================================================
# Data Loading
# ============================================================================

class OpenCellDatasetWithProteinName(Dataset):
    """OpenCell dataset that returns protein names along with images."""

    def __init__(self, csv_path, transform=None, use_max_projection=False):
        import tifffile
        self.tifffile = tifffile

        self.df = pd.read_csv(csv_path)
        self.transform = transform
        self.use_max_projection = use_max_projection

        # Get protein names from file_gene_symbol column
        self.protein_names = self.df['file_gene_symbol'].tolist()
        self.image_paths = self.df['image_path'].tolist()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        protein_name = self.protein_names[idx]

        # Load image
        img = self.tifffile.imread(img_path)  # Shape: (Z, C, Y, X)

        if self.use_max_projection:
            img = np.max(img, axis=0)  # (C, Y, X)

        data = {"image": img, "protein_name": protein_name}

        if self.transform:
            # Transform only the image
            img_data = {"image": img}
            img_data = self.transform(img_data)
            data["image"] = img_data["image"]

        return data


# ============================================================================
# Model Loading
# ============================================================================

def load_config(config_path):
    """Load config from yaml file using OmegaConf."""
    config = OmegaConf.load(config_path)
    OmegaConf.resolve(config)  # Resolve ${...} interpolations
    return config


def build_model(config, model_type='3d'):
    """Build MAE model from config."""
    if model_type == '3d':
        from lib.models.mae3d import MAE3D
        from lib.networks.mae_vit import MAEViTEncoder, MAEViTDecoder
        model = MAE3D(MAEViTEncoder, MAEViTDecoder, config)
    else:
        from lib.models.mae2d import MAE2D
        from lib.networks.mae_vit import MAEViTEncoder, MAEViTDecoder
        model = MAE2D(MAEViTEncoder, MAEViTDecoder, config)
    return model


def load_checkpoint(model, checkpoint_path, device):
    """Load model weights from checkpoint."""
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    # Remove 'module.' prefix if present (from DDP training)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict, strict=True)
    print(f"Loaded checkpoint (epoch: {checkpoint.get('epoch', 'unknown')})")
    return model


# ============================================================================
# Embedding Extraction
# ============================================================================

def extract_embeddings_with_proteins(model, dataloader, device, model_type='3d', pooling='mean'):
    """
    Extract embeddings and return with protein names.

    Returns:
        embeddings: numpy array [N, embed_dim]
        protein_names: list of protein names [N]
    """
    if model_type == '3d':
        from lib.models.mae3d import patchify_image
    else:
        from lib.models.mae2d import patchify_image

    model.eval()
    all_embeddings = []
    all_protein_names = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting embeddings"):
            images = batch['image'].to(device)
            protein_names = batch['protein_name']

            batch_size = images.size(0)

            # Patchify
            x = patchify_image(images, model.patch_size)

            # Get positional embeddings
            pos_embed = model.encoder_pos_embed.expand(batch_size, -1, -1).to(device)

            # Forward through encoder
            features = model.encoder.forward_features(x, pos_embed)

            # Pooling
            if pooling == 'cls':
                embeddings = features[:, 0, :]
            else:
                embeddings = features[:, 1:, :].mean(dim=1)

            all_embeddings.append(embeddings.cpu().numpy())
            all_protein_names.extend(protein_names)

    embeddings = np.concatenate(all_embeddings, axis=0)
    return embeddings, all_protein_names


def aggregate_protein_embeddings(embeddings, protein_names):
    """
    Compute mean embedding per protein.

    Returns:
        protein_embeddings: dict {protein_name: mean_embedding}
        protein_cell_counts: dict {protein_name: num_cells}
    """
    protein_to_embeddings = defaultdict(list)

    for emb, prot in zip(embeddings, protein_names):
        protein_to_embeddings[prot].append(emb)

    protein_embeddings = {}
    protein_cell_counts = {}

    for prot, embs in protein_to_embeddings.items():
        protein_embeddings[prot] = np.mean(embs, axis=0)
        protein_cell_counts[prot] = len(embs)

    print(f"Aggregated {len(embeddings)} cells into {len(protein_embeddings)} protein embeddings")
    return protein_embeddings, protein_cell_counts


# ============================================================================
# PPI Data Processing
# ============================================================================

def load_ppi_data(ppi_path, pval_threshold=5, enrichment_threshold=2.5,
                  stoichiometry_threshold=0.05):
    """
    Load and filter PPI data.

    Args:
        ppi_path: Path to opencell-protein-interactions.csv
        pval_threshold: Minimum -log10(pval) threshold
        enrichment_threshold: Minimum enrichment threshold
        stoichiometry_threshold: Minimum interaction_stoichiometry threshold

    Returns:
        filtered_ppi: DataFrame with filtered interactions
    """
    ppi_df = pd.read_csv(ppi_path)
    print(f"Loaded {len(ppi_df)} total PPI records")

    # Filter by thresholds
    # Note: pval column is already -log10 transformed
    filtered = ppi_df[
        (ppi_df['pval'] > pval_threshold) &
        (ppi_df['enrichment'] > enrichment_threshold) &
        (ppi_df['interaction_stoichiometry'] > stoichiometry_threshold)
    ].copy()

    print(f"After filtering (pval>{pval_threshold}, enrichment>{enrichment_threshold}, "
          f"stoich>{stoichiometry_threshold}): {len(filtered)} interactions")

    return filtered


def load_abundance_data(abundance_path):
    """
    Load protein abundance data for bucketing.

    Returns:
        abundance_dict: dict {gene_name: abundance}
    """
    abundance_df = pd.read_csv(abundance_path)

    # Use protein concentration if available, otherwise RNA TPM
    abundance_dict = {}
    for _, row in abundance_df.iterrows():
        gene = row['gene_name']
        # Prefer protein concentration, fall back to RNA
        if pd.notna(row['hek_protein_conc_nm']):
            abundance_dict[gene] = row['hek_protein_conc_nm']
        elif pd.notna(row['hek_rna_tpm']):
            abundance_dict[gene] = row['hek_rna_tpm']

    print(f"Loaded abundance data for {len(abundance_dict)} proteins")
    return abundance_dict


def assign_abundance_buckets(proteins, abundance_dict, n_buckets=10):
    """
    Assign proteins to abundance buckets.

    Returns:
        bucket_assignments: dict {protein: bucket_id}
        bucket_proteins: dict {bucket_id: [proteins]}
    """
    # Get abundance for available proteins
    protein_abundance = []
    for prot in proteins:
        if prot in abundance_dict:
            protein_abundance.append((prot, abundance_dict[prot]))
        else:
            # Assign to middle bucket if no abundance data
            protein_abundance.append((prot, None))

    # Separate proteins with and without abundance data
    with_abundance = [(p, a) for p, a in protein_abundance if a is not None]
    without_abundance = [p for p, a in protein_abundance if a is None]

    bucket_assignments = {}
    bucket_proteins = defaultdict(list)

    if with_abundance:
        # Sort by abundance and assign buckets
        with_abundance.sort(key=lambda x: x[1])
        bucket_size = len(with_abundance) / n_buckets

        for i, (prot, _) in enumerate(with_abundance):
            bucket_id = min(int(i / bucket_size), n_buckets - 1)
            bucket_assignments[prot] = bucket_id
            bucket_proteins[bucket_id].append(prot)

    # Assign proteins without abundance to a special bucket
    for prot in without_abundance:
        bucket_assignments[prot] = -1
        bucket_proteins[-1].append(prot)

    print(f"Assigned {len(bucket_assignments)} proteins to {n_buckets} buckets")
    print(f"  Proteins without abundance data: {len(without_abundance)}")

    return bucket_assignments, dict(bucket_proteins)


# ============================================================================
# Pair Generation
# ============================================================================

def build_positive_pairs(ppi_df, available_proteins):
    """
    Build positive pairs from filtered PPI data.
    Only includes pairs where both proteins have embeddings.

    Returns:
        positive_pairs: list of (protein1, protein2) tuples
    """
    available_set = set(available_proteins)
    positive_pairs = []

    for _, row in ppi_df.iterrows():
        target = row['target_gene_name']
        interactor = row['interactor_gene_name']

        if target in available_set and interactor in available_set:
            # Ensure consistent ordering
            pair = tuple(sorted([target, interactor]))
            positive_pairs.append(pair)

    # Remove duplicates
    positive_pairs = list(set(positive_pairs))
    print(f"Built {len(positive_pairs)} positive pairs (from proteins with embeddings)")

    return positive_pairs


def build_negative_pairs(positive_pairs, bucket_assignments, bucket_proteins,
                         n_negatives_per_positive=1, seed=42):
    """
    Build negative pairs matched by abundance bucket.

    For each positive pair (A, B), sample a negative pair from proteins
    in the same abundance buckets as A and B.

    Returns:
        negative_pairs: list of (protein1, protein2) tuples
    """
    np.random.seed(seed)

    positive_set = set(positive_pairs)
    negative_pairs = []

    # Get all proteins by bucket
    all_proteins = list(bucket_assignments.keys())

    for prot1, prot2 in positive_pairs:
        bucket1 = bucket_assignments.get(prot1, -1)
        bucket2 = bucket_assignments.get(prot2, -1)

        # Get candidate proteins from same buckets
        candidates1 = bucket_proteins.get(bucket1, all_proteins)
        candidates2 = bucket_proteins.get(bucket2, all_proteins)

        # Try to find valid negative pairs
        for _ in range(n_negatives_per_positive * 10):  # Try multiple times
            # Random protein from bucket1's candidates
            neg1 = np.random.choice(candidates1)
            # Random protein from bucket2's candidates
            neg2 = np.random.choice(candidates2)

            if neg1 == neg2:
                continue

            neg_pair = tuple(sorted([neg1, neg2]))

            # Check it's not a positive pair
            if neg_pair not in positive_set and neg_pair not in negative_pairs:
                negative_pairs.append(neg_pair)
                if len([p for p in negative_pairs if p == neg_pair]) >= n_negatives_per_positive:
                    break

    # If we couldn't match all, sample randomly
    while len(negative_pairs) < len(positive_pairs) * n_negatives_per_positive:
        neg1, neg2 = np.random.choice(all_proteins, 2, replace=False)
        neg_pair = tuple(sorted([neg1, neg2]))
        if neg_pair not in positive_set and neg_pair not in negative_pairs:
            negative_pairs.append(neg_pair)

    negative_pairs = negative_pairs[:len(positive_pairs) * n_negatives_per_positive]
    print(f"Built {len(negative_pairs)} negative pairs (abundance-matched)")

    return negative_pairs


# ============================================================================
# Distance Computation and Evaluation
# ============================================================================

def compute_distances(pairs, protein_embeddings, metric='cosine'):
    """
    Compute distances between protein pairs.

    Args:
        pairs: list of (protein1, protein2) tuples
        protein_embeddings: dict {protein: embedding}
        metric: 'cosine' or 'euclidean'

    Returns:
        distances: numpy array of distances
    """
    distances = []

    for prot1, prot2 in pairs:
        emb1 = protein_embeddings[prot1]
        emb2 = protein_embeddings[prot2]

        if metric == 'cosine':
            # Cosine distance = 1 - cosine_similarity
            cos_sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2) + 1e-8)
            dist = 1 - cos_sim
        else:
            # Euclidean distance
            dist = np.linalg.norm(emb1 - emb2)

        distances.append(dist)

    return np.array(distances)


def evaluate_ppi_prediction(positive_distances, negative_distances):
    """
    Evaluate PPI prediction using ROC-AUC.

    For PPI prediction:
    - Positive pairs should have SMALLER distances (more similar)
    - We want to distinguish positives from negatives

    Returns:
        metrics: dict with evaluation metrics
    """
    # Labels: 1 for positive pairs, 0 for negative pairs
    labels = np.concatenate([
        np.ones(len(positive_distances)),
        np.zeros(len(negative_distances))
    ])

    # Distances (smaller = more similar = more likely to be positive)
    distances = np.concatenate([positive_distances, negative_distances])

    # For ROC-AUC, we use negative distance as "score" (higher score = more likely positive)
    scores = -distances

    # Compute metrics
    roc_auc = roc_auc_score(labels, scores)
    avg_precision = average_precision_score(labels, scores)

    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(labels, scores)

    metrics = {
        'roc_auc': float(roc_auc),
        'average_precision': float(avg_precision),
        'n_positive_pairs': int(len(positive_distances)),
        'n_negative_pairs': int(len(negative_distances)),
        'mean_positive_distance': float(np.mean(positive_distances)),
        'mean_negative_distance': float(np.mean(negative_distances)),
        'std_positive_distance': float(np.std(positive_distances)),
        'std_negative_distance': float(np.std(negative_distances)),
        'fpr': fpr.tolist(),
        'tpr': tpr.tolist(),
    }

    return metrics


def plot_results(metrics, output_path, metric_name='cosine'):
    """Plot ROC curve and distance distributions."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ROC Curve
    ax1 = axes[0]
    ax1.plot(metrics['fpr'], metrics['tpr'], 'b-', linewidth=2,
             label=f"ROC (AUC = {metrics['roc_auc']:.3f})")
    ax1.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    ax1.set_xlabel('False Positive Rate', fontsize=12)
    ax1.set_ylabel('True Positive Rate', fontsize=12)
    ax1.set_title('ROC Curve for PPI Prediction', fontsize=14)
    ax1.legend(loc='lower right', fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Distance distribution
    ax2 = axes[1]
    ax2.bar(['Positive Pairs\n(Known PPIs)', 'Negative Pairs\n(Random)'],
            [metrics['mean_positive_distance'], metrics['mean_negative_distance']],
            yerr=[metrics['std_positive_distance'], metrics['std_negative_distance']],
            color=['green', 'red'], alpha=0.7, capsize=5)
    ax2.set_ylabel(f'{metric_name.capitalize()} Distance', fontsize=12)
    ax2.set_title(f'Mean {metric_name.capitalize()} Distance by Pair Type', fontsize=14)
    ax2.grid(True, alpha=0.3, axis='y')

    # Add counts
    ax2.text(0, metrics['mean_positive_distance'] + metrics['std_positive_distance'] + 0.02,
             f"n={metrics['n_positive_pairs']}", ha='center', fontsize=10)
    ax2.text(1, metrics['mean_negative_distance'] + metrics['std_negative_distance'] + 0.02,
             f"n={metrics['n_negative_pairs']}", ha='center', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved plot to {output_path}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Evaluate MAE embeddings on PPI prediction')

    # Model arguments
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config.yaml from the trained model')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--model_type', type=str, default='3d', choices=['2d', '3d'],
                        help='Model type (2d or 3d)')
    parser.add_argument('--pooling', type=str, default='mean', choices=['mean', 'cls'],
                        help='Pooling method for embeddings')

    # Data arguments
    parser.add_argument('--csv_path', type=str,
                        default='/ictstr01/groups/labs/marr/qscd01/datasets/SingleCellImagesDataset/opencell/opencell_dataset/single_cells/metadata/dataset1/',
                        help='Path to directory containing train/val/test.csv')
    parser.add_argument('--ppi_path', type=str,
                        default='/ictstr01/groups/labs/marr/qscd01/datasets/SingleCellImagesDataset/opencell/opencell_metadata_raw/protein-protein-interactions/opencell-protein-interactions.csv',
                        help='Path to PPI data CSV')
    parser.add_argument('--abundance_path', type=str,
                        default='/ictstr01/groups/labs/marr/qscd01/datasets/SingleCellImagesDataset/opencell/opencell_metadata_raw/protein-abundance/opencell-protein-abundance.csv',
                        help='Path to protein abundance CSV')

    # Filtering thresholds
    parser.add_argument('--pval_threshold', type=float, default=5.0,
                        help='Minimum -log10(pval) threshold for PPI filtering')
    parser.add_argument('--enrichment_threshold', type=float, default=2.5,
                        help='Minimum enrichment threshold for PPI filtering')
    parser.add_argument('--stoichiometry_threshold', type=float, default=0.05,
                        help='Minimum interaction_stoichiometry threshold for PPI filtering')

    # Evaluation arguments
    parser.add_argument('--n_abundance_buckets', type=int, default=10,
                        help='Number of abundance buckets for negative sampling')
    parser.add_argument('--n_negatives_per_positive', type=int, default=1,
                        help='Number of negative pairs per positive pair')
    parser.add_argument('--distance_metric', type=str, default='cosine',
                        choices=['cosine', 'euclidean'],
                        help='Distance metric for comparing embeddings')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for negative sampling')

    # Output arguments
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save results')

    # Other arguments
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for inference')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--split', type=str, default='test',
                        help='Dataset split to use')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load config and model
    config = load_config(args.config)
    model = build_model(config, args.model_type)
    model = load_checkpoint(model, args.checkpoint, device)
    model = model.to(device)
    model.eval()

    # Load transforms
    if args.model_type == '3d':
        from data.opencell.transforms import get_opencell_val_transforms
        transform = get_opencell_val_transforms(
            channel_wise_norm=getattr(config, 'channel_wise_norm', True)
        )
        use_max_projection = False
    else:
        from data.opencell.transforms import get_opencell_2d_val_transforms
        transform = get_opencell_2d_val_transforms()
        use_max_projection = getattr(config, 'use_max_projection', True)

    # Create dataset
    csv_path = os.path.join(args.csv_path, f'{args.split}.csv')
    print(f"Loading dataset from {csv_path}")

    dataset = OpenCellDatasetWithProteinName(
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
    print("\n" + "="*60)
    print("Step 1: Extracting cell embeddings")
    print("="*60)
    embeddings, protein_names = extract_embeddings_with_proteins(
        model, dataloader, device, args.model_type, args.pooling
    )
    print(f"Extracted embeddings shape: {embeddings.shape}")

    # Aggregate to protein level
    print("\n" + "="*60)
    print("Step 2: Aggregating to protein-level embeddings")
    print("="*60)
    protein_embeddings, protein_cell_counts = aggregate_protein_embeddings(
        embeddings, protein_names
    )

    # L2 normalize embeddings for cosine distance
    if args.distance_metric == 'cosine':
        print("Applying L2 normalization to protein embeddings (cosine distance)")
        for prot in protein_embeddings:
            emb = protein_embeddings[prot]
            protein_embeddings[prot] = emb / (np.linalg.norm(emb) + 1e-8)

    # Save protein embeddings
    protein_emb_path = os.path.join(args.output_dir, 'protein_embeddings.npz')
    np.savez(
        protein_emb_path,
        embeddings=np.array(list(protein_embeddings.values())),
        protein_names=list(protein_embeddings.keys()),
        cell_counts=np.array(list(protein_cell_counts.values()))
    )
    print(f"Saved protein embeddings to {protein_emb_path}")

    # Load PPI data
    print("\n" + "="*60)
    print("Step 3: Loading and filtering PPI data")
    print("="*60)
    ppi_df = load_ppi_data(
        args.ppi_path,
        pval_threshold=args.pval_threshold,
        enrichment_threshold=args.enrichment_threshold,
        stoichiometry_threshold=args.stoichiometry_threshold
    )

    # Load abundance data
    print("\n" + "="*60)
    print("Step 4: Loading abundance data for bucket matching")
    print("="*60)
    abundance_dict = load_abundance_data(args.abundance_path)

    # Assign abundance buckets
    available_proteins = list(protein_embeddings.keys())
    bucket_assignments, bucket_proteins = assign_abundance_buckets(
        available_proteins, abundance_dict, n_buckets=args.n_abundance_buckets
    )

    # Build positive pairs
    print("\n" + "="*60)
    print("Step 5: Building positive and negative pairs")
    print("="*60)
    positive_pairs = build_positive_pairs(ppi_df, available_proteins)

    if len(positive_pairs) == 0:
        print("ERROR: No positive pairs found! Check if proteins in PPI data match test set.")
        return

    # Build negative pairs
    negative_pairs = build_negative_pairs(
        positive_pairs, bucket_assignments, bucket_proteins,
        n_negatives_per_positive=args.n_negatives_per_positive,
        seed=args.seed
    )

    # Compute distances
    print("\n" + "="*60)
    print(f"Step 6: Computing {args.distance_metric} distances")
    print("="*60)
    positive_distances = compute_distances(
        positive_pairs, protein_embeddings, metric=args.distance_metric
    )
    negative_distances = compute_distances(
        negative_pairs, protein_embeddings, metric=args.distance_metric
    )

    print(f"Positive pairs mean distance: {np.mean(positive_distances):.4f} ± {np.std(positive_distances):.4f}")
    print(f"Negative pairs mean distance: {np.mean(negative_distances):.4f} ± {np.std(negative_distances):.4f}")

    # Evaluate
    print("\n" + "="*60)
    print("Step 7: Evaluating PPI prediction")
    print("="*60)
    metrics = evaluate_ppi_prediction(positive_distances, negative_distances)

    print(f"\n{'='*40}")
    print("RESULTS")
    print(f"{'='*40}")
    print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"Average Precision: {metrics['average_precision']:.4f}")
    print(f"Positive pairs: {metrics['n_positive_pairs']}")
    print(f"Negative pairs: {metrics['n_negative_pairs']}")

    # Save metrics
    metrics_path = os.path.join(args.output_dir, 'ppi_metrics.json')

    # Remove large arrays for JSON
    metrics_to_save = {k: v for k, v in metrics.items() if k not in ['fpr', 'tpr']}
    metrics_to_save['config'] = {
        'model_type': args.model_type,
        'pooling': args.pooling,
        'distance_metric': args.distance_metric,
        'pval_threshold': args.pval_threshold,
        'enrichment_threshold': args.enrichment_threshold,
        'stoichiometry_threshold': args.stoichiometry_threshold,
        'n_abundance_buckets': args.n_abundance_buckets,
        'checkpoint': args.checkpoint,
    }

    with open(metrics_path, 'w') as f:
        json.dump(metrics_to_save, f, indent=2)
    print(f"\nSaved metrics to {metrics_path}")

    # Plot results
    plot_path = os.path.join(args.output_dir, f'ppi_evaluation_{args.distance_metric}.png')
    plot_results(metrics, plot_path, args.distance_metric)

    # Save pairs for analysis
    pairs_path = os.path.join(args.output_dir, 'pairs.npz')
    np.savez(
        pairs_path,
        positive_pairs=np.array(positive_pairs),
        negative_pairs=np.array(negative_pairs),
        positive_distances=positive_distances,
        negative_distances=negative_distances
    )
    print(f"Saved pairs to {pairs_path}")

    print("\n" + "="*60)
    print("PPI Evaluation Complete!")
    print("="*60)


if __name__ == '__main__':
    main()
