#!/usr/bin/env python3
"""
Evaluate a trained kfold PPI metric model using precomputed MAE embeddings.

Use this script (not evaluate_ppi_metric.py) when:
  - The model was trained with --mae_embedding_csv_path (kfold embedding mode)
  - The encoder has random/frozen weights; only the projection_head was trained

Steps:
  1. Load config and build model (PPIMetric3DCrossAttention)
  2. Load checkpoint (only projection_head weights matter)
  3. Build combined embedding lookup from all dataset1 source splits
  4. For each cell in the kfold test CSV, lookup embedding -> run projection_head
  5. Aggregate per protein (mean-pool + re-normalize)
  6. Build positive/negative PPI pairs
  7. Evaluate (ROC-AUC, Average Precision)
  8. Save results

Usage:
    python src/evaluate_ppi_metric_emb.py \\
        --config configs/opencell/opencell_ppi_emb_3d_fft_kfold.yaml \\
        --checkpoint /path/to/ppi_3d_fft_emb_kfold5/fold0/ckpts/checkpoint_0099.pth.tar \\
        --mae_embedding_path /path/to/mae_opencell_3d.../fold0/mae3d_embeddings \\
        --mae_embedding_csv_path /path/to/dataset1 \\
        --csv_path /path/to/kfold5/fold0 \\
        --split test \\
        --output_dir /path/to/ppi_3d_fft_emb_kfold5/fold0/eval_results
"""

import argparse
import os
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
import matplotlib.pyplot as plt
import json

sys.path.insert(0, str(Path(__file__).parent))

from omegaconf import OmegaConf
from lib.models.ppi_metric import PPIMetric3D, PPIMetric2D, PPIMetric3DCrossAttention


# ============================================================================
# Model helpers
# ============================================================================

def load_config(config_path):
    config = OmegaConf.load(config_path)
    OmegaConf.resolve(config)
    return config


def build_model(config):
    arch = getattr(config, 'arch', 'PPIMetric3DCrossAttention')

    model_params = {
        'input_size': tuple(config.input_size),
        'patch_size':  tuple(config.patch_size),
        'in_chans':    config.in_chans,
        'embed_dim':   config.encoder_embed_dim,
        'depth':       config.encoder_depth,
        'num_heads':   config.encoder_num_heads,
        'drop_path_rate': getattr(config, 'drop_path', 0.0),
        'pos_embed_type': getattr(config, 'pos_embed_type', 'sincos'),
        'use_global_pool': getattr(config, 'use_global_pool', True),
        'proj_hidden_dim': getattr(config, 'proj_hidden_dim', 512),
        'proj_output_dim': getattr(config, 'proj_output_dim', 128),
        'proj_num_layers': getattr(config, 'proj_num_layers', 2),
    }

    if arch == 'PPIMetric3DCrossAttention':
        model_params['cross_attention_type'] = getattr(config, 'cross_attention_type', 'position_wise')
        model_params['pool_mode'] = getattr(config, 'pool_mode', 'concat')
        model = PPIMetric3DCrossAttention(**model_params)
        print(f"  arch: PPIMetric3DCrossAttention  pool_mode={model_params['pool_mode']}")
    elif arch == 'PPIMetric2D':
        model_params['input_size'] = tuple(config.input_size)[1:] if len(config.input_size) == 3 else tuple(config.input_size)
        model_params['patch_size']  = tuple(config.patch_size)[1:]  if len(config.patch_size) == 3  else tuple(config.patch_size)
        model = PPIMetric2D(**model_params)
        print(f"  arch: PPIMetric2D")
    else:
        model = PPIMetric3D(**model_params)
        print(f"  arch: PPIMetric3D")

    return model


def load_checkpoint(model, checkpoint_path, device):
    print(f"Loading checkpoint from {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt.get('state_dict', ckpt)
    new_sd = {k[7:] if k.startswith('module.') else k: v for k, v in state_dict.items()}
    model.load_state_dict(new_sd, strict=True)
    print(f"  Loaded (epoch {ckpt.get('epoch', 'unknown')})")
    return model


# ============================================================================
# Embedding extraction via projection head
# ============================================================================

def build_combined_lookup(mae_embedding_path, mae_embedding_csv_path):
    """Concatenate train/val/test .npy + CSVs and return (combined_array, lookup_dict)."""
    all_embs = []
    lookup = {}
    offset = 0
    for sname in ('train', 'val', 'test'):
        npy  = os.path.join(mae_embedding_path, f'{sname}.npy')
        csv_s = os.path.join(mae_embedding_csv_path, f'{sname}.csv')
        if os.path.exists(npy) and os.path.exists(csv_s):
            arr = np.load(npy)
            sdf = pd.read_csv(csv_s)
            assert len(arr) == len(sdf), f"{sname}: npy ({len(arr)}) != csv ({len(sdf)})"
            for local_i, (_, row) in enumerate(sdf.iterrows()):
                lookup[row['image_path']] = offset + local_i
            all_embs.append(arr)
            offset += len(arr)
            print(f"  {sname}: {len(arr)} embeddings from {csv_s}")
    combined = np.concatenate(all_embs, axis=0)
    print(f"  Combined: {combined.shape},  {len(lookup)} unique image paths")
    return combined, lookup


def extract_projection_embeddings(model, csv_path, combined_emb, lookup,
                                   batch_size=512, device='cuda'):
    """
    Run each cell's MAE embedding through the projection_head.

    Returns:
        embeddings: np.ndarray [N_found, proj_output_dim]  (L2-normalized)
        protein_names: list[str] of length N_found
    """
    df = pd.read_csv(csv_path)
    if 'file_gene_symbol' in df.columns:
        df['protein_name'] = df['file_gene_symbol']
    elif 'folder_protein' in df.columns:
        df['protein_name'] = df['folder_protein']
    else:
        raise ValueError("CSV must have 'file_gene_symbol' or 'folder_protein'")

    # Collect valid (embedding, protein) pairs
    raw_embs = []
    protein_names = []
    missing = 0
    for _, row in df.iterrows():
        img_path = row['image_path']
        if img_path not in lookup:
            missing += 1
            continue
        raw_embs.append(combined_emb[lookup[img_path]])
        protein_names.append(row['protein_name'])

    if missing > 0:
        print(f"  Warning: {missing}/{len(df)} cells not found in combined lookup (skipped)")

    print(f"  Processing {len(raw_embs)} cells through projection_head...")
    model.projection_head.eval()
    all_proj = []
    for i in range(0, len(raw_embs), batch_size):
        batch = torch.tensor(
            np.array(raw_embs[i:i + batch_size]), dtype=torch.float32
        ).to(device)
        with torch.no_grad():
            z = model.projection_head(batch)
        all_proj.append(z.cpu().numpy())

    embeddings = np.concatenate(all_proj, axis=0)
    return embeddings, protein_names


# ============================================================================
# Protein-level aggregation
# ============================================================================

def aggregate_protein_embeddings(embeddings, protein_names):
    """Mean-pool cell embeddings per protein and re-normalize."""
    protein_to_embs = defaultdict(list)
    for emb, prot in zip(embeddings, protein_names):
        protein_to_embs[prot].append(emb)

    protein_embeddings = {}
    protein_cell_counts = {}
    for prot, embs in protein_to_embs.items():
        mean_emb = np.mean(embs, axis=0)
        mean_emb = mean_emb / (np.linalg.norm(mean_emb) + 1e-8)
        protein_embeddings[prot] = mean_emb
        protein_cell_counts[prot] = len(embs)

    print(f"  Aggregated {len(embeddings)} cells into {len(protein_embeddings)} proteins")
    return protein_embeddings, protein_cell_counts


# ============================================================================
# PPI evaluation helpers  (same logic as evaluate_ppi_metric.py)
# ============================================================================

def load_ppi_data(ppi_path, pval_threshold=5, enrichment_threshold=2.5,
                  stoichiometry_threshold=0.05):
    ppi_df = pd.read_csv(ppi_path)
    print(f"  Loaded {len(ppi_df)} total PPI records")
    filtered = ppi_df[
        (ppi_df['pval'] > pval_threshold) &
        (ppi_df['enrichment'] > enrichment_threshold) &
        (ppi_df['interaction_stoichiometry'] > stoichiometry_threshold)
    ].copy()
    print(f"  After filtering: {len(filtered)} interactions")
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
    print(f"  Loaded abundance data for {len(abundance_dict)} proteins")
    return abundance_dict


def assign_abundance_buckets(proteins, abundance_dict, n_buckets=10):
    protein_abundance = [(p, abundance_dict.get(p)) for p in proteins]
    with_abundance    = [(p, a) for p, a in protein_abundance if a is not None]
    without_abundance = [p for p, a in protein_abundance if a is None]

    bucket_assignments = {}
    bucket_proteins    = defaultdict(list)

    if with_abundance:
        with_abundance.sort(key=lambda x: x[1])
        bucket_size = len(with_abundance) / n_buckets
        for i, (prot, _) in enumerate(with_abundance):
            bid = min(int(i / bucket_size), n_buckets - 1)
            bucket_assignments[prot] = bid
            bucket_proteins[bid].append(prot)

    for prot in without_abundance:
        bucket_assignments[prot] = -1
        bucket_proteins[-1].append(prot)

    return bucket_assignments, dict(bucket_proteins)


def build_positive_pairs(ppi_df, available_proteins):
    available_set = set(available_proteins)
    pairs = set()
    for _, row in ppi_df.iterrows():
        t, i = row['target_gene_name'], row['interactor_gene_name']
        if t in available_set and i in available_set:
            pairs.add(tuple(sorted([t, i])))
    pairs = list(pairs)
    print(f"  Built {len(pairs)} positive pairs")
    return pairs


def build_negative_pairs(positive_pairs, bucket_assignments, bucket_proteins,
                         n_negatives_per_positive=1, seed=42):
    np.random.seed(seed)
    positive_set  = set(positive_pairs)
    negative_pairs = []
    all_proteins   = list(bucket_assignments.keys())

    for p1, p2 in positive_pairs:
        b1 = bucket_assignments.get(p1, -1)
        b2 = bucket_assignments.get(p2, -1)
        c1 = bucket_proteins.get(b1, all_proteins)
        c2 = bucket_proteins.get(b2, all_proteins)
        for _ in range(n_negatives_per_positive * 10):
            n1 = np.random.choice(c1)
            n2 = np.random.choice(c2)
            if n1 == n2:
                continue
            neg = tuple(sorted([n1, n2]))
            if neg not in positive_set and neg not in negative_pairs:
                negative_pairs.append(neg)
                break

    target = len(positive_pairs) * n_negatives_per_positive
    while len(negative_pairs) < target:
        n1, n2 = np.random.choice(all_proteins, 2, replace=False)
        neg = tuple(sorted([n1, n2]))
        if neg not in positive_set and neg not in negative_pairs:
            negative_pairs.append(neg)

    negative_pairs = negative_pairs[:target]
    print(f"  Built {len(negative_pairs)} negative pairs (abundance-matched)")
    return negative_pairs


def compute_similarities(pairs, protein_embeddings):
    sims = []
    for p1, p2 in pairs:
        e1, e2 = protein_embeddings[p1], protein_embeddings[p2]
        sims.append(float(np.dot(e1, e2)))
    return np.array(sims)


def evaluate_ppi_prediction(positive_sims, negative_sims):
    labels = np.concatenate([np.ones(len(positive_sims)), np.zeros(len(negative_sims))])
    scores = np.concatenate([positive_sims, negative_sims])
    roc_auc = roc_auc_score(labels, scores)
    avg_precision = average_precision_score(labels, scores)
    fpr, tpr, _ = roc_curve(labels, scores)
    accuracy = ((scores > 0).astype(float) == labels).mean()

    return {
        'roc_auc':               float(roc_auc),
        'average_precision':     float(avg_precision),
        'accuracy':              float(accuracy),
        'n_positive_pairs':      int(len(positive_sims)),
        'n_negative_pairs':      int(len(negative_sims)),
        'mean_positive_sim':     float(np.mean(positive_sims)),
        'mean_negative_sim':     float(np.mean(negative_sims)),
        'std_positive_sim':      float(np.std(positive_sims)),
        'std_negative_sim':      float(np.std(negative_sims)),
        'fpr': fpr.tolist(),
        'tpr': tpr.tolist(),
    }


def plot_results(metrics, output_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax1 = axes[0]
    ax1.plot(metrics['fpr'], metrics['tpr'], 'b-', linewidth=2,
             label=f"ROC (AUC = {metrics['roc_auc']:.3f})")
    ax1.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('ROC Curve — PPI Metric Learning (embedding mode)')
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.bar(['Positive\n(Known PPIs)', 'Negative\n(Random)'],
            [metrics['mean_positive_sim'], metrics['mean_negative_sim']],
            yerr=[metrics['std_positive_sim'], metrics['std_negative_sim']],
            color=['green', 'red'], alpha=0.7, capsize=5)
    ax2.set_ylabel('Cosine Similarity')
    ax2.set_title('Mean Cosine Similarity by Pair Type')
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.text(0, metrics['mean_positive_sim'] + metrics['std_positive_sim'] + 0.05,
             f"n={metrics['n_positive_pairs']}", ha='center')
    ax2.text(1, metrics['mean_negative_sim'] + metrics['std_negative_sim'] + 0.05,
             f"n={metrics['n_negative_pairs']}", ha='center')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved plot to {output_path}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Evaluate kfold PPI metric model using precomputed MAE embeddings'
    )
    parser.add_argument('--config',     type=str, required=True,  help='Path to config.yaml')
    parser.add_argument('--checkpoint', type=str, required=True,  help='Path to checkpoint')
    parser.add_argument('--output_dir', type=str, required=True,  help='Directory to save results')

    # Embedding arguments
    parser.add_argument('--mae_embedding_path', type=str, required=True,
                        help='Dir with train/val/test .npy from embedding extraction')
    parser.add_argument('--mae_embedding_csv_path', type=str, required=True,
                        help='Dir with train/val/test .csv used during extraction (e.g. dataset1/)')

    # CSV for the split to evaluate
    parser.add_argument('--csv_path', type=str, required=True,
                        help='Dir with kfold split CSVs (e.g. kfold5/fold0)')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'],
                        help='Which split to evaluate on')

    # PPI data
    parser.add_argument('--ppi_path', type=str,
                        default='/ictstr01/groups/labs/marr/qscd01/datasets/SingleCellImagesDataset/opencell/opencell_metadata_raw/protein-protein-interactions/opencell-protein-interactions.csv')
    parser.add_argument('--abundance_path', type=str,
                        default='/ictstr01/groups/labs/marr/qscd01/datasets/SingleCellImagesDataset/opencell/opencell_metadata_raw/protein-abundance/opencell-protein-abundance.csv')

    # Filtering / evaluation
    parser.add_argument('--pval_threshold',          type=float, default=5.0)
    parser.add_argument('--enrichment_threshold',    type=float, default=2.5)
    parser.add_argument('--stoichiometry_threshold', type=float, default=0.05)
    parser.add_argument('--n_abundance_buckets',     type=int,   default=10)
    parser.add_argument('--n_negatives_per_positive',type=int,   default=1)
    parser.add_argument('--seed',                    type=int,   default=42)
    parser.add_argument('--batch_size',              type=int,   default=512)
    parser.add_argument('--device',                  type=str,   default='cuda')

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # --- Step 1: Build model and load checkpoint ---
    print("\n" + "=" * 60)
    print("Step 1: Loading model")
    print("=" * 60)
    config = load_config(args.config)
    model  = build_model(config)
    model  = load_checkpoint(model, args.checkpoint, device)
    model  = model.to(device)
    model.eval()

    # --- Step 2: Build combined embedding lookup ---
    print("\n" + "=" * 60)
    print("Step 2: Building combined embedding lookup")
    print("=" * 60)
    combined_emb, lookup = build_combined_lookup(
        args.mae_embedding_path, args.mae_embedding_csv_path
    )

    # --- Step 3: Extract projection-head embeddings per cell ---
    print("\n" + "=" * 60)
    print(f"Step 3: Extracting projection embeddings ({args.split} split)")
    print("=" * 60)
    csv_file = os.path.join(args.csv_path, f'{args.split}.csv')
    print(f"  CSV: {csv_file}")
    cell_embeddings, protein_names = extract_projection_embeddings(
        model, csv_file, combined_emb, lookup,
        batch_size=args.batch_size, device=device
    )
    print(f"  Cell embeddings: {cell_embeddings.shape}")

    # --- Step 4: Aggregate per protein ---
    print("\n" + "=" * 60)
    print("Step 4: Aggregating to protein-level embeddings")
    print("=" * 60)
    protein_embeddings, protein_cell_counts = aggregate_protein_embeddings(
        cell_embeddings, protein_names
    )

    emb_path = os.path.join(args.output_dir, f'{args.split}_protein_embeddings.npz')
    np.savez(
        emb_path,
        embeddings   = np.array(list(protein_embeddings.values())),
        protein_names= list(protein_embeddings.keys()),
        cell_counts  = np.array(list(protein_cell_counts.values()))
    )
    print(f"  Saved protein embeddings to {emb_path}")

    # --- Step 5: Load PPI data ---
    print("\n" + "=" * 60)
    print("Step 5: Loading PPI data")
    print("=" * 60)
    ppi_df = load_ppi_data(
        args.ppi_path,
        pval_threshold=args.pval_threshold,
        enrichment_threshold=args.enrichment_threshold,
        stoichiometry_threshold=args.stoichiometry_threshold
    )
    abundance_dict = load_abundance_data(args.abundance_path)

    available_proteins = list(protein_embeddings.keys())
    bucket_assignments, bucket_proteins = assign_abundance_buckets(
        available_proteins, abundance_dict, n_buckets=args.n_abundance_buckets
    )

    # --- Step 6: Build pairs ---
    print("\n" + "=" * 60)
    print("Step 6: Building positive and negative pairs")
    print("=" * 60)
    positive_pairs = build_positive_pairs(ppi_df, available_proteins)
    if len(positive_pairs) == 0:
        print("ERROR: No positive pairs found — check that proteins in PPI data "
              "match the evaluated split.")
        return

    negative_pairs = build_negative_pairs(
        positive_pairs, bucket_assignments, bucket_proteins,
        n_negatives_per_positive=args.n_negatives_per_positive,
        seed=args.seed
    )

    # --- Step 7: Compute similarities and evaluate ---
    print("\n" + "=" * 60)
    print("Step 7: Evaluating PPI prediction")
    print("=" * 60)
    positive_sims = compute_similarities(positive_pairs, protein_embeddings)
    negative_sims = compute_similarities(negative_pairs, protein_embeddings)

    print(f"  Positive mean sim: {np.mean(positive_sims):.4f} ± {np.std(positive_sims):.4f}")
    print(f"  Negative mean sim: {np.mean(negative_sims):.4f} ± {np.std(negative_sims):.4f}")

    metrics = evaluate_ppi_prediction(positive_sims, negative_sims)

    print(f"\n{'=' * 40}")
    print("RESULTS")
    print(f"{'=' * 40}")
    print(f"ROC-AUC:           {metrics['roc_auc']:.4f}")
    print(f"Average Precision: {metrics['average_precision']:.4f}")
    print(f"Accuracy (thr=0):  {metrics['accuracy']:.4f}")
    print(f"Positive pairs:    {metrics['n_positive_pairs']}")
    print(f"Negative pairs:    {metrics['n_negative_pairs']}")

    # --- Step 8: Save results ---
    metrics_path = os.path.join(args.output_dir, f'{args.split}_ppi_metrics.json')
    metrics_to_save = {k: v for k, v in metrics.items() if k not in ('fpr', 'tpr')}
    metrics_to_save['config'] = {
        'checkpoint':            args.checkpoint,
        'split':                 args.split,
        'pval_threshold':        args.pval_threshold,
        'enrichment_threshold':  args.enrichment_threshold,
        'stoichiometry_threshold': args.stoichiometry_threshold,
        'n_positive_pairs':      len(positive_pairs),
        'n_negative_pairs':      len(negative_pairs),
    }
    with open(metrics_path, 'w') as f:
        json.dump(metrics_to_save, f, indent=2)
    print(f"\nSaved metrics to {metrics_path}")

    plot_path = os.path.join(args.output_dir, f'{args.split}_ppi_evaluation.png')
    plot_results(metrics, plot_path)

    pairs_path = os.path.join(args.output_dir, f'{args.split}_pairs.npz')
    np.savez(
        pairs_path,
        positive_pairs       = np.array(positive_pairs),
        negative_pairs       = np.array(negative_pairs),
        positive_similarities= positive_sims,
        negative_similarities= negative_sims
    )
    print(f"Saved pairs to {pairs_path}")

    print("\n" + "=" * 60)
    print("PPI Embedding-Mode Evaluation Complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
