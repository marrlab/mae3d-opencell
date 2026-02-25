#!/usr/bin/env python3
"""
Evaluate a trained PPIMetricSubCell model on a kfold split.

Works for both SubCell (1536-dim) and DINO4Cell (768-dim) embeddings
produced by create_subcell_kfold_embeddings.py.

Usage:
    python src/evaluate_ppi_subcell_kfold.py \\
        --config configs/opencell/opencell_ppi_subcell_kfold.yaml \\
        --checkpoint /path/to/ppi_subcell_kfold5/fold0/ckpts/checkpoint_0049.pth.tar \\
        --embedding_dir /path/to/subcell_kfold5/fold0 \\
        --csv_path /path/to/kfold5/fold0 \\
        --split test \\
        --output_dir /path/to/ppi_subcell_kfold5/fold0/eval_results
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))

from omegaconf import OmegaConf
from lib.models.ppi_metric_subcell import PPIMetricSubCell


# ============================================================================
# Helpers
# ============================================================================

def load_config(config_path):
    config = OmegaConf.load(config_path)
    OmegaConf.resolve(config)
    return config


def build_model(config, device):
    model = PPIMetricSubCell(
        embed_dim=config.embed_dim,
        proj_hidden_dim=config.proj_hidden_dim,
        proj_output_dim=config.proj_output_dim,
        proj_num_layers=config.proj_num_layers,
    )
    ckpt = torch.load(config._checkpoint, map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt)
    model.load_state_dict(state_dict, strict=True)
    model.to(device).eval()
    print(f"Loaded checkpoint (epoch {ckpt.get('epoch', 'unknown')})")
    return model


def aggregate_protein_embeddings(embedding_path, csv_path):
    """
    Mean-pool cell embeddings per protein using the kfold split CSV.
    Returns:
        protein_embeddings: dict {protein_name: np.ndarray [D]}
    """
    emb = np.load(embedding_path)
    df  = pd.read_csv(csv_path)
    assert len(df) == len(emb), f"CSV/embedding size mismatch: {len(df)} vs {len(emb)}"

    if "file_gene_symbol" in df.columns:
        protein_col = "file_gene_symbol"
    elif "folder_protein" in df.columns:
        protein_col = "folder_protein"
    else:
        raise ValueError("CSV must contain 'file_gene_symbol' or 'folder_protein'")

    protein_to_embs = defaultdict(list)
    for idx, row in df.iterrows():
        protein_to_embs[row[protein_col]].append(emb[idx])

    protein_embeddings = {
        prot: np.mean(np.stack(embs), axis=0)
        for prot, embs in protein_to_embs.items()
    }
    print(f"Aggregated {len(emb)} cells into {len(protein_embeddings)} proteins")
    return protein_embeddings


def load_ppi_data(ppi_path, pval_threshold=5.0, enrichment_threshold=2.5,
                  stoichiometry_threshold=0.05):
    df = pd.read_csv(ppi_path)
    filtered = df[
        (df["pval"] > pval_threshold) &
        (df["enrichment"] > enrichment_threshold) &
        (df["interaction_stoichiometry"] > stoichiometry_threshold)
    ]
    print(f"PPI: {len(df)} total → {len(filtered)} after filtering")
    return filtered


def load_abundance_data(abundance_path):
    df = pd.read_csv(abundance_path)
    result = {}
    for _, row in df.iterrows():
        gene = row["gene_name"]
        if pd.notna(row.get("hek_protein_conc_nm")):
            result[gene] = row["hek_protein_conc_nm"]
        elif pd.notna(row.get("hek_rna_tpm")):
            result[gene] = row["hek_rna_tpm"]
    return result


def assign_abundance_buckets(proteins, abundance_dict, n_buckets=10):
    with_abundance = [(p, abundance_dict[p]) for p in proteins if p in abundance_dict]
    without_abundance = [p for p in proteins if p not in abundance_dict]

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


def build_pairs(ppi_df, available_proteins, bucket_assignments, bucket_proteins,
                n_negatives_per_positive=1, seed=42):
    np.random.seed(seed)
    all_proteins  = list(available_proteins)
    positive_set  = set()
    positive_pairs = []

    for _, row in ppi_df.iterrows():
        t, i = row["target_gene_name"], row["interactor_gene_name"]
        if t in available_proteins and i in available_proteins:
            pair = tuple(sorted([t, i]))
            if pair not in positive_set:
                positive_set.add(pair)
                positive_pairs.append(pair)

    # Negative pairs
    negative_pairs = []
    for prot1, prot2 in positive_pairs:
        b1 = bucket_assignments.get(prot1, -1)
        b2 = bucket_assignments.get(prot2, -1)
        cands1 = bucket_proteins.get(b1, all_proteins)
        cands2 = bucket_proteins.get(b2, all_proteins)

        for _ in range(n_negatives_per_positive * 10):
            n1 = np.random.choice(cands1)
            n2 = np.random.choice(cands2)
            if n1 == n2:
                continue
            neg_pair = tuple(sorted([n1, n2]))
            if neg_pair not in positive_set and neg_pair not in negative_pairs:
                negative_pairs.append(neg_pair)
                break

    target = len(positive_pairs) * n_negatives_per_positive
    while len(negative_pairs) < target:
        n1, n2 = np.random.choice(all_proteins, 2, replace=False)
        neg_pair = tuple(sorted([n1, n2]))
        if neg_pair not in positive_set and neg_pair not in negative_pairs:
            negative_pairs.append(neg_pair)

    negative_pairs = negative_pairs[:target]
    print(f"Built {len(positive_pairs)} positive and {len(negative_pairs)} negative pairs")
    return positive_pairs, negative_pairs


@torch.no_grad()
def compute_similarities(model, pairs, protein_embeddings, device, batch_size=512):
    all_prots = list(protein_embeddings.keys())
    prot_idx  = {p: i for i, p in enumerate(all_prots)}
    emb_matrix = torch.tensor(
        np.stack([protein_embeddings[p] for p in all_prots]), dtype=torch.float32
    ).to(device)

    # Project all proteins at once
    proj_all = []
    for start in range(0, len(all_prots), batch_size):
        batch = emb_matrix[start:start + batch_size]
        proj_all.append(model.forward_embedding(batch).cpu().numpy())
    proj_matrix = np.concatenate(proj_all, axis=0)
    proj_dict = {p: proj_matrix[prot_idx[p]] for p in all_prots}

    sims = []
    for p1, p2 in pairs:
        z1 = proj_dict[p1]
        z2 = proj_dict[p2]
        sim = float(np.dot(z1, z2) / (np.linalg.norm(z1) * np.linalg.norm(z2) + 1e-8))
        sims.append(sim)
    return np.array(sims)


def evaluate_ppi(pos_sims, neg_sims):
    labels = np.concatenate([np.ones(len(pos_sims)), np.zeros(len(neg_sims))])
    scores = np.concatenate([pos_sims, neg_sims])
    roc_auc = roc_auc_score(labels, scores)
    ap      = average_precision_score(labels, scores)
    fpr, tpr, _ = roc_curve(labels, scores)
    return {
        "roc_auc":            float(roc_auc),
        "average_precision":  float(ap),
        "n_positive_pairs":   int(len(pos_sims)),
        "n_negative_pairs":   int(len(neg_sims)),
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
    }


def plot_results(metrics, output_path):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(metrics["fpr"], metrics["tpr"],
            label=f"ROC (AUC = {metrics['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("PPI ROC Curve")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Evaluate PPIMetricSubCell kfold model")
    parser.add_argument("--config",        type=str, required=True)
    parser.add_argument("--checkpoint",    type=str, required=True)
    parser.add_argument("--embedding_dir", type=str, required=True,
                        help="Folder containing {train,val,test}.npy for this fold")
    parser.add_argument("--csv_path",      type=str, required=True,
                        help="Kfold fold dir containing {train,val,test}.csv")
    parser.add_argument("--split",         type=str, default="test",
                        choices=["train", "val", "test"])
    parser.add_argument("--output_dir",    type=str, required=True)
    parser.add_argument("--batch_size",    type=int, default=512)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    config = load_config(args.config)
    # Inject checkpoint path so build_model can access it
    OmegaConf.update(config, "_checkpoint", args.checkpoint, merge=True)

    # ---- model ----
    model = build_model(config, device)

    # ---- embeddings + protein aggregation ----
    embedding_path = os.path.join(args.embedding_dir, f"{args.split}.npy")
    csv_file       = os.path.join(args.csv_path,      f"{args.split}.csv")
    protein_embeddings = aggregate_protein_embeddings(embedding_path, csv_file)
    available_proteins = set(protein_embeddings.keys())

    # ---- PPI pairs ----
    ppi_df = load_ppi_data(
        config.ppi_csv_path,
        pval_threshold=config.pval_threshold,
        enrichment_threshold=config.enrichment_threshold,
        stoichiometry_threshold=config.stoichiometry_threshold,
    )
    abundance_dict = load_abundance_data(config.abundance_csv_path)
    bucket_assignments, bucket_proteins = assign_abundance_buckets(
        list(available_proteins), abundance_dict, config.n_abundance_buckets
    )
    positive_pairs, negative_pairs = build_pairs(
        ppi_df, available_proteins, bucket_assignments, bucket_proteins,
        n_negatives_per_positive=config.n_negatives_per_positive,
    )

    if len(positive_pairs) == 0:
        print("WARNING: No positive pairs found for this split. Skipping evaluation.")
        return

    # ---- compute similarities ----
    pos_sims = compute_similarities(model, positive_pairs, protein_embeddings,
                                    device, args.batch_size)
    neg_sims = compute_similarities(model, negative_pairs, protein_embeddings,
                                    device, args.batch_size)

    # ---- metrics ----
    metrics = evaluate_ppi(pos_sims, neg_sims)
    print(f"\nRESULTS ON {args.split.upper()} SET")
    print(f"  ROC-AUC:           {metrics['roc_auc']:.4f}")
    print(f"  Average Precision: {metrics['average_precision']:.4f}")
    print(f"  Positive pairs:    {metrics['n_positive_pairs']}")
    print(f"  Negative pairs:    {metrics['n_negative_pairs']}")

    # ---- save ----
    to_save = {k: v for k, v in metrics.items() if k not in ("fpr", "tpr")}
    to_save["config"] = {
        "checkpoint": args.checkpoint,
        "split":      args.split,
        "config":     args.config,
    }
    json_path = os.path.join(args.output_dir, f"{args.split}_ppi_metrics.json")
    with open(json_path, "w") as f:
        json.dump(to_save, f, indent=2)
    print(f"Metrics saved to {json_path}")

    plot_path = os.path.join(args.output_dir, f"{args.split}_roc_curve.png")
    plot_results(metrics, plot_path)
    print(f"ROC curve saved to {plot_path}")


if __name__ == "__main__":
    main()
