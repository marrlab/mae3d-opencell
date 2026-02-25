#!/usr/bin/env python3
"""
Preprocess WTC-11 SubCell / DINO4Cell embeddings with PCA.

Unlike the OpenCell version (which splits into train/val/test .npy files),
WTC-11 embeddings are stored as a single file row-aligned with
wtc11_cells_info_5fold.csv.

PCA is fit on the training split of fold 0 (rows where fold != 0, ~80 % of
data) to avoid data leakage, then applied to all 50 000 samples.

Usage
-----
    python src/preprocess_subcell_pca_wtc.py \
        --input_file  /path/to/wtc11_cells_info_5fold.npy \
        --info_csv    /path/to/wtc11_cells_info_5fold.csv \
        --output_file /path/to/wtc11_cells_info_5fold_pca384.npy \
        --target_dim  384
"""

import argparse
import json
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.decomposition import PCA


def main():
    parser = argparse.ArgumentParser(
        description='Apply PCA to WTC-11 single-file embeddings'
    )
    parser.add_argument('--input_file', type=str, required=True,
                        help='Path to the single .npy embedding file (N, D)')
    parser.add_argument('--info_csv', type=str, required=True,
                        help='Path to wtc11_cells_info_5fold.csv (row-aligned with input_file)')
    parser.add_argument('--output_file', type=str, required=True,
                        help='Output path for PCA-transformed .npy file')
    parser.add_argument('--target_dim', type=int, default=384,
                        help='Target PCA dimension (default: 384)')
    parser.add_argument('--fit_fold', type=int, default=0,
                        help='Hold-out fold for fitting: PCA is fit on rows where fold != fit_fold (default: 0)')
    parser.add_argument('--whiten', action='store_true',
                        help='Apply PCA whitening')
    args = parser.parse_args()

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Load embeddings ───────────────────────────────────────────────────────
    print(f"Loading embeddings from {args.input_file} ...")
    embeddings = np.load(args.input_file)
    print(f"  Shape: {embeddings.shape}")

    # ── Load fold info ────────────────────────────────────────────────────────
    print(f"Loading fold info from {args.info_csv} ...")
    df = pd.read_csv(args.info_csv, low_memory=False)
    assert len(df) == len(embeddings), (
        f"Row count mismatch: CSV has {len(df)} rows, npy has {len(embeddings)} rows"
    )
    print(f"  CSV shape: {df.shape}  |  fold distribution:\n{df['fold'].value_counts().sort_index()}")

    # ── Fit PCA on training samples of the reference fold ────────────────────
    train_mask = (df['fold'] != args.fit_fold).values
    train_embeddings = embeddings[train_mask]
    print(f"\nFitting PCA on {train_mask.sum()} samples (fold != {args.fit_fold}) ...")
    print(f"  {embeddings.shape[1]} → {args.target_dim} dimensions")

    pca = PCA(n_components=args.target_dim, whiten=args.whiten, random_state=42)
    pca.fit(train_embeddings)

    explained_var = pca.explained_variance_ratio_.sum()
    print(f"  Explained variance: {explained_var:.4f} ({explained_var * 100:.2f} %)")
    print(f"  Top 10 component variances: {pca.explained_variance_ratio_[:10]}")

    # ── Transform all 50 000 samples ─────────────────────────────────────────
    print(f"\nTransforming all {len(embeddings)} samples ...")
    embeddings_pca = pca.transform(embeddings).astype(np.float32)
    print(f"  Output shape: {embeddings_pca.shape}")

    # ── Save full transformed array ───────────────────────────────────────────
    np.save(output_path, embeddings_pca)
    print(f"\nSaved PCA embeddings: {output_path}")

    pca_model_path = output_path.parent / (output_path.stem + '_pca_model.joblib')
    joblib.dump(pca, pca_model_path)
    print(f"Saved PCA model:      {pca_model_path}")

    # ── Save per-fold train.npy / val.npy (positionally aligned with fold CSVs)
    # Fold CSVs preserve master CSV row order, so simple boolean slicing aligns.
    n_folds = int(df['fold'].max()) + 1
    print(f"\nSaving per-fold train/val splits ({n_folds} folds) ...")
    fold_dir_base = output_path.parent
    for k in range(n_folds):
        fold_dir = fold_dir_base / f'fold{k}'
        fold_dir.mkdir(parents=True, exist_ok=True)

        val_mask   = (df['fold'] == k).values
        train_mask_k = ~val_mask

        train_npy = embeddings_pca[train_mask_k]
        val_npy   = embeddings_pca[val_mask]

        np.save(fold_dir / 'train.npy', train_npy)
        np.save(fold_dir / 'val.npy',   val_npy)
        print(f"  fold{k}: train={len(train_npy)}, val={len(val_npy)} → {fold_dir}")

    metadata = {
        'input_file': args.input_file,
        'output_file': str(output_path),
        'info_csv': args.info_csv,
        'original_dim': int(embeddings.shape[1]),
        'target_dim': args.target_dim,
        'fit_fold': args.fit_fold,
        'n_fit_samples': int(train_mask.sum()),
        'n_total_samples': len(embeddings),
        'n_folds': n_folds,
        'whiten': args.whiten,
        'explained_variance_ratio': float(explained_var),
    }
    metadata_path = output_path.parent / (output_path.stem + '_pca_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata:       {metadata_path}")

    print(f"\n{'='*60}")
    print("PCA preprocessing complete!")
    print(f"  Original dim:       {embeddings.shape[1]}")
    print(f"  Target dim:         {args.target_dim}")
    print(f"  Explained variance: {explained_var * 100:.2f} %")
    print(f"  Output:             {output_path}")
    print(f"  Per-fold splits:    {fold_dir_base}/fold{{0..{n_folds-1}}}/{{train,val}}.npy")
    print('='*60)


if __name__ == '__main__':
    main()
