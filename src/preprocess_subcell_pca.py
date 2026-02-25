#!/usr/bin/env python3
"""
Preprocess SubCell embeddings using PCA for dimensionality reduction.

This script:
1. Fits PCA on training embeddings (1536 → target_dim)
2. Transforms train/val/test embeddings
3. Saves the transformed embeddings as new .npy files

Usage:
    python src/preprocess_subcell_pca.py \
        --input_dir /path/to/subcell/embeddings \
        --output_dir /path/to/output \
        --target_dim 384

This enables fair comparison with MAE3D (384-dim) since:
- PCA is fit only on training data (no data leakage)
- No additional learnable parameters during classification
- Both models use Linear(384, 17) classifier
"""

import os
import argparse
import numpy as np
from sklearn.decomposition import PCA
import joblib
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description='Apply PCA to SubCell embeddings')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Directory containing train.npy, val.npy, test.npy')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for PCA-transformed embeddings')
    parser.add_argument('--target_dim', type=int, default=384,
                        help='Target dimension after PCA (default: 384 to match MAE3D)')
    parser.add_argument('--whiten', action='store_true',
                        help='Apply whitening to PCA output')
    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load training embeddings
    print(f"Loading training embeddings from {args.input_dir}...")
    train_path = os.path.join(args.input_dir, 'train.npy')
    train_embeddings = np.load(train_path)
    print(f"  Train shape: {train_embeddings.shape}")

    # Fit PCA on training data
    print(f"\nFitting PCA: {train_embeddings.shape[1]} → {args.target_dim} dimensions...")
    pca = PCA(n_components=args.target_dim, whiten=args.whiten, random_state=42)
    train_pca = pca.fit_transform(train_embeddings)

    # Print explained variance
    explained_var = pca.explained_variance_ratio_.sum()
    print(f"  Explained variance ratio: {explained_var:.4f} ({explained_var*100:.2f}%)")
    print(f"  Top 10 component variances: {pca.explained_variance_ratio_[:10]}")

    # Save transformed training embeddings
    train_output_path = output_dir / 'train.npy'
    np.save(train_output_path, train_pca.astype(np.float32))
    print(f"  Saved: {train_output_path} (shape: {train_pca.shape})")

    # Transform and save validation embeddings
    print(f"\nTransforming validation embeddings...")
    val_path = os.path.join(args.input_dir, 'val.npy')
    val_embeddings = np.load(val_path)
    val_pca = pca.transform(val_embeddings)
    val_output_path = output_dir / 'val.npy'
    np.save(val_output_path, val_pca.astype(np.float32))
    print(f"  Val shape: {val_embeddings.shape} → {val_pca.shape}")
    print(f"  Saved: {val_output_path}")

    # Transform and save test embeddings
    print(f"\nTransforming test embeddings...")
    test_path = os.path.join(args.input_dir, 'test.npy')
    test_embeddings = np.load(test_path)
    test_pca = pca.transform(test_embeddings)
    test_output_path = output_dir / 'test.npy'
    np.save(test_output_path, test_pca.astype(np.float32))
    print(f"  Test shape: {test_embeddings.shape} → {test_pca.shape}")
    print(f"  Saved: {test_output_path}")

    # Save PCA model for reference
    pca_model_path = output_dir / 'pca_model.joblib'
    joblib.dump(pca, pca_model_path)
    print(f"\nPCA model saved: {pca_model_path}")

    # Save metadata
    metadata = {
        'input_dir': args.input_dir,
        'output_dir': str(output_dir),
        'original_dim': train_embeddings.shape[1],
        'target_dim': args.target_dim,
        'whiten': args.whiten,
        'explained_variance_ratio': float(explained_var),
        'n_train_samples': len(train_embeddings),
        'n_val_samples': len(val_embeddings),
        'n_test_samples': len(test_embeddings),
    }

    import json
    metadata_path = output_dir / 'pca_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved: {metadata_path}")

    print(f"\n" + "="*60)
    print("PCA preprocessing complete!")
    print("="*60)
    print(f"Original dimension: {train_embeddings.shape[1]}")
    print(f"Target dimension:   {args.target_dim}")
    print(f"Explained variance: {explained_var*100:.2f}%")
    print(f"Output directory:   {output_dir}")
    print("="*60)


if __name__ == '__main__':
    main()
