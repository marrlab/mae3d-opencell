#!/usr/bin/env python3
"""
Create 5-fold cross-validation splits at the protein level.

Each fold k:  test = fold k,  val = fold (k+1)%5,  train = remaining 3 folds

Usage:
    python src/create_kfold_splits.py \
        --data_dir /path/to/dataset1 \
        --output_dir /path/to/kfold \
        --n_folds 5 \
        --seed 42
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True,
                        help='Directory with train.csv, val.csv, test.csv')
    parser.add_argument('--output_dir', required=True,
                        help='Root output directory; fold dirs created inside')
    parser.add_argument('--n_folds', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    # Load and combine all splits
    dfs = []
    for split in ('train', 'val', 'test'):
        df = pd.read_csv(data_dir / f'{split}.csv')
        dfs.append(df)
    all_df = pd.concat(dfs, ignore_index=True)

    # Protein-level split to avoid leakage
    proteins = np.array(sorted(all_df['file_gene_symbol'].unique()))
    rng = np.random.default_rng(args.seed)
    rng.shuffle(proteins)

    fold_assignments = np.array_split(proteins, args.n_folds)

    print(f"Total proteins: {len(proteins)}, Total cells: {len(all_df)}")

    for k in range(args.n_folds):
        test_proteins  = set(fold_assignments[k])
        val_proteins   = set(fold_assignments[(k + 1) % args.n_folds])
        train_proteins = set(proteins) - test_proteins - val_proteins

        test_df  = all_df[all_df['file_gene_symbol'].isin(test_proteins)]
        val_df   = all_df[all_df['file_gene_symbol'].isin(val_proteins)]
        train_df = all_df[all_df['file_gene_symbol'].isin(train_proteins)]

        fold_dir = output_dir / f'fold{k}'
        fold_dir.mkdir(parents=True, exist_ok=True)

        train_df.to_csv(fold_dir / 'train.csv', index=False)
        val_df.to_csv(fold_dir / 'val.csv', index=False)
        test_df.to_csv(fold_dir / 'test.csv', index=False)

        print(f"Fold {k}: train={len(train_proteins)} proteins ({len(train_df)} cells) | "
              f"val={len(val_proteins)} proteins ({len(val_df)} cells) | "
              f"test={len(test_proteins)} proteins ({len(test_df)} cells)")

    print(f"\nDone. Fold dirs: {output_dir}/fold{{0..{args.n_folds-1}}}/")


if __name__ == '__main__':
    main()
