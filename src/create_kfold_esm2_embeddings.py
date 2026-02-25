"""
Create fold-specific ESM2 embedding files from the global embedding arrays.

The OpenCellDataset loads ESM2 embeddings positionally (row i of the .npy
file corresponds to row i of the CSV).  K-fold splits re-partition the
dataset into different train/val CSVs, so the global train.npy / val.npy
cannot be used directly — the row order no longer matches.

This script builds a lookup table (image_path → embedding vector) from the
global CSVs and embedding files, then writes per-fold train.npy / val.npy
(and optionally test.npy) whose rows are aligned with the corresponding
fold CSV files.

Usage:
    python src/create_kfold_esm2_embeddings.py \
        --global_csv_dir  /path/to/dataset1 \
        --global_esm2_dir /path/to/esm2_embeddings \
        --kfold_dir       /path/to/kfold5 \
        --output_dir      /path/to/esm2_embeddings_kfold5 \
        --n_folds 5

Arguments:
    --global_csv_dir   Directory containing the global train.csv / val.csv
                       (and optionally test.csv) used when the ESM2 embeddings
                       were originally computed.
    --global_esm2_dir  Directory containing train.npy / val.npy
                       (and optionally test.npy) from the global run.
    --kfold_dir        Directory produced by create_kfold_splits.py, containing
                       sub-directories fold0/ … fold{N-1}/ each with
                       train.csv and val.csv.
    --output_dir       Root directory for output. Fold embeddings are written to
                       {output_dir}/fold{i}/train.npy  etc.
    --n_folds          Number of folds (default: 5).
    --splits           Comma-separated list of splits to process
                       (default: train,val,test).
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd


def build_lookup(csv_dir, emb_dir, splits):
    """
    Build a dict mapping image_path → embedding vector.

    Reads each split CSV and the matching .npy file, then combines them
    into one lookup covering all splits.

    Args:
        csv_dir: Directory with {split}.csv files.
        emb_dir: Directory with {split}.npy files.
        splits:  List of split names to load (e.g. ['train', 'val', 'test']).

    Returns:
        lookup: dict {image_path (str) → np.ndarray of shape (embed_dim,)}
    """
    lookup = {}

    for split in splits:
        csv_path = os.path.join(csv_dir, f'{split}.csv')
        npy_path = os.path.join(emb_dir, f'{split}.npy')

        if not os.path.exists(csv_path):
            print(f'  [skip] {csv_path} not found')
            continue
        if not os.path.exists(npy_path):
            print(f'  [skip] {npy_path} not found')
            continue

        df = pd.read_csv(csv_path)
        embeddings = np.load(npy_path)

        if len(df) != len(embeddings):
            raise ValueError(
                f'Row count mismatch for split "{split}": '
                f'CSV has {len(df)} rows but {npy_path} has {len(embeddings)} rows.'
            )

        for idx, row in df.iterrows():
            image_path = row['image_path']
            local_idx = df.index.get_loc(idx)
            if image_path in lookup:
                raise ValueError(
                    f'Duplicate image_path detected: {image_path}. '
                    f'Each image should appear in exactly one global split.'
                )
            lookup[image_path] = embeddings[local_idx]

        print(f'  Loaded {len(df):>6d} embeddings from {split}.npy  (dim={embeddings.shape[1]})')

    return lookup


def reindex_split(fold_csv_path, lookup, split_name):
    """
    Reindex embeddings for one fold split.

    Args:
        fold_csv_path: Path to the fold's CSV file (e.g. fold0/train.csv).
        lookup:        dict {image_path → embedding vector}.
        split_name:    For logging only ('train', 'val', etc.).

    Returns:
        np.ndarray of shape (N, embed_dim) aligned with the fold CSV rows,
        or None if the CSV does not exist.
    """
    if not os.path.exists(fold_csv_path):
        return None

    df = pd.read_csv(fold_csv_path)
    n = len(df)
    missing = []
    embeddings = []

    for _, row in df.iterrows():
        image_path = row['image_path']
        if image_path not in lookup:
            missing.append(image_path)
        else:
            embeddings.append(lookup[image_path])

    if missing:
        raise KeyError(
            f'{len(missing)} image_path(s) from fold {split_name} not found in the '
            f'global lookup.  First missing entry:\n  {missing[0]}\n'
            f'Make sure --global_csv_dir covers all samples used in the folds.'
        )

    result = np.stack(embeddings, axis=0)
    assert result.shape[0] == n, f'Shape mismatch: expected {n} rows, got {result.shape[0]}'
    return result


def main():
    parser = argparse.ArgumentParser(
        description='Create fold-specific ESM2 embedding files for k-fold training.'
    )
    parser.add_argument('--global_csv_dir', type=str, required=True,
                        help='Directory with the global train.csv / val.csv / test.csv')
    parser.add_argument('--global_esm2_dir', type=str, required=True,
                        help='Directory with the global train.npy / val.npy / test.npy')
    parser.add_argument('--kfold_dir', type=str, required=True,
                        help='Directory produced by create_kfold_splits.py (contains fold0/ … foldN-1/)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Root output directory; fold embeddings go into {output_dir}/fold{i}/')
    parser.add_argument('--n_folds', type=int, default=5,
                        help='Number of folds (default: 5)')
    parser.add_argument('--splits', type=str, default='train,val,test',
                        help='Comma-separated splits to process (default: train,val,test)')
    args = parser.parse_args()

    splits = [s.strip() for s in args.splits.split(',')]

    print('=' * 60)
    print('Creating fold-specific ESM2 embeddings')
    print('=' * 60)
    print(f'Global CSV dir:  {args.global_csv_dir}')
    print(f'Global ESM2 dir: {args.global_esm2_dir}')
    print(f'K-fold dir:      {args.kfold_dir}')
    print(f'Output dir:      {args.output_dir}')
    print(f'Folds:           {args.n_folds}')
    print(f'Splits:          {splits}')
    print('=' * 60)

    # ------------------------------------------------------------------
    # 1.  Build global lookup: image_path → embedding vector
    # ------------------------------------------------------------------
    print('\nBuilding global lookup table...')
    lookup = build_lookup(args.global_csv_dir, args.global_esm2_dir, splits)
    print(f'  Total entries in lookup: {len(lookup)}')

    # ------------------------------------------------------------------
    # 2.  Process each fold
    # ------------------------------------------------------------------
    for fold in range(args.n_folds):
        fold_csv_dir = os.path.join(args.kfold_dir, f'fold{fold}')
        fold_out_dir = os.path.join(args.output_dir, f'fold{fold}')
        os.makedirs(fold_out_dir, exist_ok=True)

        print(f'\nFold {fold}  ({fold_csv_dir})')

        for split in splits:
            fold_csv_path = os.path.join(fold_csv_dir, f'{split}.csv')
            out_path = os.path.join(fold_out_dir, f'{split}.npy')

            emb = reindex_split(fold_csv_path, lookup, f'{fold}/{split}')
            if emb is None:
                print(f'  [{split:5s}] CSV not found — skipped')
                continue

            np.save(out_path, emb)
            print(f'  [{split:5s}] {emb.shape[0]:>6d} samples  dim={emb.shape[1]}  → {out_path}')

    print('\n' + '=' * 60)
    print('Done.')
    print('=' * 60)


if __name__ == '__main__':
    main()
