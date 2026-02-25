#!/usr/bin/env python3
"""
Create per-fold SubCell and DINO4Cell embedding .npy files for k-fold evaluation.

The global train/val/test embeddings are aligned with the dataset1 CSVs.
This script builds an image_path → embedding lookup from those files and
re-exports per-fold train/val/test .npy files aligned with the kfold CSVs.

Usage:
    python src/create_subcell_kfold_embeddings.py

Outputs (one .npy per split per fold):
    <output_base>/subcell_kfold5/fold{0-4}/{train,val,test}.npy       (1536-dim)
    <output_base>/dino4cells_wtc_kfold5/fold{0-4}/{train,val,test}.npy (768-dim)
"""

import argparse
import os
import numpy as np
import pandas as pd
from pathlib import Path

# ============================================================================
# Paths – edit here if needed
# ============================================================================
GLOBAL_CSV_DIR   = "/ictstr01/groups/labs/marr/qscd01/datasets/SingleCellImagesDataset/opencell/opencell_dataset/single_cells/metadata/dataset1"
KFOLD_CSV_BASE   = "/ictstr01/groups/labs/marr/qscd01/datasets/SingleCellImagesDataset/opencell/opencell_dataset/single_cells/metadata/kfold5"
N_FOLDS          = 5
SPLITS           = ["train", "val", "test"]

SOURCES = {
    "subcell_pca384": {
        "emb_dir":  "/ictstr01/groups/labs/marr/qscd01/datasets/SingleCellImagesDataset/opencell/subcell_pca384",
        "out_dir":  "/ictstr01/groups/labs/marr/qscd01/datasets/SingleCellImagesDataset/opencell/subcell_pca384_kfold5",
    },
    "dino4cell_pca384": {
        "emb_dir":  "/ictstr01/groups/labs/marr/qscd01/datasets/SingleCellImagesDataset/opencell/dino4cell_pca384",
        "out_dir":  "/ictstr01/groups/labs/marr/qscd01/datasets/SingleCellImagesDataset/opencell/dino4cell_pca384_kfold5",
    },
}
# ============================================================================


def build_combined(global_csv_dir, emb_dir):
    """
    Concatenate global train/val/test CSVs and their embeddings in order.
    Returns:
        combined_df   : pd.DataFrame with all rows, ordered train→val→test
        combined_emb  : np.ndarray  [N, D]
        lookup        : dict  image_path → row index in combined_emb
    """
    dfs  = []
    embs = []

    for split in SPLITS:
        csv_path = os.path.join(global_csv_dir, f"{split}.csv")
        npy_path = os.path.join(emb_dir, f"{split}.npy")

        df  = pd.read_csv(csv_path)
        emb = np.load(npy_path)

        assert len(df) == len(emb), \
            f"CSV/embedding size mismatch for {split}: {len(df)} vs {len(emb)}"

        dfs.append(df)
        embs.append(emb)
        print(f"  Loaded {split}: {len(df)} rows, embedding shape {emb.shape}")

    combined_df  = pd.concat(dfs, ignore_index=True)
    combined_emb = np.concatenate(embs, axis=0)

    assert len(combined_df) == combined_emb.shape[0]
    print(f"  Combined: {combined_emb.shape}")

    lookup = {row["image_path"]: i for i, row in combined_df.iterrows()}
    return combined_df, combined_emb, lookup


def extract_fold_split(kfold_csv_dir, fold, split, combined_emb, lookup):
    """
    For a given fold + split, look up each image_path and return the embedding array.
    """
    csv_path = os.path.join(kfold_csv_dir, f"fold{fold}", f"{split}.csv")
    df = pd.read_csv(csv_path)

    missing = [p for p in df["image_path"] if p not in lookup]
    if missing:
        raise KeyError(f"fold{fold}/{split}: {len(missing)} image_paths not found in global lookup")

    indices = [lookup[p] for p in df["image_path"]]
    return combined_emb[indices]


def main():
    parser = argparse.ArgumentParser(description="Create per-fold SubCell/DINO4Cell embeddings")
    parser.add_argument("--n_folds", type=int, default=N_FOLDS)
    args = parser.parse_args()

    for name, cfg in SOURCES.items():
        print(f"\n{'='*60}")
        print(f"Processing: {name}")
        print(f"  Source embeddings: {cfg['emb_dir']}")
        print(f"  Output directory:  {cfg['out_dir']}")
        print(f"{'='*60}")

        print("Building combined lookup …")
        _, combined_emb, lookup = build_combined(GLOBAL_CSV_DIR, cfg["emb_dir"])

        for fold in range(args.n_folds):
            fold_dir = os.path.join(cfg["out_dir"], f"fold{fold}")
            os.makedirs(fold_dir, exist_ok=True)

            for split in SPLITS:
                out_path = os.path.join(fold_dir, f"{split}.npy")
                if os.path.exists(out_path):
                    print(f"  fold{fold}/{split}: already exists, skipping")
                    continue

                emb = extract_fold_split(KFOLD_CSV_BASE, fold, split, combined_emb, lookup)
                np.save(out_path, emb)
                print(f"  fold{fold}/{split}: saved {emb.shape} → {out_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
