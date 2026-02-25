"""
Create per-fold train.csv / val.csv files for WTC-11 k-fold training.

Reads wtc11_cells_info_5fold.csv (produced by create_wtc_folds.py) and
writes one pair of CSV files per fold:

    output_dir/fold{k}/train.csv  — all cells NOT in fold k
    output_dir/fold{k}/val.csv    — all cells IN fold k

Each CSV contains an ``image_path`` column (= file_path from the info CSV)
so that the standard trainers can load the data directly.

Usage
-----
    python src/wtc/create_wtc_kfold_csvs.py \
        --info_path /path/to/.../wtc11/wtc11_cells_info_5fold.csv \
        --output_dir /path/to/.../wtc11/kfold5 \
        --n_folds 5
"""

import argparse
import os

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create per-fold train/val CSVs for WTC-11 k-fold training"
    )
    parser.add_argument(
        "--info_path",
        default=(
            "/path/to/groups/labs/lab/user/datasets/"
            "SingleCellImagesDataset/wtc11/wtc11_cells_info_5fold.csv"
        ),
        help="Path to wtc11_cells_info_5fold.csv (output of create_wtc_folds.py)",
    )
    parser.add_argument(
        "--output_dir",
        default=(
            "/path/to/groups/labs/lab/user/datasets/"
            "SingleCellImagesDataset/wtc11/kfold5"
        ),
        help="Root directory for per-fold CSV files",
    )
    parser.add_argument("--n_folds", type=int, default=5)
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Reading {args.info_path} ...")
    df = pd.read_csv(args.info_path, low_memory=False)

    if "fold" not in df.columns:
        raise ValueError(
            "'fold' column not found. "
            "Run create_wtc_folds.py first to assign fold labels."
        )

    print(f"  {len(df)} cells, folds: {sorted(df['fold'].unique())}")

    for fold_k in range(args.n_folds):
        fold_dir = os.path.join(args.output_dir, f"fold{fold_k}")
        os.makedirs(fold_dir, exist_ok=True)

        train_df = df[df["fold"] != fold_k].copy()
        val_df   = df[df["fold"] == fold_k].copy()

        # Rename file_path → image_path for trainer compatibility
        train_df = train_df.rename(columns={"file_path": "image_path"})
        val_df   = val_df.rename(columns={"file_path": "image_path"})

        # Keep all columns (trainers only require image_path; others are ignored)
        train_csv = os.path.join(fold_dir, "train.csv")
        val_csv   = os.path.join(fold_dir, "val.csv")

        train_df.to_csv(train_csv, index=False)
        val_df.to_csv(val_csv,   index=False)

        n_fovs_train = train_df["fov_id"].nunique() if "fov_id" in train_df.columns else "?"
        n_fovs_val   = val_df["fov_id"].nunique()   if "fov_id" in val_df.columns   else "?"

        print(
            f"  fold {fold_k}: "
            f"train={len(train_df)} cells ({n_fovs_train} FOVs) → {train_csv}  |  "
            f"val={len(val_df)} cells ({n_fovs_val} FOVs) → {val_csv}"
        )

    print(f"\nDone. Fold CSV files written to: {args.output_dir}")


if __name__ == "__main__":
    main()
