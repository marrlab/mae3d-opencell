"""
Create stratified 5-fold cross-validation splits for the WTC-11 dataset.

Split strategy
--------------
- Unit of split : FOV (field of view), not individual cells.
  Cells from the same FOV share imaging conditions and must not appear
  in both train and test of the same fold.
- Stratification : structure_name (25 protein classes), so every fold
  has the same protein-class distribution.
- Each FOV belongs to exactly one protein class, so FOV-level
  stratification by structure_name is clean and unambiguous.

Output
------
Adds a `fold` column (0–4) to wtc11_cells_info.csv and saves the result
as wtc11_cells_info_5fold.csv.  All cells belonging to a given FOV are
assigned the same fold.

Fold interpretation
-------------------
For k-fold training:  train on folds != k, evaluate on fold == k.
A typical 5-fold run trains 5 models; the held-out fold acts as the
test set.  There is no separate validation set by default, but you can
carve one out of the training folds if needed.

Usage
-----
    python src/wtc/create_wtc_folds.py \
        --info_path   /ictstr01/.../wtc11/wtc11_cells_info.csv \
        --output_path /ictstr01/.../wtc11/wtc11_cells_info_5fold.csv \
        --n_folds 5 \
        --seed 42
"""

import argparse

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


def parse_args():
    parser = argparse.ArgumentParser(description="Create WTC-11 5-fold splits")
    parser.add_argument(
        "--info_path",
        default="/ictstr01/groups/labs/marr/qscd01/datasets/SingleCellImagesDataset/wtc11/wtc11_cells_info.csv",
    )
    parser.add_argument(
        "--output_path",
        default="/ictstr01/groups/labs/marr/qscd01/datasets/SingleCellImagesDataset/wtc11/wtc11_cells_info_5fold.csv",
    )
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--seed",    type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Reading {args.info_path} ...")
    df = pd.read_csv(args.info_path, low_memory=False)
    print(f"  {len(df)} cells, {df['fov_id'].nunique()} unique FOVs")

    # ------------------------------------------------------------------ #
    # Build one row per FOV with its structure_name for stratification    #
    # Each FOV belongs to exactly one protein class                       #
    # ------------------------------------------------------------------ #
    fov_df = (
        df.groupby("fov_id")["structure_name"]
        .first()
        .reset_index()
        .rename(columns={"structure_name": "structure_name"})
    )
    print(f"  FOVs per protein class:\n"
          f"{fov_df['structure_name'].value_counts().to_string()}\n")

    # ------------------------------------------------------------------ #
    # Stratified k-fold on FOVs                                           #
    # ------------------------------------------------------------------ #
    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
    fov_df["fold"] = -1

    for fold_idx, (_, test_idx) in enumerate(
        skf.split(fov_df["fov_id"], fov_df["structure_name"])
    ):
        fov_df.loc[test_idx, "fold"] = fold_idx

    assert (fov_df["fold"] == -1).sum() == 0, "Some FOVs were not assigned a fold"

    # ------------------------------------------------------------------ #
    # Map fold assignments back to individual cells                       #
    # ------------------------------------------------------------------ #
    fov_to_fold = fov_df.set_index("fov_id")["fold"].to_dict()
    df["fold"] = df["fov_id"].map(fov_to_fold).astype(int)

    # ------------------------------------------------------------------ #
    # Verify: no FOV appears in more than one fold                        #
    # ------------------------------------------------------------------ #
    fov_fold_counts = df.groupby("fov_id")["fold"].nunique()
    assert (fov_fold_counts > 1).sum() == 0, "FOV leakage detected!"

    # ------------------------------------------------------------------ #
    # Save                                                                 #
    # ------------------------------------------------------------------ #
    df.to_csv(args.output_path, index=False)
    print(f"Saved → {args.output_path}")

    # ------------------------------------------------------------------ #
    # Summary                                                              #
    # ------------------------------------------------------------------ #
    print("\n--- Cells per fold ---")
    print(df["fold"].value_counts().sort_index().to_string())

    print("\n--- FOVs per fold ---")
    print(df.groupby("fold")["fov_id"].nunique().to_string())

    print("\n--- structure_name distribution per fold (cells) ---")
    pivot = df.pivot_table(
        index="structure_name", columns="fold", values="cell_id", aggfunc="count"
    )
    print(pivot.to_string())

    print("\n--- cell_stage distribution per fold (cells) ---")
    pivot_stage = df.pivot_table(
        index="cell_stage", columns="fold", values="cell_id", aggfunc="count"
    )
    print(pivot_stage.to_string())


if __name__ == "__main__":
    main()
