"""
Create fold-specific ESM2 embedding files for WTC-11 k-fold training.

Background
----------
The WTCDataset loads ESM2 embeddings positionally: row i of the .npy file
must correspond to row i of the CSV.  After k-fold splitting, each fold has
a different train.csv / val.csv with a different row order, so a single
global embedding file cannot be used directly.

This script builds a lookup table  ``structure_name → ESM2 vector``  and then
writes per-fold ``train.npy`` / ``val.npy`` files whose rows are row-aligned
with the corresponding fold CSV.

WTC vs. OpenCell difference
----------------------------
OpenCell uses ``image_path`` as the lookup key because every cell has its own
protein (one ESM2 vector per cell is possible, though typically per-gene).
WTC has only **25 unique proteins** (``structure_name``), so all ~50 K cells
with the same protein share **one** ESM2 vector.  The lookup is therefore
``structure_name → vector``, not ``image_path → vector``.

Input formats (two options)
----------------------------
Option A — per-protein array (recommended):
    ``--protein_emb_file``  : .npy file, shape (N_proteins, embed_dim), e.g. (25, 1280)
    ``--protein_names_file`` : plain-text file, one protein name per line,
                                in the same row order as the .npy file.

Option B — global cell-level arrays (same format as OpenCell ESM2 embeddings):
    ``--global_csv``        : CSV with an ``image_path`` column and a
                               ``structure_name`` column covering all cells.
    ``--global_esm2_file``  : .npy file, shape (N_cells, embed_dim),
                               row-aligned with ``--global_csv``.
    The script deduplicates by ``structure_name`` (all cells of the same
    protein have the same embedding, so any row is a valid representative).

Usage (Option A)
----------------
    python src/wtc/create_wtc_kfold_esm2_embeddings.py \\
        --protein_emb_file   /path/to/.../wtc11/esm2_embeddings/embeddings.npy \\
        --protein_names_file /path/to/.../wtc11/esm2_embeddings/protein_names.txt \\
        --kfold_dir          /path/to/.../wtc11/kfold5 \\
        --output_dir         /path/to/.../wtc11/esm2_embeddings_kfold5 \\
        --n_folds 5

Usage (Option B)
----------------
    python src/wtc/create_wtc_kfold_esm2_embeddings.py \\
        --global_csv         /path/to/.../wtc11/wtc11_cells_info_5fold.csv \\
        --global_esm2_file   /path/to/.../wtc11/esm2_embeddings/all_cells.npy \\
        --kfold_dir          /path/to/.../wtc11/kfold5 \\
        --output_dir         /path/to/.../wtc11/esm2_embeddings_kfold5 \\
        --n_folds 5
"""

import argparse
import os

import numpy as np
import pandas as pd


# ── Lookup-building helpers ──────────────────────────────────────────────────

def build_lookup_from_protein_files(emb_file: str, names_file: str) -> dict:
    """
    Build ``{structure_name: np.ndarray(embed_dim,)}`` from
    a per-protein .npy array and a matching names text file.

    Args:
        emb_file:   Path to .npy file of shape (N_proteins, embed_dim).
        names_file: Path to plain-text file with one protein name per line,
                    in the same row order as emb_file.

    Returns:
        dict mapping protein name → 1-D embedding vector.
    """
    embeddings = np.load(emb_file)          # (N_proteins, embed_dim)
    with open(names_file) as fh:
        names = [line.strip() for line in fh if line.strip()]

    if len(names) != len(embeddings):
        raise ValueError(
            f"Row count mismatch: {names_file} has {len(names)} names "
            f"but {emb_file} has {len(embeddings)} rows."
        )

    lookup = {name: embeddings[i] for i, name in enumerate(names)}
    print(f"  Loaded {len(lookup)} protein embeddings from {emb_file}")
    print(f"  Embed dim: {embeddings.shape[1]}")
    return lookup


def build_lookup_from_global_cell_arrays(global_csv: str, global_npy: str) -> dict:
    """
    Build ``{structure_name: np.ndarray(embed_dim,)}`` from a global
    per-cell CSV and its matching .npy array (OpenCell-style input).

    All cells sharing the same ``structure_name`` are assumed to have
    identical embeddings; the first occurrence is used as the representative.

    Args:
        global_csv: CSV containing ``image_path`` and ``structure_name`` columns.
        global_npy: .npy file of shape (N_cells, embed_dim), row-aligned with global_csv.

    Returns:
        dict mapping protein name → 1-D embedding vector.
    """
    df = pd.read_csv(global_csv, low_memory=False)
    embeddings = np.load(global_npy)        # (N_cells, embed_dim)

    if len(df) != len(embeddings):
        raise ValueError(
            f"Row count mismatch: {global_csv} has {len(df)} rows "
            f"but {global_npy} has {len(embeddings)} rows."
        )

    if 'structure_name' not in df.columns:
        raise ValueError(
            f"'structure_name' column not found in {global_csv}. "
            "This column is required to map cells to protein embeddings."
        )

    # Deduplicate: keep first occurrence per protein
    lookup = {}
    for idx, row in df.iterrows():
        name = row['structure_name']
        if name not in lookup:
            local_idx = df.index.get_loc(idx)
            lookup[name] = embeddings[local_idx]

    print(f"  Loaded {len(df)} cell rows → {len(lookup)} unique proteins")
    print(f"  Embed dim: {embeddings.shape[1]}")
    return lookup


# ── Core reindexing ──────────────────────────────────────────────────────────

def reindex_fold_split(fold_csv_path: str, lookup: dict, label: str) -> np.ndarray | None:
    """
    Build a (N_cells, embed_dim) array for one fold split by looking up
    each cell's ``structure_name`` in ``lookup``.

    Args:
        fold_csv_path: Path to the fold CSV (must contain ``structure_name``).
        lookup:        dict {protein_name → embedding vector}.
        label:         Human-readable label for error messages.

    Returns:
        np.ndarray of shape (N_cells, embed_dim) aligned with the CSV rows,
        or None if the file does not exist.
    """
    if not os.path.exists(fold_csv_path):
        return None

    df = pd.read_csv(fold_csv_path, low_memory=False)

    if 'structure_name' not in df.columns:
        raise ValueError(
            f"'structure_name' column not found in {fold_csv_path}. "
            "The fold CSVs must retain this column (create_wtc_kfold_csvs.py keeps it)."
        )

    missing = sorted({n for n in df['structure_name'] if n not in lookup})
    if missing:
        raise KeyError(
            f"{len(missing)} protein name(s) in {label} are not in the lookup.\n"
            f"Missing: {missing}\n"
            f"Available: {sorted(lookup.keys())}"
        )

    vectors = [lookup[name] for name in df['structure_name']]
    result = np.stack(vectors, axis=0)       # (N_cells, embed_dim)
    assert result.shape[0] == len(df)
    return result


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Create fold-specific ESM2 embedding files for WTC-11 k-fold training."
        )
    )

    # ── Option A: per-protein files ──────────────────────────────────────────
    parser.add_argument(
        '--protein_emb_file', type=str, default=None,
        help=(
            "[Option A] .npy file of shape (N_proteins, embed_dim) — "
            "one embedding per unique WTC-11 protein (structure_name)."
        ),
    )
    parser.add_argument(
        '--protein_names_file', type=str, default=None,
        help=(
            "[Option A] Plain-text file with one protein name per line, "
            "row-aligned with --protein_emb_file."
        ),
    )

    # ── Option B: global cell-level arrays (OpenCell-style) ──────────────────
    parser.add_argument(
        '--global_csv', type=str, default=None,
        help=(
            "[Option B] CSV with 'structure_name' (and optionally 'image_path') "
            "columns covering all WTC-11 cells, row-aligned with --global_esm2_file."
        ),
    )
    parser.add_argument(
        '--global_esm2_file', type=str, default=None,
        help=(
            "[Option B] .npy file of shape (N_cells, embed_dim) whose rows match "
            "--global_csv.  One embedding per cell (same protein → same vector)."
        ),
    )

    # ── Required ─────────────────────────────────────────────────────────────
    parser.add_argument(
        '--kfold_dir',
        default=(
            "/path/to/groups/labs/lab/user/datasets/"
            "SingleCellImagesDataset/wtc11/kfold5"
        ),
        help="Root directory of fold CSVs (fold0/train.csv, fold0/val.csv, …).",
    )
    parser.add_argument(
        '--output_dir',
        default=(
            "/path/to/groups/labs/lab/user/datasets/"
            "SingleCellImagesDataset/wtc11/esm2_embeddings_kfold5"
        ),
        help="Root output directory; per-fold .npy files go into {output_dir}/fold{k}/.",
    )
    parser.add_argument('--n_folds', type=int, default=5)
    parser.add_argument(
        '--splits', type=str, default='train,val',
        help="Comma-separated splits to process (default: train,val).",
    )
    args = parser.parse_args()

    splits = [s.strip() for s in args.splits.split(',')]

    # ── Validate input mode ───────────────────────────────────────────────────
    use_option_a = args.protein_emb_file is not None or args.protein_names_file is not None
    use_option_b = args.global_csv is not None or args.global_esm2_file is not None

    if use_option_a and use_option_b:
        parser.error(
            "Specify either Option A (--protein_emb_file + --protein_names_file) "
            "or Option B (--global_csv + --global_esm2_file), not both."
        )
    if not use_option_a and not use_option_b:
        parser.error(
            "Specify one input mode:\n"
            "  Option A: --protein_emb_file  --protein_names_file\n"
            "  Option B: --global_csv  --global_esm2_file"
        )
    if use_option_a:
        if args.protein_emb_file is None or args.protein_names_file is None:
            parser.error("Option A requires both --protein_emb_file and --protein_names_file.")
    if use_option_b:
        if args.global_csv is None or args.global_esm2_file is None:
            parser.error("Option B requires both --global_csv and --global_esm2_file.")

    # ── Header ────────────────────────────────────────────────────────────────
    print('=' * 60)
    print('WTC-11 — Creating fold-specific ESM2 embeddings')
    print('=' * 60)
    if use_option_a:
        print(f'Mode:              Option A (per-protein .npy)')
        print(f'Protein emb file:  {args.protein_emb_file}')
        print(f'Protein names:     {args.protein_names_file}')
    else:
        print(f'Mode:              Option B (global cell-level arrays)')
        print(f'Global CSV:        {args.global_csv}')
        print(f'Global ESM2 file:  {args.global_esm2_file}')
    print(f'K-fold dir:        {args.kfold_dir}')
    print(f'Output dir:        {args.output_dir}')
    print(f'Folds:             {args.n_folds}')
    print(f'Splits:            {splits}')
    print('=' * 60)

    # ── 1. Build protein → embedding lookup ──────────────────────────────────
    print('\nBuilding protein → embedding lookup ...')
    if use_option_a:
        lookup = build_lookup_from_protein_files(
            args.protein_emb_file, args.protein_names_file
        )
    else:
        lookup = build_lookup_from_global_cell_arrays(
            args.global_csv, args.global_esm2_file
        )
    print(f'  Lookup size: {len(lookup)} proteins')

    # ── 2. Process each fold ──────────────────────────────────────────────────
    for fold in range(args.n_folds):
        fold_csv_dir = os.path.join(args.kfold_dir, f'fold{fold}')
        fold_out_dir = os.path.join(args.output_dir, f'fold{fold}')
        os.makedirs(fold_out_dir, exist_ok=True)

        print(f'\nFold {fold}  ({fold_csv_dir})')

        for split in splits:
            fold_csv_path = os.path.join(fold_csv_dir, f'{split}.csv')
            out_path      = os.path.join(fold_out_dir,  f'{split}.npy')

            emb = reindex_fold_split(fold_csv_path, lookup, f'fold{fold}/{split}')
            if emb is None:
                print(f'  [{split:5s}] CSV not found — skipped')
                continue

            np.save(out_path, emb)
            print(
                f'  [{split:5s}] {emb.shape[0]:>6d} cells  '
                f'dim={emb.shape[1]}  → {out_path}'
            )

    print('\n' + '=' * 60)
    print('Done.')
    print('=' * 60)


if __name__ == '__main__':
    main()
