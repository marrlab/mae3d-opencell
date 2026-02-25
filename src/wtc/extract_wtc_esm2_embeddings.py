"""
Extract ESM2 embeddings for the 25 WTC-11 proteins.

Fetches canonical human protein sequences from UniProt by gene symbol,
runs ESM2 (mean-pooled over residues), and saves:

    output_dir/embeddings.npy       shape (N_proteins, embed_dim)
    output_dir/protein_names.txt    one gene symbol per line, row-aligned

These are the "Option A" inputs for create_wtc_kfold_esm2_embeddings.py.

Special case
------------
AAVS1 is a genomic safe-harbor locus (not a protein), used as a negative
control in WTC-11. It has no UniProt sequence.  Its embedding is set to
zeros and a warning is printed.

Usage
-----
    # GPU (recommended)
    python src/wtc/extract_wtc_esm2_embeddings.py \
        --info_csv  /path/to/.../wtc11/wtc11_cells_info_5fold.csv \
        --output_dir /path/to/.../wtc11/esm2_embeddings \
        --device cuda:0

    # CPU fallback
    python src/wtc/extract_wtc_esm2_embeddings.py \
        --info_csv  /path/to/.../wtc11/wtc11_cells_info_5fold.csv \
        --output_dir /path/to/.../wtc11/esm2_embeddings \
        --device cpu

    # Use Hugging Face transformers instead of fair-esm
    python src/wtc/extract_wtc_esm2_embeddings.py \
        --info_csv  /path/to/.../wtc11/wtc11_cells_info_5fold.csv \
        --output_dir /path/to/.../wtc11/esm2_embeddings \
        --device cuda:0 --use_transformers
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import requests
import torch
from tqdm import tqdm

# Re-use ESM2 helpers from the OpenCell script
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from extract_esm2_embeddings import (
    load_esm2_model,
    extract_esm2_embedding,
)

# Proteins that have no UniProt sequence — use zero embedding
NO_SEQUENCE_PROTEINS = {'AAVS1'}


# ── Sequence fetching ─────────────────────────────────────────────────────────

def fetch_uniprot_sequence_by_gene(gene_name: str, retries: int = 3) -> str | None:
    """Fetch canonical human protein sequence from UniProt by gene symbol."""
    url = (
        f"https://rest.uniprot.org/uniprotkb/search"
        f"?query=gene_exact:{gene_name}+AND+organism_id:9606"
        f"+AND+reviewed:true&format=fasta&size=1"
    )
    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=30)
            if r.status_code == 200 and r.text.strip():
                lines = r.text.strip().split('\n')
                seq = ''.join(l for l in lines if not l.startswith('>'))
                if seq:
                    return seq
            time.sleep(0.5)
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"    HTTP error fetching {gene_name}: {e}")
    return None


def load_or_fetch_sequences(gene_symbols: list[str], cache_path: str) -> dict[str, str | None]:
    """
    Return {gene_symbol: sequence_or_None}.
    Uses a JSON cache to avoid repeated UniProt calls.
    """
    if os.path.exists(cache_path):
        print(f"  Loading sequence cache: {cache_path}")
        with open(cache_path) as f:
            cache = json.load(f)
    else:
        cache = {}

    to_fetch = [g for g in gene_symbols if g not in cache and g not in NO_SEQUENCE_PROTEINS]

    if to_fetch:
        print(f"  Fetching {len(to_fetch)} sequences from UniProt ...")
        for gene in tqdm(to_fetch, desc="UniProt"):
            seq = fetch_uniprot_sequence_by_gene(gene)
            cache[gene] = seq          # may be None if failed
            if seq is None:
                print(f"    WARNING: no sequence found for {gene}")
            time.sleep(0.3)            # be polite to UniProt

        os.makedirs(os.path.dirname(cache_path) or '.', exist_ok=True)
        with open(cache_path, 'w') as f:
            json.dump(cache, f, indent=2)
        print(f"  Saved cache: {cache_path}")
    else:
        print(f"  All {len(gene_symbols)} sequences already cached.")

    return cache


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Extract ESM2 embeddings for the 25 WTC-11 proteins."
    )
    parser.add_argument(
        '--info_csv',
        default=(
            "/path/to/groups/labs/lab/user/datasets/"
            "SingleCellImagesDataset/wtc11/wtc11_cells_info_5fold.csv"
        ),
        help="Path to wtc11_cells_info_5fold.csv (must have 'structure_name' column).",
    )
    parser.add_argument(
        '--output_dir',
        default=(
            "/path/to/groups/labs/lab/user/datasets/"
            "SingleCellImagesDataset/wtc11/esm2_embeddings"
        ),
        help="Directory to write embeddings.npy and protein_names.txt.",
    )
    parser.add_argument(
        '--esm2_model', type=str, default='esm2_t33_650M_UR50D',
        choices=[
            'esm2_t6_8M_UR50D', 'esm2_t12_35M_UR50D', 'esm2_t30_150M_UR50D',
            'esm2_t33_650M_UR50D', 'esm2_t36_3B_UR50D',
        ],
    )
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument(
        '--use_transformers', action='store_true',
        help="Use Hugging Face transformers backend instead of fair-esm.",
    )
    parser.add_argument(
        '--cache_path',
        default=None,
        help="Path for the UniProt sequence JSON cache (default: output_dir/wtc_sequences.json).",
    )
    args = parser.parse_args()

    if args.cache_path is None:
        args.cache_path = os.path.join(args.output_dir, 'wtc_sequences.json')

    os.makedirs(args.output_dir, exist_ok=True)

    print('=' * 60)
    print('WTC-11 — ESM2 embedding extraction')
    print('=' * 60)
    print(f'  Info CSV:   {args.info_csv}')
    print(f'  Output dir: {args.output_dir}')
    print(f'  ESM2 model: {args.esm2_model}')
    print(f'  Device:     {args.device}')
    print(f'  Backend:    {"transformers" if args.use_transformers else "fair-esm"}')
    print('=' * 60)

    # ── 1. Collect unique protein names ──────────────────────────────────────
    df = pd.read_csv(args.info_csv, low_memory=False)
    gene_symbols = sorted(df['structure_name'].unique().tolist())
    print(f'\n{len(gene_symbols)} unique proteins: {gene_symbols}')

    # ── 2. Fetch sequences ───────────────────────────────────────────────────
    print('\nFetching UniProt sequences ...')
    sequences = load_or_fetch_sequences(gene_symbols, args.cache_path)

    # ── 3. Load ESM2 model ───────────────────────────────────────────────────
    print(f'\nLoading ESM2 model on {args.device} ...')
    model, tokenizer_or_converter, batch_converter, embed_dim = load_esm2_model(
        args.esm2_model, args.device, use_transformers=args.use_transformers
    )
    tc = tokenizer_or_converter if args.use_transformers else batch_converter

    # ── 4. Compute embeddings ─────────────────────────────────────────────────
    print(f'\nComputing embeddings (dim={embed_dim}) ...')
    embeddings = np.zeros((len(gene_symbols), embed_dim), dtype=np.float32)

    for i, gene in enumerate(tqdm(gene_symbols, desc="ESM2")):
        if gene in NO_SEQUENCE_PROTEINS:
            print(f'  {gene}: no protein sequence → zero embedding')
            # embeddings[i] already zero
            continue

        seq = sequences.get(gene)
        if seq is None:
            print(f'  WARNING: {gene}: sequence missing → zero embedding')
            continue

        emb = extract_esm2_embedding(
            model, tc, seq, args.device,
            use_transformers=args.use_transformers,
        )
        embeddings[i] = emb.astype(np.float32)

    # ── 5. Save ───────────────────────────────────────────────────────────────
    emb_path   = os.path.join(args.output_dir, 'embeddings.npy')
    names_path = os.path.join(args.output_dir, 'protein_names.txt')

    np.save(emb_path, embeddings)
    with open(names_path, 'w') as f:
        f.write('\n'.join(gene_symbols) + '\n')

    print(f'\nSaved:')
    print(f'  {emb_path}  — shape {embeddings.shape}, dtype {embeddings.dtype}')
    print(f'  {names_path}  — {len(gene_symbols)} protein names')

    # ── 6. Quick sanity check ─────────────────────────────────────────────────
    zero_rows = np.where(~embeddings.any(axis=1))[0]
    if zero_rows.size:
        zero_names = [gene_symbols[i] for i in zero_rows]
        print(f'\n  NOTE: {len(zero_names)} zero embeddings: {zero_names}')

    print('\n' + '=' * 60)
    print('Done.')
    print('=' * 60)


if __name__ == '__main__':
    main()
