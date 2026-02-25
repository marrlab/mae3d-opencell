"""
Extract ESM2 embeddings for OpenCell proteins.

This script:
1. Loads protein sequences from UniProt (cached locally after first fetch)
2. Computes ESM2 embeddings for each unique protein
3. Saves embeddings aligned with CSV rows for train/val/test splits

The embeddings are saved in the same order as the CSV rows, so each image
gets the corresponding protein's ESM2 embedding.

Requirements:
    pip install fair-esm  # Facebook AI Research ESM library
    # OR
    pip install transformers  # Hugging Face transformers (alternative)

Usage:
    python src/extract_esm2_embeddings.py --output_dir /path/to/output

    # With custom model
    python src/extract_esm2_embeddings.py --esm2_model esm2_t33_650M_UR50D

    # On GPU
    python src/extract_esm2_embeddings.py --device cuda:0

    # Use Hugging Face transformers instead of fair-esm
    python src/extract_esm2_embeddings.py --use_transformers
"""

import os
import sys
import argparse
import json
import time
import requests
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import torch

# Add src to path
sys.path.insert(0, os.path.dirname(__file__))


def fetch_uniprot_sequence_by_ensembl(ensembl_id: str, retries: int = 3) -> Optional[str]:
    """
    Fetch protein sequence from UniProt using Ensembl gene ID.

    Args:
        ensembl_id: Ensembl gene ID (e.g., ENSG00000127837)
        retries: Number of retry attempts

    Returns:
        Protein sequence string or None if not found
    """
    # UniProt ID mapping API
    url = f"https://rest.uniprot.org/uniprotkb/search?query=xref:ensembl-{ensembl_id}&format=fasta&size=1"

    for attempt in range(retries):
        try:
            response = requests.get(url, timeout=30)
            if response.status_code == 200 and response.text.strip():
                # Parse FASTA format
                lines = response.text.strip().split('\n')
                sequence = ''.join(line for line in lines if not line.startswith('>'))
                if sequence:
                    return sequence
            time.sleep(0.5)  # Rate limiting
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                print(f"Failed to fetch {ensembl_id}: {e}")

    return None


def fetch_uniprot_sequence_by_gene(gene_name: str, organism: str = "human", retries: int = 3) -> Optional[str]:
    """
    Fetch protein sequence from UniProt using gene name.

    Args:
        gene_name: Gene symbol (e.g., AAMP)
        organism: Organism name (default: human)
        retries: Number of retry attempts

    Returns:
        Protein sequence string or None if not found
    """
    # Search UniProt by gene name and organism
    url = f"https://rest.uniprot.org/uniprotkb/search?query=gene:{gene_name}+AND+organism_name:{organism}&format=fasta&size=1"

    for attempt in range(retries):
        try:
            response = requests.get(url, timeout=30)
            if response.status_code == 200 and response.text.strip():
                lines = response.text.strip().split('\n')
                sequence = ''.join(line for line in lines if not line.startswith('>'))
                if sequence:
                    return sequence
            time.sleep(0.5)
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"Failed to fetch {gene_name}: {e}")

    return None


def load_or_fetch_sequences(
    csv_paths: List[str],
    cache_path: str,
    ensembl_col: str = 'folder_ensembl_id',
    gene_col: str = 'file_gene_symbol'
) -> Dict[str, str]:
    """
    Load protein sequences from cache or fetch from UniProt.

    Args:
        csv_paths: List of paths to CSV files with protein identifiers
        cache_path: Path to cache JSON file
        ensembl_col: Column name for Ensembl IDs
        gene_col: Column name for gene symbols

    Returns:
        Dictionary mapping Ensembl ID to protein sequence
    """
    # Try to load from cache
    if os.path.exists(cache_path):
        print(f"Loading sequences from cache: {cache_path}")
        with open(cache_path, 'r') as f:
            sequences = json.load(f)
        return sequences

    # Load all CSVs to get unique proteins across all splits
    dfs = [pd.read_csv(p) for p in csv_paths if os.path.exists(p)]
    df = pd.concat(dfs, ignore_index=True)
    unique_proteins = df[[ensembl_col, gene_col]].drop_duplicates()
    print(f"Found {len(unique_proteins)} unique proteins across {len(dfs)} splits")

    sequences = {}
    failed = []

    for _, row in tqdm(unique_proteins.iterrows(), total=len(unique_proteins), desc="Fetching sequences"):
        ensembl_id = row[ensembl_col]
        gene_name = row[gene_col]

        if ensembl_id in sequences:
            continue

        # Try Ensembl ID first, then gene name
        seq = fetch_uniprot_sequence_by_ensembl(ensembl_id)
        if seq is None:
            seq = fetch_uniprot_sequence_by_gene(gene_name)

        if seq is not None:
            sequences[ensembl_id] = seq
        else:
            failed.append((ensembl_id, gene_name))
            print(f"  Failed to fetch: {gene_name} ({ensembl_id})")

    print(f"Successfully fetched {len(sequences)} sequences")
    if failed:
        print(f"Failed to fetch {len(failed)} sequences: {failed[:5]}...")

    # Save to cache
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, 'w') as f:
        json.dump(sequences, f)
    print(f"Saved sequences to cache: {cache_path}")

    return sequences


def load_esm2_model_fairesm(model_name: str, device: str):
    """
    Load ESM2 model using fair-esm library.

    Args:
        model_name: ESM2 model name (e.g., 'esm2_t33_650M_UR50D')
        device: Device to load model on

    Returns:
        model, alphabet, batch_converter, embed_dim
    """
    import esm

    print(f"Loading ESM2 model (fair-esm): {model_name}")

    # Load model
    if model_name == 'esm2_t33_650M_UR50D':
        model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    elif model_name == 'esm2_t36_3B_UR50D':
        model, alphabet = esm.pretrained.esm2_t36_3B_UR50D()
    elif model_name == 'esm2_t30_150M_UR50D':
        model, alphabet = esm.pretrained.esm2_t30_150M_UR50D()
    elif model_name == 'esm2_t12_35M_UR50D':
        model, alphabet = esm.pretrained.esm2_t12_35M_UR50D()
    elif model_name == 'esm2_t6_8M_UR50D':
        model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
    else:
        raise ValueError(f"Unknown ESM2 model: {model_name}")

    batch_converter = alphabet.get_batch_converter()
    model.eval()
    model = model.to(device)

    embed_dim = model.embed_dim
    print(f"  Embedding dimension: {embed_dim}")

    return model, alphabet, batch_converter, embed_dim


def load_esm2_model_transformers(model_name: str, device: str):
    """
    Load ESM2 model using Hugging Face transformers library.

    Args:
        model_name: ESM2 model name (e.g., 'esm2_t33_650M_UR50D')
        device: Device to load model on

    Returns:
        model, tokenizer, None, embed_dim
    """
    from transformers import AutoTokenizer, AutoModel

    # Map model names to Hugging Face model IDs
    hf_model_map = {
        'esm2_t6_8M_UR50D': 'facebook/esm2_t6_8M_UR50D',
        'esm2_t12_35M_UR50D': 'facebook/esm2_t12_35M_UR50D',
        'esm2_t30_150M_UR50D': 'facebook/esm2_t30_150M_UR50D',
        'esm2_t33_650M_UR50D': 'facebook/esm2_t33_650M_UR50D',
        'esm2_t36_3B_UR50D': 'facebook/esm2_t36_3B_UR50D',
    }

    if model_name not in hf_model_map:
        raise ValueError(f"Unknown ESM2 model: {model_name}")

    hf_model_id = hf_model_map[model_name]
    print(f"Loading ESM2 model (transformers): {hf_model_id}")

    tokenizer = AutoTokenizer.from_pretrained(hf_model_id)
    model = AutoModel.from_pretrained(hf_model_id)
    model.eval()
    model = model.to(device)

    embed_dim = model.config.hidden_size
    print(f"  Embedding dimension: {embed_dim}")

    return model, tokenizer, None, embed_dim


def load_esm2_model(model_name: str, device: str, use_transformers: bool = False):
    """
    Load ESM2 model and tokenizer.

    Args:
        model_name: ESM2 model name (e.g., 'esm2_t33_650M_UR50D')
        device: Device to load model on
        use_transformers: If True, use Hugging Face transformers library

    Returns:
        model, tokenizer/alphabet, batch_converter (None for transformers), embed_dim
    """
    if use_transformers:
        return load_esm2_model_transformers(model_name, device)
    else:
        return load_esm2_model_fairesm(model_name, device)


def extract_esm2_embedding_fairesm(
    model,
    batch_converter,
    sequence: str,
    device: str,
    max_length: int = 1022
) -> np.ndarray:
    """
    Extract ESM2 embedding using fair-esm library.
    """
    if len(sequence) > max_length:
        sequence = sequence[:max_length]

    data = [("protein", sequence)]
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    batch_tokens = batch_tokens.to(device)

    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[model.num_layers], return_contacts=False)

    token_representations = results["representations"][model.num_layers]
    sequence_representation = token_representations[0, 1:-1, :].mean(dim=0)

    return sequence_representation.cpu().numpy()


def extract_esm2_embedding_transformers(
    model,
    tokenizer,
    sequence: str,
    device: str,
    max_length: int = 1022
) -> np.ndarray:
    """
    Extract ESM2 embedding using Hugging Face transformers library.
    """
    if len(sequence) > max_length:
        sequence = sequence[:max_length]

    inputs = tokenizer(sequence, return_tensors="pt", truncation=True, max_length=max_length + 2)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    # Get last hidden state and mean pool (exclude special tokens)
    last_hidden_state = outputs.last_hidden_state
    # Exclude [CLS] and [EOS] tokens
    sequence_representation = last_hidden_state[0, 1:-1, :].mean(dim=0)

    return sequence_representation.cpu().numpy()


def extract_esm2_embedding(
    model,
    tokenizer_or_converter,
    sequence: str,
    device: str,
    max_length: int = 1022,
    use_transformers: bool = False
) -> np.ndarray:
    """
    Extract ESM2 embedding for a single protein sequence.

    Uses mean pooling over sequence positions (excluding special tokens).

    Args:
        model: ESM2 model
        tokenizer_or_converter: ESM2 batch converter or HF tokenizer
        sequence: Protein sequence string
        device: Device
        max_length: Maximum sequence length (truncate if longer)
        use_transformers: If True, use Hugging Face transformers

    Returns:
        Embedding array of shape [embed_dim]
    """
    if use_transformers:
        return extract_esm2_embedding_transformers(model, tokenizer_or_converter, sequence, device, max_length)
    else:
        return extract_esm2_embedding_fairesm(model, tokenizer_or_converter, sequence, device, max_length)


def extract_embeddings_for_split(
    csv_path: str,
    sequences: Dict[str, str],
    model,
    tokenizer_or_converter,
    device: str,
    embed_dim: int,
    ensembl_col: str = 'folder_ensembl_id',
    batch_size: int = 8,
    use_transformers: bool = False
) -> np.ndarray:
    """
    Extract ESM2 embeddings for all samples in a CSV split.

    Args:
        csv_path: Path to CSV file
        sequences: Dictionary mapping Ensembl ID to sequence
        model: ESM2 model
        tokenizer_or_converter: ESM2 batch converter or HF tokenizer
        device: Device
        embed_dim: Embedding dimension
        ensembl_col: Column name for Ensembl IDs
        batch_size: Batch size for processing
        use_transformers: If True, use Hugging Face transformers

    Returns:
        Embeddings array of shape [num_samples, embed_dim]
    """
    df = pd.read_csv(csv_path)
    num_samples = len(df)

    print(f"Extracting embeddings for {num_samples} samples from {csv_path}")

    # Pre-compute embeddings for unique proteins
    unique_ensembl_ids = df[ensembl_col].unique()
    protein_embeddings = {}

    print(f"Computing embeddings for {len(unique_ensembl_ids)} unique proteins...")
    for ensembl_id in tqdm(unique_ensembl_ids, desc="Extracting embeddings"):
        if ensembl_id not in sequences:
            print(f"  Warning: No sequence for {ensembl_id}, using zero embedding")
            protein_embeddings[ensembl_id] = np.zeros(embed_dim, dtype=np.float32)
        else:
            seq = sequences[ensembl_id]
            emb = extract_esm2_embedding(model, tokenizer_or_converter, seq, device,
                                         use_transformers=use_transformers)
            protein_embeddings[ensembl_id] = emb.astype(np.float32)

    # Create embeddings array aligned with CSV rows
    embeddings = np.zeros((num_samples, embed_dim), dtype=np.float32)

    for idx, row in df.iterrows():
        ensembl_id = row[ensembl_col]
        embeddings[idx] = protein_embeddings[ensembl_id]

    return embeddings


def main():
    parser = argparse.ArgumentParser(description='Extract ESM2 embeddings for OpenCell')
    parser.add_argument('--csv_dir', type=str,
                        default='/path/to/datasets/opencell/opencell_dataset/single_cells/metadata/dataset1/',
                        help='Directory containing train.csv, val.csv, test.csv')
    parser.add_argument('--output_dir', type=str,
                        default='/path/to/datasets/opencell_embeddings/esm2',
                        help='Output directory for embeddings')
    parser.add_argument('--cache_dir', type=str,
                        default='/path/to/datasets/opencell_embeddings/protein_sequences',
                        help='Directory to cache protein sequences')
    parser.add_argument('--esm2_model', type=str, default='esm2_t33_650M_UR50D',
                        choices=['esm2_t6_8M_UR50D', 'esm2_t12_35M_UR50D', 'esm2_t30_150M_UR50D',
                                 'esm2_t33_650M_UR50D', 'esm2_t36_3B_UR50D'],
                        help='ESM2 model to use')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to use (cuda:0, cpu, etc.)')
    parser.add_argument('--splits', type=str, nargs='+', default=['train', 'val', 'test'],
                        help='Splits to process')
    parser.add_argument('--use_transformers', action='store_true',
                        help='Use Hugging Face transformers instead of fair-esm')
    args = parser.parse_args()

    print("=" * 60)
    print("ESM2 Embedding Extraction for OpenCell")
    print("=" * 60)
    print(f"CSV directory: {args.csv_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"ESM2 model: {args.esm2_model}")
    print(f"Device: {args.device}")
    print(f"Splits: {args.splits}")
    print(f"Backend: {'transformers' if args.use_transformers else 'fair-esm'}")
    print("=" * 60)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)

    # Load or fetch protein sequences from ALL splits
    all_csv_paths = [os.path.join(args.csv_dir, f'{split}.csv') for split in args.splits]
    cache_path = os.path.join(args.cache_dir, 'opencell_sequences.json')

    sequences = load_or_fetch_sequences(all_csv_paths, cache_path)
    print(f"Loaded {len(sequences)} protein sequences")

    # Load ESM2 model
    model, tokenizer_or_converter, batch_converter, embed_dim = load_esm2_model(
        args.esm2_model, args.device, use_transformers=args.use_transformers
    )

    # Process each split
    for split in args.splits:
        csv_path = os.path.join(args.csv_dir, f'{split}.csv')
        if not os.path.exists(csv_path):
            print(f"Skipping {split}: {csv_path} not found")
            continue

        print(f"\n{'='*60}")
        print(f"Processing {split} split...")
        print(f"{'='*60}")

        embeddings = extract_embeddings_for_split(
            csv_path=csv_path,
            sequences=sequences,
            model=model,
            tokenizer_or_converter=tokenizer_or_converter if args.use_transformers else batch_converter,
            device=args.device,
            embed_dim=embed_dim,
            use_transformers=args.use_transformers
        )

        # Save embeddings
        output_path = os.path.join(args.output_dir, f'{split}.npy')
        np.save(output_path, embeddings)
        print(f"Saved {split} embeddings: {output_path}")
        print(f"  Shape: {embeddings.shape}")
        print(f"  Dtype: {embeddings.dtype}")
        print(f"  Size: {embeddings.nbytes / 1e6:.2f} MB")

    # Save metadata
    metadata = {
        'esm2_model': args.esm2_model,
        'embed_dim': embed_dim,
        'csv_dir': args.csv_dir,
        'num_sequences': len(sequences),
        'splits': args.splits
    }
    metadata_path = os.path.join(args.output_dir, 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"\nSaved metadata: {metadata_path}")

    print("\n" + "=" * 60)
    print("ESM2 embedding extraction complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
