"""
OpenCell Protein-Protein Interaction Dataset for metric learning.

Creates pairs of proteins with positive/negative labels for contrastive learning.
"""

import pandas as pd
import tifffile
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
import torch


class OpenCellPPIDataset(Dataset):
    """
    OpenCell Dataset for PPI prediction via metric learning.

    Returns pairs of protein images with labels:
    - 1: Known PPI (positive pair)
    - 0: No known interaction (negative pair)

    Each protein is represented by one randomly sampled cell image.
    """

    def __init__(self,
                 csv_path,
                 ppi_csv_path,
                 abundance_csv_path=None,
                 split='train',
                 transform=None,
                 cache_rate=0.0,
                 num_workers=4,
                 use_max_projection=False,
                 pval_threshold=5.0,
                 enrichment_threshold=2.5,
                 stoichiometry_threshold=0.05,
                 n_abundance_buckets=10,
                 n_negatives_per_positive=1,
                 seed=42,
                 mae_embedding_path=None,
                 mae_embedding_array=None,
                 mae_embedding_lookup_dict=None):
        """
        Args:
            csv_path: Path to CSV file with 'image_path' and protein information
            ppi_csv_path: Path to PPI interactions CSV
            abundance_csv_path: Path to protein abundance CSV for bucket matching
            split: Dataset split ('train', 'val', 'test')
            transform: Transforms to apply to images
            cache_rate: Fraction of dataset to cache in memory
            num_workers: Number of workers for parallel caching
            use_max_projection: If True, apply max projection along Z-axis for 2D
            pval_threshold: Minimum -log10(pval) threshold for PPI filtering
            enrichment_threshold: Minimum enrichment threshold
            stoichiometry_threshold: Minimum interaction_stoichiometry threshold
            n_abundance_buckets: Number of buckets for abundance-matched negative sampling
            n_negatives_per_positive: Ratio of negative to positive pairs
            seed: Random seed for reproducibility
            mae_embedding_path: Path to precomputed MAE embeddings .npy file (optional).
                               Rows are aligned with the CSV rows (row i = embedding for CSV row i).
                               When provided, loads embeddings instead of images (faster training).
        """
        self.split = split
        self.transform = transform
        self.use_max_projection = use_max_projection
        self.seed = seed
        np.random.seed(seed)

        # Embedding mode
        self.use_mae_embeddings = mae_embedding_path is not None or mae_embedding_array is not None
        self.mae_embeddings = None
        self.mae_embedding_lookup = None   # image_path -> row_idx (kfold combined mode)
        self.protein_to_image_paths = None # protein -> [image_paths] for lookup mode
        self.protein_to_indices = None     # protein -> [row_indices] for positional mode

        # Load image metadata
        df = pd.read_csv(csv_path)

        # Get protein names
        if 'file_gene_symbol' in df.columns:
            df['protein_name'] = df['file_gene_symbol']
        elif 'folder_protein' in df.columns:
            df['protein_name'] = df['folder_protein']
        else:
            raise ValueError("CSV must contain 'file_gene_symbol' or 'folder_protein' column")

        if self.use_mae_embeddings:
            # Load or use pre-built embedding array
            if mae_embedding_array is not None:
                self.mae_embeddings = mae_embedding_array
            else:
                self.mae_embeddings = np.load(mae_embedding_path)

            if mae_embedding_lookup_dict is not None:
                # Kfold combined mode: map protein -> image_paths, then lookup by image_path.
                # Cells from the kfold CSV may reside in any of the dataset1 source splits.
                self.mae_embedding_lookup = mae_embedding_lookup_dict
                self.protein_to_image_paths = defaultdict(list)
                for _, row in df.iterrows():
                    img_path = row['image_path']
                    if img_path in self.mae_embedding_lookup:
                        self.protein_to_image_paths[row['protein_name']].append(img_path)
                self.available_proteins = set(self.protein_to_image_paths.keys())
                print(f"Loaded {len(self.mae_embeddings)} combined embeddings, "
                      f"{len(self.available_proteins)} proteins (kfold lookup mode)")
            else:
                # Legacy positional mode: row i in npy == row i in CSV (same dataset split).
                self.protein_to_indices = defaultdict(list)
                for row_idx, (_, row) in enumerate(df.iterrows()):
                    self.protein_to_indices[row['protein_name']].append(row_idx)
                self.available_proteins = set(self.protein_to_indices.keys())
                print(f"Loaded {len(self.mae_embeddings)} embeddings (dim={self.mae_embeddings.shape[1]}) "
                      f"for {len(self.available_proteins)} proteins (positional mode)")
        else:
            # Build protein → image paths mapping
            self.protein_to_images = defaultdict(list)
            for idx, row in df.iterrows():
                self.protein_to_images[row['protein_name']].append(row['image_path'])
            self.available_proteins = set(self.protein_to_images.keys())
            print(f"Loaded {len(df)} images for {len(self.available_proteins)} proteins")

        # Load and filter PPI data
        ppi_df = pd.read_csv(ppi_csv_path)
        print(f"Loaded {len(ppi_df)} total PPI records")

        # Filter by thresholds
        filtered_ppi = ppi_df[
            (ppi_df['pval'] > pval_threshold) &
            (ppi_df['enrichment'] > enrichment_threshold) &
            (ppi_df['interaction_stoichiometry'] > stoichiometry_threshold)
        ].copy()
        print(f"After filtering: {len(filtered_ppi)} interactions")

        # Build positive pairs
        self.positive_pairs = self._build_positive_pairs(filtered_ppi)
        print(f"Built {len(self.positive_pairs)} positive pairs")

        # Load abundance data and build buckets
        if abundance_csv_path is not None:
            abundance_dict = self._load_abundance_data(abundance_csv_path)
            bucket_assignments, bucket_proteins = self._assign_abundance_buckets(
                list(self.available_proteins), abundance_dict, n_abundance_buckets
            )
        else:
            bucket_assignments = {p: 0 for p in self.available_proteins}
            bucket_proteins = {0: list(self.available_proteins)}

        # Build negative pairs
        self.negative_pairs = self._build_negative_pairs(
            self.positive_pairs,
            bucket_assignments,
            bucket_proteins,
            n_negatives_per_positive
        )
        print(f"Built {len(self.negative_pairs)} negative pairs")

        # Create combined pairs list with labels
        self.pairs = []
        for pair in self.positive_pairs:
            self.pairs.append((pair[0], pair[1], 1))  # positive
        for pair in self.negative_pairs:
            self.pairs.append((pair[0], pair[1], 0))  # negative

        # Shuffle pairs
        np.random.shuffle(self.pairs)
        print(f"Total pairs: {len(self.pairs)} ({len(self.positive_pairs)} pos, {len(self.negative_pairs)} neg)")

        # Cache for images
        self.cache_rate = cache_rate
        self._cache = {}

        if cache_rate > 0.0:
            self._preload_cache(num_workers)

    def _build_positive_pairs(self, ppi_df):
        """Build positive pairs from filtered PPI data."""
        positive_pairs = []
        positive_set = set()

        for _, row in ppi_df.iterrows():
            target = row['target_gene_name']
            interactor = row['interactor_gene_name']

            if target in self.available_proteins and interactor in self.available_proteins:
                pair = tuple(sorted([target, interactor]))
                if pair not in positive_set:
                    positive_set.add(pair)
                    positive_pairs.append(pair)

        return positive_pairs

    def _load_abundance_data(self, abundance_path):
        """Load protein abundance data."""
        abundance_df = pd.read_csv(abundance_path)
        abundance_dict = {}

        for _, row in abundance_df.iterrows():
            gene = row['gene_name']
            if pd.notna(row.get('hek_protein_conc_nm')):
                abundance_dict[gene] = row['hek_protein_conc_nm']
            elif pd.notna(row.get('hek_rna_tpm')):
                abundance_dict[gene] = row['hek_rna_tpm']

        return abundance_dict

    def _assign_abundance_buckets(self, proteins, abundance_dict, n_buckets):
        """Assign proteins to abundance buckets."""
        protein_abundance = []
        for prot in proteins:
            if prot in abundance_dict:
                protein_abundance.append((prot, abundance_dict[prot]))
            else:
                protein_abundance.append((prot, None))

        with_abundance = [(p, a) for p, a in protein_abundance if a is not None]
        without_abundance = [p for p, a in protein_abundance if a is None]

        bucket_assignments = {}
        bucket_proteins = defaultdict(list)

        if with_abundance:
            with_abundance.sort(key=lambda x: x[1])
            bucket_size = len(with_abundance) / n_buckets

            for i, (prot, _) in enumerate(with_abundance):
                bucket_id = min(int(i / bucket_size), n_buckets - 1)
                bucket_assignments[prot] = bucket_id
                bucket_proteins[bucket_id].append(prot)

        for prot in without_abundance:
            bucket_assignments[prot] = -1
            bucket_proteins[-1].append(prot)

        return bucket_assignments, dict(bucket_proteins)

    def _build_negative_pairs(self, positive_pairs, bucket_assignments,
                               bucket_proteins, n_negatives_per_positive):
        """Build abundance-matched negative pairs."""
        positive_set = set(positive_pairs)
        negative_pairs = []
        all_proteins = list(bucket_assignments.keys())

        for prot1, prot2 in positive_pairs:
            bucket1 = bucket_assignments.get(prot1, -1)
            bucket2 = bucket_assignments.get(prot2, -1)

            candidates1 = bucket_proteins.get(bucket1, all_proteins)
            candidates2 = bucket_proteins.get(bucket2, all_proteins)

            for _ in range(n_negatives_per_positive * 10):
                neg1 = np.random.choice(candidates1)
                neg2 = np.random.choice(candidates2)

                if neg1 == neg2:
                    continue

                neg_pair = tuple(sorted([neg1, neg2]))

                if neg_pair not in positive_set and neg_pair not in negative_pairs:
                    negative_pairs.append(neg_pair)
                    break

        # Fill remaining with random pairs if needed
        target_count = len(positive_pairs) * n_negatives_per_positive
        while len(negative_pairs) < target_count:
            neg1, neg2 = np.random.choice(all_proteins, 2, replace=False)
            neg_pair = tuple(sorted([neg1, neg2]))
            if neg_pair not in positive_set and neg_pair not in negative_pairs:
                negative_pairs.append(neg_pair)

        return negative_pairs[:target_count]

    def _preload_cache(self, num_workers):
        """Pre-cache a portion of unique image paths."""
        all_image_paths = set()
        for images in self.protein_to_images.values():
            all_image_paths.update(images)

        all_image_paths = list(all_image_paths)
        num_to_cache = int(len(all_image_paths) * self.cache_rate)
        print(f"Caching {num_to_cache}/{len(all_image_paths)} images...")

        def _load_image(path):
            return path, tifffile.imread(path)

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(_load_image, p) for p in all_image_paths[:num_to_cache]]
            for i, future in enumerate(futures):
                path, img = future.result()
                self._cache[path] = img
                if (i + 1) % max(1, num_to_cache // 10) == 0:
                    print(f"  Cached {i + 1}/{num_to_cache} images")

        print(f"Caching complete! {len(self._cache)} images in memory.")

    def _load_image(self, image_path):
        """Load image from cache or disk."""
        if image_path in self._cache:
            img = self._cache[image_path]
        else:
            img = tifffile.imread(image_path)

        if self.use_max_projection:
            img = np.max(img, axis=0)

        return img

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        prot1, prot2, label = self.pairs[idx]

        if self.use_mae_embeddings:
            # Embedding mode: sample a random cell embedding for each protein
            if self.mae_embedding_lookup is not None:
                # Kfold combined mode: look up embedding row by image_path
                img_path1 = np.random.choice(self.protein_to_image_paths[prot1])
                img_path2 = np.random.choice(self.protein_to_image_paths[prot2])
                row_idx1 = self.mae_embedding_lookup[img_path1]
                row_idx2 = self.mae_embedding_lookup[img_path2]
            else:
                # Legacy positional mode
                row_idx1 = np.random.choice(self.protein_to_indices[prot1])
                row_idx2 = np.random.choice(self.protein_to_indices[prot2])
            emb1 = torch.tensor(self.mae_embeddings[row_idx1], dtype=torch.float32)
            emb2 = torch.tensor(self.mae_embeddings[row_idx2], dtype=torch.float32)
            return {
                "embedding1": emb1,
                "embedding2": emb2,
                "label": torch.tensor(label, dtype=torch.float32),
                "protein1": prot1,
                "protein2": prot2,
            }

        # Image mode: sample a random image for each protein
        img_path1 = np.random.choice(self.protein_to_images[prot1])
        img_path2 = np.random.choice(self.protein_to_images[prot2])

        img1 = self._load_image(img_path1)
        img2 = self._load_image(img_path2)

        data1 = {"image": img1}
        data2 = {"image": img2}

        if self.transform:
            data1 = self.transform(data1)
            data2 = self.transform(data2)

        return {
            "image1": data1["image"],
            "image2": data2["image"],
            "label": torch.tensor(label, dtype=torch.float32),
            "protein1": prot1,
            "protein2": prot2,
        }

    def get_statistics(self):
        """Return dataset statistics."""
        stats = {
            'total_pairs': len(self.pairs),
            'positive_pairs': len(self.positive_pairs),
            'negative_pairs': len(self.negative_pairs),
            'unique_proteins': len(self.available_proteins),
        }
        if self.protein_to_image_paths is not None:
            stats['total_images'] = sum(len(v) for v in self.protein_to_image_paths.values())
        elif getattr(self, 'protein_to_images', None) is not None:
            stats['total_images'] = sum(len(v) for v in self.protein_to_images.values())
        return stats


class OpenCellPPITestDataset(Dataset):
    """
    OpenCell PPI Test Dataset.

    For testing, we need to extract embeddings for all cells,
    then aggregate per protein and evaluate pairs.
    This dataset returns individual cells with protein names.
    """

    def __init__(self,
                 csv_path,
                 transform=None,
                 use_max_projection=False):
        """
        Args:
            csv_path: Path to CSV file with 'image_path' and protein information
            transform: Transforms to apply to images
            use_max_projection: If True, apply max projection along Z-axis
        """
        self.transform = transform
        self.use_max_projection = use_max_projection

        df = pd.read_csv(csv_path)

        if 'file_gene_symbol' in df.columns:
            df['protein_name'] = df['file_gene_symbol']
        elif 'folder_protein' in df.columns:
            df['protein_name'] = df['folder_protein']
        else:
            raise ValueError("CSV must contain 'file_gene_symbol' or 'folder_protein'")

        self.image_paths = df['image_path'].tolist()
        self.protein_names = df['protein_name'].tolist()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        protein_name = self.protein_names[idx]

        img = tifffile.imread(img_path)

        if self.use_max_projection:
            img = np.max(img, axis=0)

        data = {"image": img, "protein_name": protein_name}

        if self.transform:
            img_data = {"image": img}
            img_data = self.transform(img_data)
            data["image"] = img_data["image"]

        return data
