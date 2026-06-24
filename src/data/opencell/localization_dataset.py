import pandas as pd
import tifffile
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from concurrent.futures import ThreadPoolExecutor
import torch


# Define all 17 localization labels
LOCALIZATION_LABELS = [
    'big_aggregates',
    'cell_contact',
    'centrosome',
    'chromatin',
    'cytoplasmic',
    'cytoskeleton',
    'er',
    'focal_adhesions',
    'golgi',
    'membrane',
    'mitochondria',
    'nuclear_membrane',
    'nuclear_punctae',
    'nucleolus_fc_dfc',
    'nucleolus_gc',
    'nucleoplasm',
    'vesicles',
]

# Create label to index mapping
LABEL_TO_IDX = {label: idx for idx, label in enumerate(LOCALIZATION_LABELS)}


class OpenCellLocalizationDataset(Dataset):
    """
    OpenCell Dataset for protein localization classification.

    Supports two modes:
    1. Image mode (default): Loads images and runs them through MAE encoder
    2. Embedding mode: Loads precomputed MAE embeddings (much faster when encoder is frozen)
    """

    def __init__(self,
                 csv_path,
                 localization_csv_path,
                 split='train',
                 transform=None,
                 cache_rate=0.0,
                 num_workers=4,
                 use_max_projection=False,
                 grade_weights=None,
                 z_slice_start=None,
                 z_slice_end=None,
                 mae_embedding_path=None,
                 mae_embedding_csv_path=None,
                 mae_embedding_array=None,
                 mae_embedding_lookup_dict=None):
        """
        OpenCell Dataset for protein localization classification.

        Args:
            csv_path: Path to CSV file with 'image_path' and protein information
            localization_csv_path: Path to localization annotations CSV
            split: Dataset split ('train', 'val', 'test')
            transform: MONAI transforms to apply
            cache_rate: Fraction of dataset to cache in memory (0.0 to 1.0)
            num_workers: Number of workers for parallel caching
            use_max_projection: If True, apply max projection along Z-axis for 2D
            grade_weights: Dict with weights for each grade, e.g., {3: 1.0, 2: 0.5, 1: 0.25}
                          If None, defaults to {3: 1.0, 2: 0.5, 1: 0.25}
            z_slice_start: Start index for Z-slice selection (inclusive). If None, use all slices.
            z_slice_end: End index for Z-slice selection (exclusive). If None, use all slices.
            mae_embedding_path: Path to precomputed MAE embeddings .npy file (optional).
                               If provided, loads embeddings instead of images (faster training).
            mae_embedding_csv_path: Path to the CSV that was used when extracting the embeddings
                               (i.e. the CSV whose row i corresponds to embedding row i).
                               Required when the localization csv_path and the extraction CSV
                               differ in size (e.g. kfold splits vs global dataset1 splits).
                               When set, embeddings are looked up by image_path instead of
                               positional index, so size mismatches are handled correctly.
            mae_embedding_array: Pre-loaded numpy array of embeddings. When provided, overrides
                               loading from mae_embedding_path. Use with mae_embedding_lookup_dict
                               for kfold where cells may span multiple dataset splits.
            mae_embedding_lookup_dict: Pre-built {image_path: row_idx} dict. When provided,
                               overrides building the lookup from mae_embedding_csv_path.
        """
        # Load image metadata CSV
        df = pd.read_csv(csv_path)

        # Check if we're using precomputed MAE embeddings
        self.use_mae_embeddings = mae_embedding_path is not None or mae_embedding_array is not None
        self.mae_embeddings = None
        self.mae_embedding_lookup = None  # image_path -> row_idx, built when sizes differ

        if self.use_mae_embeddings:
            # Load embedding array (pre-built takes priority over file path)
            if mae_embedding_array is not None:
                self.mae_embeddings = mae_embedding_array
            else:
                self.mae_embeddings = np.load(mae_embedding_path)

            # Build / set lookup
            if mae_embedding_lookup_dict is not None:
                # Pre-built lookup passed directly (e.g. combined all-splits lookup for kfold)
                self.mae_embedding_lookup = mae_embedding_lookup_dict
                print(f"Loaded MAE embeddings: {self.mae_embeddings.shape} "
                      f"(lookup by image_path, pre-built dict with {len(self.mae_embedding_lookup)} entries)")
            elif mae_embedding_csv_path is not None:
                # Build image_path -> row_idx lookup from the extraction source CSV.
                # This lets the localization CSV be a different size from the embedding array.
                src_df = pd.read_csv(mae_embedding_csv_path)
                assert len(src_df) == len(self.mae_embeddings), (
                    f"Embedding source CSV rows ({len(src_df)}) != "
                    f"embedding array rows ({len(self.mae_embeddings)})"
                )
                self.mae_embedding_lookup = {
                    row['image_path']: i
                    for i, (_, row) in enumerate(src_df.iterrows())
                }
                print(f"Loaded MAE embeddings: {self.mae_embeddings.shape} "
                      f"(lookup by image_path, source={mae_embedding_csv_path})")
            else:
                # Legacy positional mode: embedding array must align row-for-row with csv_path.
                assert len(self.mae_embeddings) == len(df), (
                    f"MAE embedding count ({len(self.mae_embeddings)}) doesn't match "
                    f"CSV rows ({len(df)}). Pass mae_embedding_csv_path to use "
                    f"image_path-based lookup instead."
                )
                print(f"Loaded MAE embeddings: {self.mae_embeddings.shape} (positional lookup)")

            print(f"  Mode: Embedding-only (fast training)")
        else:
            print(f"  Mode: Image loading (full forward pass)")

        # Add original index column before filtering
        df['original_idx'] = df.index

        # Load localization annotations
        loc_df = pd.read_csv(localization_csv_path)

        # Set default grade weights if not provided
        if grade_weights is None:
            grade_weights = {3: 1.0, 2: 0.5, 1: 0.25}
        self.grade_weights = grade_weights

        # Create protein name mapping (from metadata CSV)
        # The metadata CSV has 'file_gene_symbol' column
        if 'file_gene_symbol' in df.columns:
            df['protein_name'] = df['file_gene_symbol']
        elif 'folder_protein' in df.columns:
            df['protein_name'] = df['folder_protein']
        else:
            raise ValueError("CSV must contain 'file_gene_symbol' or 'folder_protein' column")

        # Merge with localization annotations on protein name
        # localization CSV has 'target_name' column
        loc_df = loc_df.rename(columns={'target_name': 'protein_name'})
        df_merged = df.merge(loc_df[['protein_name', 'annotations_grade_3',
                                      'annotations_grade_2', 'annotations_grade_1']],
                             on='protein_name', how='inner')

        # Filter out proteins without any annotations
        df_merged = df_merged.dropna(subset=['annotations_grade_3'], how='all')

        print(f"Loaded {len(df)} total images, {len(df_merged)} with localization annotations")

        # Store the original indices for embedding lookup
        self.original_indices = df_merged['original_idx'].tolist()

        # Store data
        self.image_paths = df_merged['image_path'].tolist()
        self.protein_names = df_merged['protein_name'].tolist()
        self.annotations_grade_3 = df_merged['annotations_grade_3'].tolist()
        self.annotations_grade_2 = df_merged['annotations_grade_2'].tolist()
        self.annotations_grade_1 = df_merged['annotations_grade_1'].tolist()

        self.transform = transform
        self.cache_rate = cache_rate
        self.use_max_projection = use_max_projection
        self.z_slice_start = z_slice_start
        self.z_slice_end = z_slice_end
        self._cache = {}

        # Log Z-slice selection if specified
        if z_slice_start is not None or z_slice_end is not None:
            print(f"  Z-slice selection: [{z_slice_start}:{z_slice_end}]")

        # Pre-cache a portion of the dataset if cache_rate > 0
        if cache_rate > 0.0:
            num_to_cache = int(len(self.image_paths) * cache_rate)
            print(f"Caching {num_to_cache}/{len(self.image_paths)} images...")

            def _load_image(idx):
                img = tifffile.imread(self.image_paths[idx])
                return idx, img

            # Use ThreadPoolExecutor for parallel loading
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = [executor.submit(_load_image, i) for i in range(num_to_cache)]
                for i, future in enumerate(futures):
                    idx, img = future.result()
                    self._cache[idx] = img
                    if (i + 1) % max(1, num_to_cache // 10) == 0:
                        print(f"  Cached {i + 1}/{num_to_cache} images")

            print(f"Caching complete! {len(self._cache)} images in memory.")

    def _parse_annotations(self, annotation_str):
        """Parse annotation string to list of labels."""
        if pd.isna(annotation_str) or annotation_str == '':
            return []
        return [label.strip() for label in annotation_str.split(';') if label.strip()]

    def _create_multilabel_target(self, idx):
        """
        Create multi-label target with weighted grades.
        Returns a tensor of shape [num_classes] with weights for each class.
        """
        target = torch.zeros(len(LOCALIZATION_LABELS), dtype=torch.float32)

        # Process each grade with its weight
        for grade_num, grade_col, weight in [
            (3, self.annotations_grade_3[idx], self.grade_weights.get(3, 1.0)),
            (2, self.annotations_grade_2[idx], self.grade_weights.get(2, 0.5)),
            (1, self.annotations_grade_1[idx], self.grade_weights.get(1, 0.25))
        ]:
            labels = self._parse_annotations(grade_col)
            for label in labels:
                if label in LABEL_TO_IDX:
                    target[LABEL_TO_IDX[label]] = max(target[LABEL_TO_IDX[label]], weight)

        return target

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Get original index for embedding lookup
        original_idx = self.original_indices[idx]

        # Create multi-label target
        target = self._create_multilabel_target(idx)

        # If using precomputed MAE embeddings, return embeddings only (no images)
        if self.use_mae_embeddings:
            if self.mae_embedding_lookup is not None:
                img_path = self.image_paths[idx]
                row_idx = self.mae_embedding_lookup[img_path]
            else:
                row_idx = original_idx
            mae_emb = torch.from_numpy(self.mae_embeddings[row_idx]).float()

            return {
                "mae_embedding": mae_emb,
                "label": target,
            }

        # Otherwise, load image as usual
        if idx in self._cache:
            img = self._cache[idx]
        else:
            img_path = self.image_paths[idx]
            img = tifffile.imread(img_path)  # Shape: (Z, C, Y, X)

        # Apply Z-slice selection if specified
        if self.z_slice_start is not None or self.z_slice_end is not None:
            z_start = self.z_slice_start if self.z_slice_start is not None else 0
            z_end = self.z_slice_end if self.z_slice_end is not None else img.shape[0]
            img = img[z_start:z_end]  # Shape: (Z_selected, C, Y, X)

        # Apply max projection if requested
        if self.use_max_projection:
            # Max projection along Z-axis: (Z, C, Y, X) -> (C, Y, X)
            img = np.max(img, axis=0)

        # Create data dict for MONAI transforms
        data = {"image": img, "label": target}

        # Apply transforms if provided
        if self.transform:
            data = self.transform(data)

        return data

    def get_label_distribution(self):
        """Compute label distribution across the dataset."""
        label_counts = {label: 0 for label in LOCALIZATION_LABELS}

        for idx in range(len(self)):
            target = self._create_multilabel_target(idx)
            for label_idx, count in enumerate(target):
                if count > 0:
                    label_counts[LOCALIZATION_LABELS[label_idx]] += 1

        return label_counts

    def get_cache_stats(self):
        """Return cache statistics."""
        return {
            'total_images': len(self.image_paths),
            'cached_images': len(self._cache),
            'cache_rate': self.cache_rate,
            'cache_hit_rate': len(self._cache) / len(self.image_paths) if len(self.image_paths) > 0 else 0.0
        }

    def get_mae_embedding_dim(self):
        """Return MAE embedding dimension (if using precomputed embeddings)."""
        if self.mae_embeddings is not None:
            return self.mae_embeddings.shape[1]
        return None

    def is_embedding_mode(self):
        """Return True if using precomputed MAE embeddings."""
        return self.use_mae_embeddings
