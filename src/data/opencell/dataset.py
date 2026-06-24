import pandas as pd
import tifffile
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch


class OpenCellDataset(Dataset):
    def __init__(self,
                 csv_path,
                 split='train',
                 transform=None,
                 cache_rate=0.0,
                 num_workers=4,
                 use_max_projection=False,
                 z_slice_start=None,
                 z_slice_end=None,
                 embedding_path=None,
                 esm2_embedding_path=None,
                 concat_embedding_path=None,
                 return_protein_label=False):
        """
        OpenCell Dataset that loads TIFF files with tifffile.

        Args:
            csv_path: Path to CSV file with 'image_path' column
            split: Dataset split ('train', 'val', 'test') - used for logging only
            transform: MONAI transforms to apply (expects pre-loaded numpy arrays)
            cache_rate: Fraction of dataset to cache in memory (0.0 to 1.0)
            num_workers: Number of workers for parallel caching
            use_max_projection: If True, apply max projection along Z-axis to create 2D images
            z_slice_start: Start index for Z-slice selection (inclusive). If None, use all slices.
            z_slice_end: End index for Z-slice selection (exclusive). If None, use all slices.
            embedding_path: Optional path to .npy file with precomputed embeddings (e.g., SubCell).
                           Embeddings must be in the same order as CSV rows.
                           If provided, returns 'teacher_embedding' in the data dict for distillation.
            esm2_embedding_path: Optional path to .npy file with ESM2 protein embeddings.
                                Embeddings must be in the same order as CSV rows.
                                If provided, returns 'esm2_embedding' in the data dict.
            concat_embedding_path: Optional path to .npy file with embeddings for decoder concatenation.
                                  Separate from embedding_path (which is for teacher/distillation).
                                  If provided, returns 'concat_embedding' in the data dict.
            return_protein_label: If True, return integer protein class label from 'folder_protein' column.
                                 Builds a sorted protein→index mapping. Returns 'protein_label' in data dict.
        """
        # Load CSV (no filtering needed since CSVs are already split)
        df = pd.read_csv(csv_path)

        # Store image paths
        self.image_paths = df['image_path'].tolist()
        self.transform = transform
        self.cache_rate = cache_rate
        self.use_max_projection = use_max_projection
        self.z_slice_start = z_slice_start
        self.z_slice_end = z_slice_end
        self._cache = {}

        # Build protein label mapping if requested
        self.return_protein_label = return_protein_label
        self.protein_labels = None
        self.protein_to_idx = None
        self.num_classes = 0
        if return_protein_label:
            proteins = df['folder_protein'].tolist()
            unique_proteins = sorted(set(proteins))
            self.protein_to_idx = {p: i for i, p in enumerate(unique_proteins)}
            self.protein_labels = [self.protein_to_idx[p] for p in proteins]
            self.num_classes = len(unique_proteins)
            print(f"  Protein labels: {self.num_classes} unique classes")

        # Load teacher embeddings if provided (for distillation)
        self.embeddings = None
        if embedding_path is not None:
            self.embeddings = np.load(embedding_path)
            assert len(self.embeddings) == len(self.image_paths), \
                f"Embedding count ({len(self.embeddings)}) doesn't match image count ({len(self.image_paths)})"
            print(f"Loaded teacher embeddings from {embedding_path}: shape {self.embeddings.shape}")

        # Load ESM2 embeddings if provided (for protein conditioning)
        self.esm2_embeddings = None
        if esm2_embedding_path is not None:
            self.esm2_embeddings = np.load(esm2_embedding_path)
            assert len(self.esm2_embeddings) == len(self.image_paths), \
                f"ESM2 embedding count ({len(self.esm2_embeddings)}) doesn't match image count ({len(self.image_paths)})"
            print(f"Loaded ESM2 embeddings from {esm2_embedding_path}: shape {self.esm2_embeddings.shape}")

        # Load concat embeddings if provided (for decoder concatenation)
        self.concat_embeddings = None
        if concat_embedding_path is not None:
            self.concat_embeddings = np.load(concat_embedding_path)
            assert len(self.concat_embeddings) == len(self.image_paths), \
                f"Concat embedding count ({len(self.concat_embeddings)}) doesn't match image count ({len(self.image_paths)})"
            print(f"Loaded concat embeddings from {concat_embedding_path}: shape {self.concat_embeddings.shape}")

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

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load from cache if available, otherwise load from disk
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
        data = {"image": img}

        # Apply transforms if provided
        if self.transform:
            data = self.transform(data)

        # Add teacher embedding if available (for distillation)
        if self.embeddings is not None:
            data["teacher_embedding"] = torch.from_numpy(self.embeddings[idx]).float()

        # Add ESM2 embedding if available (for protein conditioning)
        if self.esm2_embeddings is not None:
            data["esm2_embedding"] = torch.from_numpy(self.esm2_embeddings[idx]).float()

        # Add concat embedding if available (for decoder concatenation)
        if self.concat_embeddings is not None:
            data["concat_embedding"] = torch.from_numpy(self.concat_embeddings[idx]).float()

        # Add protein label if available (for supervised classification)
        if self.protein_labels is not None:
            data["protein_label"] = self.protein_labels[idx]

        return data

    def get_cache_stats(self):
        """Return cache statistics."""
        return {
            'total_images': len(self.image_paths),
            'cached_images': len(self._cache),
            'cache_rate': self.cache_rate,
            'cache_hit_rate': len(self._cache) / len(self.image_paths) if len(self.image_paths) > 0 else 0.0
        }

    def get_dataloader(self, batch_size=4,
                       shuffle=True, num_workers=4):
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
        )

    def get_embedding_dim(self):
        """Return embedding dimension if embeddings are loaded, else None."""
        if self.embeddings is not None:
            return self.embeddings.shape[1]
        return None


class OpenCellSliceDataset(Dataset):
    """
    OpenCell Dataset that returns individual 2D slices from 3D volumes.

    Instead of max-projection, this dataset samples slices from a specified
    Z-range, effectively increasing the number of training samples.
    For example, with 1000 volumes and slice_range (20, 80), the dataset
    contains 1000 * 61 = 61,000 samples.
    """

    def __init__(self,
                 csv_path,
                 split='train',
                 transform=None,
                 cache_rate=0.0,
                 num_workers=4,
                 slice_start=20,
                 slice_end=80):
        """
        Args:
            csv_path: Path to CSV file with 'image_path' column
            split: Dataset split ('train', 'val', 'test') - used for logging only
            transform: MONAI transforms to apply (expects pre-loaded numpy arrays)
            cache_rate: Fraction of dataset to cache in memory (0.0 to 1.0)
            num_workers: Number of workers for parallel caching
            slice_start: Start index of Z-slices to use (inclusive)
            slice_end: End index of Z-slices to use (inclusive)
        """
        # Load CSV
        df = pd.read_csv(csv_path)

        # Store image paths
        self.image_paths = df['image_path'].tolist()
        self.transform = transform
        self.cache_rate = cache_rate
        self.slice_start = slice_start
        self.slice_end = slice_end
        self._cache = {}

        # Number of slices per volume
        self.slices_per_volume = slice_end - slice_start + 1
        self.num_volumes = len(self.image_paths)

        print(f"OpenCellSliceDataset: {self.num_volumes} volumes × {self.slices_per_volume} slices = {len(self)} total samples")
        print(f"  Slice range: [{slice_start}, {slice_end}]")

        # Pre-cache volumes if cache_rate > 0
        if cache_rate > 0.0:
            num_to_cache = int(len(self.image_paths) * cache_rate)
            print(f"Caching {num_to_cache}/{len(self.image_paths)} volumes...")

            def _load_image(idx):
                img = tifffile.imread(self.image_paths[idx])
                return idx, img

            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = [executor.submit(_load_image, i) for i in range(num_to_cache)]
                for i, future in enumerate(futures):
                    idx, img = future.result()
                    self._cache[idx] = img
                    if (i + 1) % max(1, num_to_cache // 10) == 0:
                        print(f"  Cached {i + 1}/{num_to_cache} volumes")

            print(f"Caching complete! {len(self._cache)} volumes in memory.")

    def __len__(self):
        # Total samples = num_volumes * slices_per_volume
        return self.num_volumes * self.slices_per_volume

    def __getitem__(self, idx):
        # Map linear index to (volume_idx, slice_idx)
        volume_idx = idx // self.slices_per_volume
        slice_offset = idx % self.slices_per_volume
        slice_idx = self.slice_start + slice_offset

        # Load volume from cache or disk
        if volume_idx in self._cache:
            volume = self._cache[volume_idx]
        else:
            img_path = self.image_paths[volume_idx]
            volume = tifffile.imread(img_path)  # Shape: (Z, C, Y, X)

        # Extract the specific slice: (Z, C, Y, X) -> (C, Y, X)
        img = volume[slice_idx]  # Shape: (C, Y, X)

        # Create data dict for MONAI transforms
        data = {"image": img}

        # Apply transforms if provided
        if self.transform:
            data = self.transform(data)

        return data

    def get_cache_stats(self):
        """Return cache statistics."""
        return {
            'total_volumes': self.num_volumes,
            'total_samples': len(self),
            'slices_per_volume': self.slices_per_volume,
            'cached_volumes': len(self._cache),
            'cache_rate': self.cache_rate,
        }

    def get_dataloader(self, batch_size=4,
                       shuffle=True, num_workers=4):
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
        )
