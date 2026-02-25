"""
Download a stratified subset of WTC-11 hiPSC single-cell raw crop images.

Uses the Allen Cell quilt3 package to download 3D z-stack OME-TIFF crops.
Resume-friendly: skips files that already exist on disk.
"""

import argparse
import logging
import os

import pandas as pd
import quilt3
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download a stratified subset of WTC-11 single-cell raw crops"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="data/wtc11",
        help="Root directory for downloaded data",
    )
    parser.add_argument(
        "--metadata_path",
        type=str,
        default="notebooks/wtc/metadata.csv",
        help="Path to the full metadata CSV",
    )
    parser.add_argument(
        "--n_subset",
        type=int,
        default=50_000,
        help="Number of cells to sample",
    )
    parser.add_argument(
        "--shard_idx",
        type=int,
        default=0,
        help="Index of this shard (0-based)",
    )
    parser.add_argument(
        "--n_shards",
        type=int,
        default=1,
        help="Total number of shards (1 = no sharding)",
    )
    return parser.parse_args()


def create_subset(metadata_path: str, n_subset: int, seed: int = 42) -> pd.DataFrame:
    """Load metadata, filter outliers/edge cells, and stratified-sample."""
    df = pd.read_csv(metadata_path, low_memory=False)
    print(f"Total cells in metadata: {len(df)}")

    df_clean = df[(df["outlier"] == "No") & (df["edge_flag"] == 0)].copy()
    print(f"After removing outliers + edge cells: {len(df_clean)}")

    if n_subset >= len(df_clean):
        print(f"Requested {n_subset} >= available {len(df_clean)}, using all clean cells")
        return df_clean

    # Stratified sample proportional to structure_name
    df_subset = df_clean.groupby("structure_name", group_keys=False).apply(
        lambda x: x.sample(
            n=max(1, int(n_subset * len(x) / len(df_clean))),
            random_state=seed,
        )
    )

    # Fill remaining quota if rounding caused a shortfall
    if len(df_subset) < n_subset:
        remaining = df_clean[~df_clean.index.isin(df_subset.index)]
        extra = remaining.sample(n=n_subset - len(df_subset), random_state=seed)
        df_subset = pd.concat([df_subset, extra])
    elif len(df_subset) > n_subset:
        df_subset = df_subset.sample(n=n_subset, random_state=seed)

    print(f"Subset size: {len(df_subset)}")
    print(f"\nStructure distribution:\n{df_subset['structure_name'].value_counts().sort_index()}")
    print(f"\nCell stage distribution:\n{df_subset['cell_stage'].value_counts()}")
    return df_subset


def download_crops(df_subset: pd.DataFrame, save_dir: str, shard_idx: int = 0, n_shards: int = 1):
    """Download crop_raw files from the Allen Cell quilt3 package."""
    pkg = quilt3.Package.browse(
        "aics/hipsc_single_cell_image_dataset", "s3://allencell"
    )

    raw_dir = os.path.join(save_dir, "crop_raw")
    os.makedirs(raw_dir, exist_ok=True)

    error_log = os.path.join(save_dir, "download_errors.log")
    logger = logging.getLogger("download")
    logger.setLevel(logging.WARNING)
    fh = logging.FileHandler(error_log)
    fh.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
    logger.addHandler(fh)

    raw_paths = df_subset["crop_raw"].tolist()
    raw_paths = raw_paths[shard_idx::n_shards]
    print(f"Shard {shard_idx}/{n_shards}: {len(raw_paths)} files to process")
    skipped = 0
    failed = 0

    for rp in tqdm(raw_paths, desc="Downloading raw crops"):
        dest = os.path.join(save_dir, rp)
        if os.path.exists(dest):
            skipped += 1
            continue
        try:
            pkg[rp].fetch(dest)
        except Exception as e:
            failed += 1
            logger.warning(f"FAILED {rp}: {e}")

    print(f"\nDownload complete: {len(raw_paths)} total, {skipped} skipped (existed), {failed} failed")
    if failed > 0:
        print(f"See {error_log} for details on failed downloads")


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    # Create stratified subset
    df_subset = create_subset(args.metadata_path, args.n_subset)

    # Save subset metadata
    subset_csv = os.path.join(args.save_dir, "metadata_50k_subset.csv")
    df_subset.to_csv(subset_csv, index=False)
    print(f"\nSaved subset metadata to {subset_csv}")

    # Download raw crops
    download_crops(df_subset, args.save_dir, args.shard_idx, args.n_shards)


if __name__ == "__main__":
    main()
