"""
Create a standalone WTC-11 cell info CSV.

For each cell in the 50k subset this script records:
  - where the raw OME-TIFF lives (file_path)
  - the exact 3-D bounding box to crop from that file (roi_*)
  - the actual and target crop dimensions
  - which channels to use (DNA=0, structure/protein=2)
  - biological labels (structure_name for protein localisation,
    cell_stage for cell-cycle phase)

The CSV is self-contained: hand it to anyone and they can
  1. load the OME-TIFF at `file_path`  — it is ALREADY the cell crop (Z, C, Y, X)
  2. keep only channels [dna_channel, structure_channel]  → (Z, 2, Y, X)
  3. centre-crop / pad to (target_z, target_xy, target_xy) → always same shape
  4. look up the ESM2 protein embedding by `structure_name` (gene symbol)

NOTE on roi_* columns
---------------------
roi_z0/z1, roi_y0/y1, roi_x0/x1 are the bounding box coordinates of this
cell inside the *original full FOV image* (not inside crop_raw).  They are
provided as reference — e.g. for embedding methods (DINO4Cell, SubCell) that
operate on the full FOV and need to know where the cell is located.
Do NOT apply these indices to the crop_raw file; it is already cropped.

Preprocessing recipe
--------------------
target_z  = 80   (p5 of Z distribution, rounded to multiple of 10)
target_xy = 224  (p5 of XY distribution, rounded to multiple of 8)

For images larger than the target: centre-crop.
For images smaller than the target: centre-pad with zeros.

Usage
-----
    python src/wtc/create_wtc_info_csv.py \
        --metadata_path /ictstr01/groups/labs/marr/qscd01/datasets/SingleCellImagesDataset/wtc11/metadata_50k_subset.csv \
        --data_root     /ictstr01/groups/labs/marr/qscd01/datasets/SingleCellImagesDataset/wtc11 \
        --output_path   /ictstr01/groups/labs/marr/qscd01/datasets/SingleCellImagesDataset/wtc11/wtc11_cells_info.csv
"""

import argparse
import ast
import os

import pandas as pd


TARGET_Z  = 80    # p5 of Z distribution, multiple of 10 (patch_size_z=10)
TARGET_XY = 224   # p5 of XY distribution, multiple of 8  (patch_size_xy=8)
DNA_CHANNEL       = 0   # channel index in the raw OME-TIFF
STRUCTURE_CHANNEL = 2   # channel index in the raw OME-TIFF


def parse_args():
    parser = argparse.ArgumentParser(description="Create WTC-11 cell info CSV")
    parser.add_argument(
        "--metadata_path",
        default="/ictstr01/groups/labs/marr/qscd01/datasets/SingleCellImagesDataset/wtc11/metadata_50k_subset.csv",
    )
    parser.add_argument(
        "--data_root",
        default="/ictstr01/groups/labs/marr/qscd01/datasets/SingleCellImagesDataset/wtc11",
        help="Root directory where crop_raw/ folder lives",
    )
    parser.add_argument(
        "--output_path",
        default="/ictstr01/groups/labs/marr/qscd01/datasets/SingleCellImagesDataset/wtc11/wtc11_cells_info.csv",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Reading metadata from {args.metadata_path} ...")
    df = pd.read_csv(args.metadata_path, low_memory=False)
    print(f"  {len(df)} cells loaded")

    # ------------------------------------------------------------------ #
    # Parse ROI  →  [z0, z1, y0, y1, x0, x1]                            #
    # ------------------------------------------------------------------ #
    rois = df["roi"].apply(ast.literal_eval)
    roi_df = pd.DataFrame(rois.tolist(), columns=["roi_z0", "roi_z1", "roi_y0", "roi_y1", "roi_x0", "roi_x1"])

    # Actual crop dimensions
    roi_df["z_size"] = roi_df["roi_z1"] - roi_df["roi_z0"]
    roi_df["y_size"] = roi_df["roi_y1"] - roi_df["roi_y0"]
    roi_df["x_size"] = roi_df["roi_x1"] - roi_df["roi_x0"]

    # ------------------------------------------------------------------ #
    # Construct absolute file paths                                       #
    # ------------------------------------------------------------------ #
    file_paths = df["crop_raw"].apply(lambda rp: os.path.join(args.data_root, rp))

    # ------------------------------------------------------------------ #
    # Assemble output dataframe                                           #
    # ------------------------------------------------------------------ #
    out = pd.DataFrame({
        # --- Identity ---
        "cell_id":          df["CellId"].values,

        # --- File location ---
        "file_path":        file_paths.values,

        # --- Bounding box inside the OME-TIFF (0-based, exclusive end) ---
        "roi_z0":           roi_df["roi_z0"].values,
        "roi_z1":           roi_df["roi_z1"].values,
        "roi_y0":           roi_df["roi_y0"].values,
        "roi_y1":           roi_df["roi_y1"].values,
        "roi_x0":           roi_df["roi_x0"].values,
        "roi_x1":           roi_df["roi_x1"].values,

        # --- Actual crop dimensions (z1-z0, y1-y0, x1-x0) ---
        "z_size":           roi_df["z_size"].values,
        "y_size":           roi_df["y_size"].values,
        "x_size":           roi_df["x_size"].values,

        # --- Target dimensions after centre-crop / pad ---
        "target_z":         TARGET_Z,
        "target_xy":        TARGET_XY,

        # --- Channel indices to load from OME-TIFF ---
        "dna_channel":      DNA_CHANNEL,
        "structure_channel": STRUCTURE_CHANNEL,

        # --- Biological labels ---
        # structure_name: gene symbol of the tagged protein
        #   → use as protein-localisation class label (25 classes)
        #   → use as key to look up ESM2 protein-sequence embedding
        "structure_name":   df["structure_name"].values,

        # cell_stage: cell-cycle phase (6 classes, highly imbalanced)
        #   M0 (interphase) ~94.6 %
        #   M1M2, M3, M4M5, M6M7_single, M6M7_complete are mitotic stages
        "cell_stage":       df["cell_stage"].values,

        # FOVId: field-of-view identifier
        #   cells from the same FOV share imaging conditions → use FOVId
        #   (not CellId) as the unit of splitting for train/val/test
        "fov_id":           df["FOVId"].values,
    })

    # ------------------------------------------------------------------ #
    # Save                                                                 #
    # ------------------------------------------------------------------ #
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    out.to_csv(args.output_path, index=False)
    print(f"\nSaved {len(out)} rows → {args.output_path}")

    # ------------------------------------------------------------------ #
    # Summary                                                              #
    # ------------------------------------------------------------------ #
    print("\n--- Dimension summary (actual crop sizes) ---")
    for dim in ["z_size", "y_size", "x_size"]:
        s = out[dim]
        print(f"  {dim}: min={s.min()}, max={s.max()}, "
              f"mean={s.mean():.1f}, median={s.median():.1f}")

    print(f"\n--- Target dimensions ---")
    print(f"  Z  : {TARGET_Z}  (centre-crop or pad to this)")
    print(f"  XY : {TARGET_XY} (centre-crop or pad to this)")
    print(f"  Channels used: DNA={DNA_CHANNEL}, Structure={STRUCTURE_CHANNEL}")

    print(f"\n--- structure_name distribution (25 classes) ---")
    print(out["structure_name"].value_counts().to_string())

    print(f"\n--- cell_stage distribution ---")
    print(out["cell_stage"].value_counts().to_string())

    print(f"\n--- FOV summary ---")
    print(f"  {out['fov_id'].nunique()} unique FOVs, avg {len(out)/out['fov_id'].nunique():.1f} cells/FOV")

    print(f"\n--- ESM2 note ---")
    print(f"  {out['structure_name'].nunique()} unique gene symbols → one ESM2 embedding per symbol.")
    print(f"  Symbols: {sorted(out['structure_name'].unique())}")


if __name__ == "__main__":
    main()
