"""
Plot UMAP visualization from pre-extracted MAE3D embeddings.

This script loads saved embeddings and creates UMAP visualizations
colored by ground-truth protein localization.

Supports single-modality and dual-modality (e.g., MAE3D vs SubCell) plots.
In dual mode, both embedding sets are projected into the same UMAP space
with different markers (circles vs triangles) so you can see if they converge.

Usage:
    # Single modality
    python src/plot_umap_from_embeddings.py --embeddings_path /path/to/embeddings.npy --output_dir /path/to/output --csv_path /path/to/split.csv

    # Dual modality
    python src/plot_umap_from_embeddings.py --embeddings_path /path/to/mae3d.npy --embeddings_path_2 /path/to/subcell.npy --output_dir /path/to/output --csv_path /path/to/split.csv --modality_names MAE3D SubCell
"""

import argparse
import os
import numpy as np
import pandas as pd
import umap
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap
import seaborn as sns


# Define all 17 localization labels (same as in dataset)
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


def get_primary_label(label_vector):
    """Get the primary (highest weighted) label for a sample."""
    if label_vector.max() == 0:
        return -1  # No label
    return np.argmax(label_vector)


def compute_umap(embeddings, n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42):
    """Compute UMAP embedding."""
    print(f"Computing UMAP (n_neighbors={n_neighbors}, min_dist={min_dist}, metric={metric})...")
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
        n_jobs=-1
    )
    umap_embeddings = reducer.fit_transform(embeddings)
    print(f"UMAP embedding shape: {umap_embeddings.shape}")
    return umap_embeddings


def plot_umap_all_classes(umap_embeddings, labels, output_path, figsize=(16, 14)):
    """
    Create UMAP visualization with all classes colored.

    Args:
        umap_embeddings: UMAP coordinates [N, 2]
        labels: Multi-label array [N, num_classes]
        output_path: Path to save the plot
        figsize: Figure size
    """
    # Get primary label for each sample
    primary_labels = np.array([get_primary_label(l) for l in labels])

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Get unique labels (excluding -1 for no label)
    unique_labels = np.unique(primary_labels[primary_labels >= 0])

    # Use a colormap with enough distinct colors
    n_classes = len(LOCALIZATION_LABELS)
    colors = plt.cm.get_cmap('tab20', n_classes)(np.linspace(0, 1, n_classes))

    # Plot each class
    for label_idx in unique_labels:
        mask = primary_labels == label_idx
        label_name = LOCALIZATION_LABELS[label_idx]
        ax.scatter(
            umap_embeddings[mask, 0],
            umap_embeddings[mask, 1],
            c=[colors[label_idx]],
            label=f"{label_name} ({mask.sum()})",
            alpha=0.6,
            s=15,
            edgecolors='none'
        )

    # Plot samples with no label (if any)
    no_label_mask = primary_labels == -1
    if no_label_mask.sum() > 0:
        ax.scatter(
            umap_embeddings[no_label_mask, 0],
            umap_embeddings[no_label_mask, 1],
            c='gray',
            label=f"No label ({no_label_mask.sum()})",
            alpha=0.3,
            s=5
        )

    ax.set_xlabel('UMAP 1', fontsize=14)
    ax.set_ylabel('UMAP 2', fontsize=14)
    ax.set_title('UMAP of MAE3D Embeddings by Protein Localization', fontsize=16)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9, markerscale=2)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved all-classes UMAP plot to {output_path}")


def plot_umap_per_class(umap_embeddings, labels, output_dir, figsize=(10, 8)):
    """Create individual UMAP plots highlighting each localization class."""
    os.makedirs(output_dir, exist_ok=True)

    for class_idx, class_name in enumerate(LOCALIZATION_LABELS):
        # Get samples where this class has weight > 0
        has_class = labels[:, class_idx] > 0
        n_positive = has_class.sum()

        if n_positive == 0:
            print(f"  Skipping {class_name}: no samples")
            continue

        fig, ax = plt.subplots(figsize=figsize)

        # Plot all samples in gray (background)
        ax.scatter(
            umap_embeddings[:, 0],
            umap_embeddings[:, 1],
            c='lightgray',
            alpha=0.3,
            s=5,
            edgecolors='none',
            label='Other'
        )

        # Highlight samples with this class
        ax.scatter(
            umap_embeddings[has_class, 0],
            umap_embeddings[has_class, 1],
            c='crimson',
            alpha=0.7,
            s=20,
            edgecolors='none',
            label=f'{class_name} (n={n_positive})'
        )

        ax.set_xlabel('UMAP 1', fontsize=12)
        ax.set_ylabel('UMAP 2', fontsize=12)
        ax.set_title(f'UMAP - {class_name}', fontsize=14)
        ax.legend(loc='upper right', fontsize=10)

        plt.tight_layout()
        output_path = os.path.join(output_dir, f'umap_{class_name}.png')
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        plt.close()

    print(f"Saved per-class UMAP plots to {output_dir}")


def plot_umap_multilabel(umap_embeddings, labels, output_path, figsize=(16, 14)):
    """
    Create UMAP visualization showing multi-label nature.
    Points are colored by primary label, with size indicating number of labels.
    """
    primary_labels = np.array([get_primary_label(l) for l in labels])
    num_labels_per_sample = (labels > 0).sum(axis=1)

    fig, ax = plt.subplots(figsize=figsize)

    unique_labels = np.unique(primary_labels[primary_labels >= 0])
    n_classes = len(LOCALIZATION_LABELS)
    colors = plt.cm.get_cmap('tab20', n_classes)(np.linspace(0, 1, n_classes))

    # Scale point sizes based on number of labels
    sizes = 10 + num_labels_per_sample * 5

    for label_idx in unique_labels:
        mask = primary_labels == label_idx
        label_name = LOCALIZATION_LABELS[label_idx]
        ax.scatter(
            umap_embeddings[mask, 0],
            umap_embeddings[mask, 1],
            c=[colors[label_idx]],
            s=sizes[mask],
            label=f"{label_name} ({mask.sum()})",
            alpha=0.6,
            edgecolors='none'
        )

    ax.set_xlabel('UMAP 1', fontsize=14)
    ax.set_ylabel('UMAP 2', fontsize=14)
    ax.set_title('UMAP - Point size indicates number of localization labels', fontsize=14)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9, markerscale=1.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved multi-label UMAP plot to {output_path}")


def plot_umap_dual_modality(umap_coords_1, umap_coords_2, labels, output_path,
                            modality_names=('Modality 1', 'Modality 2'), figsize=(18, 14)):
    """
    Plot two modalities in the same UMAP space.
    Both embedding sets must be projected jointly (concatenated before UMAP).
    Points are colored by localization class; modalities use different markers
    (circles for modality 1, triangles for modality 2).

    Args:
        umap_coords_1: UMAP coordinates for modality 1 [N, 2]
        umap_coords_2: UMAP coordinates for modality 2 [N, 2]
        labels: Multi-label array [N, num_classes] (shared, same samples)
        output_path: Path to save the plot
        modality_names: Tuple of (name1, name2)
        figsize: Figure size
    """
    primary_labels = np.array([get_primary_label(l) for l in labels])
    unique_labels = np.unique(primary_labels[primary_labels >= 0])

    n_classes = len(LOCALIZATION_LABELS)
    colors = plt.cm.get_cmap('tab20', n_classes)(np.linspace(0, 1, n_classes))

    fig, ax = plt.subplots(figsize=figsize)

    # Plot each class with both modalities
    for label_idx in unique_labels:
        mask = primary_labels == label_idx
        label_name = LOCALIZATION_LABELS[label_idx]
        c = [colors[label_idx]]

        # Modality 1: circles
        ax.scatter(
            umap_coords_1[mask, 0], umap_coords_1[mask, 1],
            c=c, marker='o', alpha=0.5, s=15, edgecolors='none',
            label=f"{label_name} - {modality_names[0]} ({mask.sum()})",
        )
        # Modality 2: triangles
        ax.scatter(
            umap_coords_2[mask, 0], umap_coords_2[mask, 1],
            c=c, marker='^', alpha=0.5, s=20, edgecolors='none',
            label=f"{label_name} - {modality_names[1]} ({mask.sum()})",
        )

    # No-label samples
    no_label_mask = primary_labels == -1
    if no_label_mask.sum() > 0:
        ax.scatter(umap_coords_1[no_label_mask, 0], umap_coords_1[no_label_mask, 1],
                   c='gray', marker='o', alpha=0.2, s=5,
                   label=f"No label - {modality_names[0]} ({no_label_mask.sum()})")
        ax.scatter(umap_coords_2[no_label_mask, 0], umap_coords_2[no_label_mask, 1],
                   c='gray', marker='^', alpha=0.2, s=5,
                   label=f"No label - {modality_names[1]} ({no_label_mask.sum()})")

    ax.set_xlabel('UMAP 1', fontsize=14)
    ax.set_ylabel('UMAP 2', fontsize=14)
    ax.set_title(f'UMAP: {modality_names[0]} (circles) vs {modality_names[1]} (triangles)', fontsize=16)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, markerscale=2, ncol=2)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved dual-modality UMAP plot to {output_path}")


def plot_umap_dual_modality_per_class(umap_coords_1, umap_coords_2, labels, output_dir,
                                       modality_names=('Modality 1', 'Modality 2'), figsize=(10, 8)):
    """Create per-class dual-modality UMAP plots highlighting one class at a time."""
    os.makedirs(output_dir, exist_ok=True)

    for class_idx, class_name in enumerate(LOCALIZATION_LABELS):
        has_class = labels[:, class_idx] > 0
        n_positive = has_class.sum()
        if n_positive == 0:
            print(f"  Skipping {class_name}: no samples")
            continue

        fig, ax = plt.subplots(figsize=figsize)

        # Background: all samples in gray
        ax.scatter(umap_coords_1[:, 0], umap_coords_1[:, 1],
                   c='lightgray', alpha=0.2, s=5, edgecolors='none')
        ax.scatter(umap_coords_2[:, 0], umap_coords_2[:, 1],
                   c='lightgray', alpha=0.2, s=5, edgecolors='none', marker='^')

        # Highlight this class
        ax.scatter(umap_coords_1[has_class, 0], umap_coords_1[has_class, 1],
                   c='crimson', marker='o', alpha=0.7, s=20, edgecolors='none',
                   label=f'{modality_names[0]} (n={n_positive})')
        ax.scatter(umap_coords_2[has_class, 0], umap_coords_2[has_class, 1],
                   c='dodgerblue', marker='^', alpha=0.7, s=25, edgecolors='none',
                   label=f'{modality_names[1]} (n={n_positive})')

        ax.set_xlabel('UMAP 1', fontsize=12)
        ax.set_ylabel('UMAP 2', fontsize=12)
        ax.set_title(f'{class_name}: {modality_names[0]} (circles) vs {modality_names[1]} (triangles)', fontsize=13)
        ax.legend(loc='upper right', fontsize=10)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'umap_dual_{class_name}.png'), dpi=200, bbox_inches='tight')
        plt.close()

    print(f"Saved per-class dual-modality UMAP plots to {output_dir}")


def plot_label_distribution(labels, output_path):
    """Plot bar chart of label distribution."""
    label_counts = (labels > 0).sum(axis=0)

    fig, ax = plt.subplots(figsize=(14, 6))

    x = np.arange(len(LOCALIZATION_LABELS))
    bars = ax.bar(x, label_counts, color='steelblue', edgecolor='black', linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(LOCALIZATION_LABELS, rotation=45, ha='right', fontsize=10)
    ax.set_xlabel('Localization Class', fontsize=12)
    ax.set_ylabel('Number of Samples', fontsize=12)
    ax.set_title('Distribution of Protein Localization Labels', fontsize=14)

    # Add count labels on bars
    for bar, count in zip(bars, label_counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                str(count), ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved label distribution plot to {output_path}")


def _parse_annotations(annotation_str, weight=1.0, num_classes=17):
    """Parse semicolon-separated annotation string into weighted label vector."""
    target = np.zeros(num_classes, dtype=np.float32)
    if pd.isna(annotation_str) or not isinstance(annotation_str, str) or annotation_str.strip() == '':
        return target
    for label in annotation_str.split(';'):
        label = label.strip()
        if label in LABEL_TO_IDX:
            target[LABEL_TO_IDX[label]] = max(target[LABEL_TO_IDX[label]], weight)
    return target


LABEL_TO_IDX = {label: idx for idx, label in enumerate(LOCALIZATION_LABELS)}


def load_labels_from_csv(csv_path, localization_csv_path, num_samples):
    """
    Load localization labels by merging metadata CSV with localization annotations.

    Args:
        csv_path: Path to metadata CSV (train.csv, val.csv, test.csv)
        localization_csv_path: Path to localization annotations CSV
        num_samples: Expected number of samples (must match embeddings)

    Returns:
        labels: numpy array of shape [N, 17] with weighted multi-labels
    """
    df = pd.read_csv(csv_path)
    assert len(df) == num_samples, \
        f"CSV has {len(df)} rows but embeddings have {num_samples} samples"

    # Load localization annotations
    loc_df = pd.read_csv(localization_csv_path)

    # Map protein name
    if 'file_gene_symbol' in df.columns:
        df['protein_name'] = df['file_gene_symbol']
    elif 'folder_protein' in df.columns:
        df['protein_name'] = df['folder_protein']
    else:
        raise ValueError("CSV must contain 'file_gene_symbol' or 'folder_protein' column")

    # Merge on protein name
    loc_df = loc_df.rename(columns={'target_name': 'protein_name'})
    df = df.merge(
        loc_df[['protein_name', 'annotations_grade_3', 'annotations_grade_2', 'annotations_grade_1']],
        on='protein_name', how='left'
    )

    # Build label vectors
    grade_weights = {3: 1.0, 2: 0.5, 1: 0.25}
    labels = np.zeros((len(df), len(LOCALIZATION_LABELS)), dtype=np.float32)
    for i in range(len(df)):
        for grade, col in [(3, 'annotations_grade_3'), (2, 'annotations_grade_2'), (1, 'annotations_grade_1')]:
            ann = df.iloc[i][col]
            target = _parse_annotations(ann, weight=grade_weights[grade])
            labels[i] = np.maximum(labels[i], target)

    print(f"Loaded labels for {len(df)} samples, {(labels.sum(axis=1) > 0).sum()} have annotations")
    return labels


def main():
    parser = argparse.ArgumentParser(description='Plot UMAP from pre-extracted embeddings')
    parser.add_argument('--embeddings_path', type=str, required=True,
                        help='Path to embeddings .npz or .npy file')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save plots')
    parser.add_argument('--csv_path', type=str, default=None,
                        help='Path to metadata CSV (required for .npy files without labels)')
    parser.add_argument('--localization_csv', type=str,
                        default='/path/to/datasets/opencell/opencell_metadata_raw/protein-localization-annotations/opencell-localization-annotations.csv',
                        help='Path to localization annotations CSV')
    parser.add_argument('--n_neighbors', type=int, default=15,
                        help='UMAP n_neighbors parameter')
    parser.add_argument('--min_dist', type=float, default=0.1,
                        help='UMAP min_dist parameter')
    parser.add_argument('--metric', type=str, default='cosine',
                        help='UMAP distance metric')
    parser.add_argument('--skip_per_class', action='store_true',
                        help='Skip generating per-class plots')
    parser.add_argument('--embeddings_path_2', type=str, default=None,
                        help='Path to second modality embeddings .npy file (for dual-modality UMAP)')
    parser.add_argument('--modality_names', type=str, nargs=2, default=['MAE3D', 'SubCell'],
                        help='Names for the two modalities (default: MAE3D SubCell)')
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load embeddings
    print(f"Loading embeddings from {args.embeddings_path}")
    data = np.load(args.embeddings_path, allow_pickle=True)

    if args.embeddings_path.endswith('.npz'):
        # .npz file with named arrays
        embeddings = data['embeddings']
        labels = data['labels']
        if 'label_names' in data.files:
            label_names = data['label_names']
            print(f"Label names from file: {label_names}")
    elif args.embeddings_path.endswith('.npy'):
        # .npy file — plain array of embeddings, labels loaded separately
        embeddings = data
        if args.csv_path is None:
            raise ValueError(
                "For .npy embedding files, --csv_path is required to load labels. "
                "Provide the metadata CSV path (e.g., .../metadata/dataset1/test.csv)"
            )
        labels = load_labels_from_csv(args.csv_path, args.localization_csv, len(embeddings))
    else:
        raise ValueError(f"Unsupported file format: {args.embeddings_path}. Use .npy or .npz")

    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Labels shape: {labels.shape}")

    # Print label statistics
    print("\nLabel statistics:")
    for i, name in enumerate(LOCALIZATION_LABELS):
        count = (labels[:, i] > 0).sum()
        print(f"  {name}: {count} samples")

    # Check for dual-modality mode
    dual_mode = args.embeddings_path_2 is not None
    if dual_mode:
        print(f"\nLoading second modality embeddings from {args.embeddings_path_2}")
        embeddings_2 = np.load(args.embeddings_path_2)
        if args.embeddings_path_2.endswith('.npz'):
            embeddings_2 = embeddings_2['embeddings']
        print(f"Embeddings 2 shape: {embeddings_2.shape}")
        assert len(embeddings) == len(embeddings_2), \
            f"Embedding counts must match: {len(embeddings)} vs {len(embeddings_2)}"

        # Joint UMAP: concatenate both, then split back
        print(f"\nComputing joint UMAP for {args.modality_names[0]} and {args.modality_names[1]}...")
        joint = np.concatenate([embeddings, embeddings_2], axis=0)
        joint_umap = compute_umap(
            joint,
            n_neighbors=args.n_neighbors,
            min_dist=args.min_dist,
            metric=args.metric,
        )
        n = len(embeddings)
        umap_coords_1 = joint_umap[:n]
        umap_coords_2 = joint_umap[n:]

        # Save joint UMAP coordinates
        umap_coords_path = os.path.join(args.output_dir, 'umap_coords_dual.npz')
        np.savez(umap_coords_path,
                 umap_coords_1=umap_coords_1, umap_coords_2=umap_coords_2,
                 labels=labels, modality_names=args.modality_names)
        print(f"Saved dual UMAP coordinates to {umap_coords_path}")

        # Dual-modality all-classes plot
        dual_path = os.path.join(args.output_dir, 'umap_dual_all_classes.png')
        plot_umap_dual_modality(umap_coords_1, umap_coords_2, labels, dual_path,
                                modality_names=tuple(args.modality_names))

        # Dual-modality per-class plots
        if not args.skip_per_class:
            dual_per_class_dir = os.path.join(args.output_dir, 'per_class_dual')
            plot_umap_dual_modality_per_class(umap_coords_1, umap_coords_2, labels,
                                               dual_per_class_dir,
                                               modality_names=tuple(args.modality_names))

        # Also generate individual single-modality plots
        print(f"\nGenerating individual UMAP plots for {args.modality_names[0]}...")
        all_classes_path_1 = os.path.join(args.output_dir, f'umap_all_classes_{args.modality_names[0]}.png')
        plot_umap_all_classes(umap_coords_1, labels, all_classes_path_1)

        print(f"Generating individual UMAP plots for {args.modality_names[1]}...")
        all_classes_path_2 = os.path.join(args.output_dir, f'umap_all_classes_{args.modality_names[1]}.png')
        plot_umap_all_classes(umap_coords_2, labels, all_classes_path_2)

    else:
        # Single-modality mode (original behavior)
        umap_embeddings = compute_umap(
            embeddings,
            n_neighbors=args.n_neighbors,
            min_dist=args.min_dist,
            metric=args.metric
        )

        # Save UMAP coordinates
        umap_coords_path = os.path.join(args.output_dir, 'umap_coords.npz')
        np.savez(umap_coords_path, umap_embeddings=umap_embeddings, labels=labels)
        print(f"Saved UMAP coordinates to {umap_coords_path}")

        # Plot all classes UMAP
        all_classes_path = os.path.join(args.output_dir, 'umap_all_classes.png')
        plot_umap_all_classes(umap_embeddings, labels, all_classes_path)

        # Plot multi-label UMAP
        multilabel_path = os.path.join(args.output_dir, 'umap_multilabel.png')
        plot_umap_multilabel(umap_embeddings, labels, multilabel_path)

        # Plot per-class UMAPs
        if not args.skip_per_class:
            per_class_dir = os.path.join(args.output_dir, 'per_class')
            plot_umap_per_class(umap_embeddings, labels, per_class_dir)

    # Plot label distribution (always)
    dist_path = os.path.join(args.output_dir, 'label_distribution.png')
    plot_label_distribution(labels, dist_path)

    print("\nDone! All plots saved to:", args.output_dir)


if __name__ == '__main__':
    main()
