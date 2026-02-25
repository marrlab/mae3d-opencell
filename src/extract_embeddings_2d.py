"""
Extract embeddings from MAE2D model and visualize with UMAP.

This script:
1. Loads a trained MAE2D model
2. Extracts embeddings from the test set using the encoder
3. Saves embeddings with localization labels
4. Creates UMAP visualization colored by localization
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from omegaconf import OmegaConf

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from lib.models.mae2d import MAE2D, patchify_image, build_2d_sincos_position_embedding
from lib.networks.mae_vit import MAEViTEncoder, MAEViTDecoder
from data.opencell.localization_dataset import (
    OpenCellLocalizationDataset,
    LOCALIZATION_LABELS,
    LABEL_TO_IDX
)
from data.opencell.transforms import get_opencell_2d_val_transforms


def load_config(config_path):
    """Load config from yaml file using OmegaConf to resolve interpolations."""
    config = OmegaConf.load(config_path)
    OmegaConf.resolve(config)  # Resolve ${...} interpolations
    return config


def build_mae2d_model(args):
    """Build MAE2D model from config."""
    from lib.networks import patch_embed_layers

    encoder = MAEViTEncoder
    decoder = MAEViTDecoder

    model = MAE2D(encoder, decoder, args)
    return model


def load_checkpoint(model, checkpoint_path, device):
    """Load model weights from checkpoint."""
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Handle different checkpoint formats
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    # Remove 'module.' prefix if present (from DDP training)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict, strict=True)
    print(f"Loaded checkpoint (epoch: {checkpoint.get('epoch', 'unknown')})")
    return model


def extract_embeddings(model, dataloader, device, pooling='mean'):
    """
    Extract embeddings from the encoder.

    Args:
        model: MAE2D model
        dataloader: DataLoader for the dataset
        device: torch device
        pooling: 'mean' for mean pooling, 'cls' for CLS token

    Returns:
        embeddings: numpy array of shape [N, embed_dim]
        labels: numpy array of shape [N, num_classes]
    """
    model.eval()
    all_embeddings = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting embeddings"):
            images = batch['image'].to(device)  # [B, C, H, W] = [B, 2, 176, 176]
            labels = batch['label']  # [B, num_classes]

            batch_size = images.size(0)

            # Patchify the images
            x = patchify_image(images, model.patch_size)  # [B, num_patches, patch_dim]

            # Get positional embeddings
            pos_embed = model.encoder_pos_embed.expand(batch_size, -1, -1).to(device)

            # Forward through encoder (with all patches, no masking)
            features = model.encoder.forward_features(x, pos_embed)  # [B, num_patches+1, embed_dim]

            # Pooling
            if pooling == 'cls':
                embeddings = features[:, 0, :]  # CLS token
            else:
                embeddings = features[:, 1:, :].mean(dim=1)  # Mean of patch embeddings (exclude CLS)

            all_embeddings.append(embeddings.cpu().numpy())
            all_labels.append(labels.numpy())

    embeddings = np.concatenate(all_embeddings, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    return embeddings, labels


def get_primary_label(label_vector):
    """Get the primary (highest weighted) label for a sample."""
    if label_vector.max() == 0:
        return -1  # No label
    return np.argmax(label_vector)


def plot_umap(embeddings, labels, output_path, n_neighbors=15, min_dist=0.1, metric='cosine'):
    """
    Create UMAP visualization colored by localization.

    Args:
        embeddings: numpy array of shape [N, embed_dim]
        labels: numpy array of shape [N, num_classes]
        output_path: path to save the plot
        n_neighbors: UMAP parameter
        min_dist: UMAP parameter
        metric: UMAP distance metric
    """
    import umap
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    print("Computing UMAP embedding...")
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=42,
        n_jobs=-1
    )
    umap_embeddings = reducer.fit_transform(embeddings)

    # Get primary label for each sample
    primary_labels = np.array([get_primary_label(l) for l in labels])

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 12))

    # Get unique labels (excluding -1 for no label)
    unique_labels = np.unique(primary_labels[primary_labels >= 0])

    # Use a colormap with enough colors
    colors = cm.tab20(np.linspace(0, 1, len(LOCALIZATION_LABELS)))

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
            s=10
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

    ax.set_xlabel('UMAP 1', fontsize=12)
    ax.set_ylabel('UMAP 2', fontsize=12)
    ax.set_title('UMAP of MAE2D Embeddings by Localization', fontsize=14)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved UMAP plot to {output_path}")

    # Also save UMAP coordinates
    umap_output = output_path.replace('.png', '_coords.npz')
    np.savez(umap_output, umap_embeddings=umap_embeddings, primary_labels=primary_labels)
    print(f"Saved UMAP coordinates to {umap_output}")

    return umap_embeddings


def plot_umap_per_class(embeddings, labels, output_dir, n_neighbors=15, min_dist=0.1, metric='cosine'):
    """Create individual UMAP plots for each localization class."""
    import umap
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    print("Computing UMAP embedding for per-class visualization...")
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=42,
        n_jobs=-1
    )
    umap_embeddings = reducer.fit_transform(embeddings)

    # Create per-class plots
    os.makedirs(output_dir, exist_ok=True)

    for class_idx, class_name in enumerate(LOCALIZATION_LABELS):
        # Get samples where this class has weight > 0
        has_class = labels[:, class_idx] > 0

        fig, ax = plt.subplots(figsize=(10, 8))

        # Plot all samples in gray
        ax.scatter(
            umap_embeddings[:, 0],
            umap_embeddings[:, 1],
            c='lightgray',
            alpha=0.3,
            s=5,
            label='Other'
        )

        # Highlight samples with this class
        ax.scatter(
            umap_embeddings[has_class, 0],
            umap_embeddings[has_class, 1],
            c='red',
            alpha=0.7,
            s=15,
            label=f'{class_name} ({has_class.sum()})'
        )

        ax.set_xlabel('UMAP 1', fontsize=12)
        ax.set_ylabel('UMAP 2', fontsize=12)
        ax.set_title(f'UMAP - {class_name}', fontsize=14)
        ax.legend()

        plt.tight_layout()
        output_path = os.path.join(output_dir, f'umap_{class_name}.png')
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        plt.close()

    print(f"Saved per-class UMAP plots to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Extract MAE2D embeddings and visualize with UMAP')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config.yaml from the trained model or project config')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save embeddings and plots')
    parser.add_argument('--localization_csv', type=str,
                        default='/path/to/datasets/opencell/opencell_metadata_raw/protein-localization-annotations/opencell-localization-annotations.csv',
                        help='Path to localization annotations CSV')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for inference (can be larger for 2D)')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='Number of data loading workers')
    parser.add_argument('--pooling', type=str, default='mean', choices=['mean', 'cls'],
                        help='Pooling method for embeddings')
    parser.add_argument('--n_neighbors', type=int, default=15,
                        help='UMAP n_neighbors parameter')
    parser.add_argument('--min_dist', type=float, default=0.1,
                        help='UMAP min_dist parameter')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--split', type=str, default='test',
                        help='Dataset split to use (train, val, test)')
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load config
    config = load_config(args.config)
    print(f"Loaded config from {args.config}")

    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Build model
    model = build_mae2d_model(config)
    model = load_checkpoint(model, args.checkpoint, device)
    model = model.to(device)
    model.eval()

    # Create dataset and dataloader
    csv_path = os.path.join(config.csv_path, f'{args.split}.csv')
    print(f"Loading dataset from {csv_path}")

    # 2D uses max projection - use 2D-specific transforms
    transform = get_opencell_2d_val_transforms(
        channel_wise_norm=getattr(config, 'channel_wise_norm', True)
    )

    dataset = OpenCellLocalizationDataset(
        csv_path=csv_path,
        localization_csv_path=args.localization_csv,
        split=args.split,
        transform=transform,
        cache_rate=0.0,
        num_workers=args.num_workers,
        use_max_projection=True  # 2D model uses max projection: (Z,C,Y,X) -> (C,Y,X)
    )

    print(f"Dataset size: {len(dataset)} samples")

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # Extract embeddings
    print(f"Extracting embeddings with {args.pooling} pooling...")
    embeddings, labels = extract_embeddings(model, dataloader, device, pooling=args.pooling)
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Labels shape: {labels.shape}")

    # Save embeddings
    embeddings_path = os.path.join(args.output_dir, f'embeddings_{args.split}.npz')
    np.savez(
        embeddings_path,
        embeddings=embeddings,
        labels=labels,
        label_names=LOCALIZATION_LABELS
    )
    print(f"Saved embeddings to {embeddings_path}")

    # Create UMAP visualization
    umap_path = os.path.join(args.output_dir, f'umap_{args.split}.png')
    plot_umap(
        embeddings, labels, umap_path,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist
    )

    # Create per-class UMAP plots
    per_class_dir = os.path.join(args.output_dir, f'umap_per_class_{args.split}')
    plot_umap_per_class(
        embeddings, labels, per_class_dir,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist
    )

    print("Done!")


if __name__ == '__main__':
    main()
