"""
Extract embeddings from MAE3D Cross-Attention model and visualize with 2D UMAP.

This script:
1. Checks if embeddings already exist - if so, loads them
2. Otherwise, loads a trained MAE3DChannelCrossAttention model and extracts embeddings
3. Creates 2D UMAP visualization colored by localization

Usage:
    python src/extract_embeddings_3d_cross_attention.py \
        --config configs/opencell/opencell_3d_cross_attention.yaml \
        --checkpoint /path/to/checkpoint.pth.tar \
        --output_dir /path/to/output
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

from lib.models.mae3d_cross_attention import (
    MAE3DChannelCrossAttention,
    patchify_image_channelwise
)
from data.opencell.localization_dataset import (
    OpenCellLocalizationDataset,
    LOCALIZATION_LABELS,
    LABEL_TO_IDX
)
from data.opencell.transforms import get_opencell_val_transforms


def load_config(config_path):
    """Load config from yaml file using OmegaConf to resolve interpolations."""
    config = OmegaConf.load(config_path)
    OmegaConf.resolve(config)  # Resolve ${...} interpolations
    return config


def build_model(args):
    """Build MAE3DChannelCrossAttention model from config."""
    model = MAE3DChannelCrossAttention(args)
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


def extract_embeddings(model, dataloader, device, pooling='mean', pool_mode='concat'):
    """
    Extract embeddings from the dual-stream cross-attention encoder.

    Args:
        model: MAE3DChannelCrossAttention model
        dataloader: DataLoader for the dataset
        device: torch device
        pooling: 'mean' for mean pooling over patches, 'cls' for CLS token
        pool_mode: 'concat' to concatenate both channels, 'mean' to average them

    Returns:
        embeddings: numpy array of shape [N, embed_dim] or [N, embed_dim*2] for concat
        labels: numpy array of shape [N, num_classes]
    """
    model.eval()
    all_embeddings = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting embeddings"):
            images = batch['image'].to(device)  # [B, C, Z, Y, X] = [B, 2, 100, 176, 176]
            labels = batch['label']  # [B, num_classes]

            B = images.size(0)
            args = model.args

            # Patchify per channel
            x_channels = patchify_image_channelwise(images, model.patch_size)

            # Embed patches per channel (no masking for inference)
            embedded_channels = []
            for i, (x_ch, embed, cls_token) in enumerate(
                zip(x_channels, model.patch_embeds, model.cls_tokens)):

                # Project patches
                x_emb = embed(x_ch)  # [B, num_patches, embed_dim]

                # Add CLS token
                cls = cls_token.expand(B, -1, -1)
                x_emb = torch.cat([cls, x_emb], dim=1)

                # Add position embedding (full, no masking)
                pos_embed = model.encoder_pos_embed.expand(B, -1, -1)
                cls_pe = torch.zeros(B, 1, args.encoder_embed_dim, device=device)
                pos_embed_full = torch.cat([cls_pe, pos_embed], dim=1)

                x_emb = model.pos_drop(x_emb + pos_embed_full)
                embedded_channels.append(x_emb)

            # Process through encoder blocks with cross-attention
            x_ch0, x_ch1 = embedded_channels[0], embedded_channels[1]
            for block in model.encoder_blocks:
                x_ch0, x_ch1 = block(x_ch0, x_ch1)

            # Normalize
            x_ch0 = model.encoder_norms[0](x_ch0)
            x_ch1 = model.encoder_norms[1](x_ch1)

            # Pooling
            if pooling == 'cls':
                emb_ch0 = x_ch0[:, 0, :]  # CLS token
                emb_ch1 = x_ch1[:, 0, :]
            else:
                emb_ch0 = x_ch0[:, 1:, :].mean(dim=1)  # Mean of patches (exclude CLS)
                emb_ch1 = x_ch1[:, 1:, :].mean(dim=1)

            # Combine channel embeddings
            if pool_mode == 'concat':
                embeddings = torch.cat([emb_ch0, emb_ch1], dim=1)  # [B, embed_dim*2]
            elif pool_mode == 'mean':
                embeddings = (emb_ch0 + emb_ch1) / 2  # [B, embed_dim]
            elif pool_mode == 'sum':
                embeddings = emb_ch0 + emb_ch1  # [B, embed_dim]
            else:
                raise ValueError(f"Unknown pool_mode: {pool_mode}")

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


def plot_umap_2d(embeddings, labels, output_path, n_neighbors=15, min_dist=0.1, metric='cosine'):
    """
    Create 2D UMAP visualization colored by localization.

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

    print("Computing 2D UMAP embedding...")
    reducer = umap.UMAP(
        n_components=2,  # 2D UMAP
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
    ax.set_title('2D UMAP of MAE3D Cross-Attention Embeddings by Localization', fontsize=14)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved 2D UMAP plot to {output_path}")

    # Also save UMAP coordinates
    umap_output = output_path.replace('.png', '_coords.npz')
    np.savez(umap_output, umap_embeddings=umap_embeddings, primary_labels=primary_labels)
    print(f"Saved 2D UMAP coordinates to {umap_output}")

    return umap_embeddings


def plot_umap_2d_per_class(umap_embeddings, labels, output_dir):
    """Create individual 2D UMAP plots for each localization class."""
    import matplotlib.pyplot as plt

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
        ax.set_title(f'2D UMAP - {class_name}', fontsize=14)
        ax.legend()

        plt.tight_layout()
        output_path = os.path.join(output_dir, f'umap_2d_{class_name}.png')
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        plt.close()

    print(f"Saved per-class 2D UMAP plots to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Extract MAE3D Cross-Attention embeddings and visualize with 2D UMAP')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config.yaml from the trained model')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save embeddings and plots')
    parser.add_argument('--localization_csv', type=str,
                        default='/path/to/datasets/opencell/opencell_metadata_raw/protein-localization-annotations/opencell-localization-annotations.csv',
                        help='Path to localization annotations CSV')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size for inference')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--pooling', type=str, default='mean', choices=['mean', 'cls'],
                        help='Pooling method for embeddings')
    parser.add_argument('--pool_mode', type=str, default='concat', choices=['concat', 'mean', 'sum'],
                        help='How to combine channel embeddings')
    parser.add_argument('--n_neighbors', type=int, default=15,
                        help='UMAP n_neighbors parameter')
    parser.add_argument('--min_dist', type=float, default=0.1,
                        help='UMAP min_dist parameter')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--split', type=str, default='test',
                        help='Dataset split to use (train, val, test)')
    parser.add_argument('--per_class', action='store_true',
                        help='Create per-class UMAP plots')
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Check if embeddings already exist
    embeddings_path = os.path.join(args.output_dir, f'embeddings_{args.split}.npz')

    if os.path.exists(embeddings_path):
        # Load existing embeddings
        print(f"Found existing embeddings at {embeddings_path}")
        print("Loading embeddings...")
        data = np.load(embeddings_path)
        embeddings = data['embeddings']
        labels = data['labels']
        print(f"Loaded embeddings shape: {embeddings.shape}")
        print(f"Loaded labels shape: {labels.shape}")
    else:
        # Extract embeddings from model
        print("No existing embeddings found. Extracting from model...")

        # Load config
        config = load_config(args.config)
        print(f"Loaded config from {args.config}")

        # Set device
        device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")

        # Build model
        model = build_model(config)
        model = load_checkpoint(model, args.checkpoint, device)
        model = model.to(device)
        model.eval()

        # Create dataset and dataloader
        csv_path = os.path.join(config.csv_path, f'{args.split}.csv')
        print(f"Loading dataset from {csv_path}")

        transform = get_opencell_val_transforms(
            channel_wise_norm=getattr(config, 'channel_wise_norm', True)
        )

        dataset = OpenCellLocalizationDataset(
            csv_path=csv_path,
            localization_csv_path=args.localization_csv,
            split=args.split,
            transform=transform,
            cache_rate=0.0,
            num_workers=args.num_workers,
            use_max_projection=False
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
        print(f"Extracting embeddings with {args.pooling} pooling and {args.pool_mode} channel combination...")
        embeddings, labels = extract_embeddings(
            model, dataloader, device,
            pooling=args.pooling,
            pool_mode=args.pool_mode
        )
        print(f"Embeddings shape: {embeddings.shape}")
        print(f"Labels shape: {labels.shape}")

        # Save embeddings
        np.savez(
            embeddings_path,
            embeddings=embeddings,
            labels=labels,
            label_names=LOCALIZATION_LABELS,
            pooling=args.pooling,
            pool_mode=args.pool_mode
        )
        print(f"Saved embeddings to {embeddings_path}")

    # Create 2D UMAP visualization
    umap_path = os.path.join(args.output_dir, f'umap_2d_{args.split}.png')
    umap_embeddings = plot_umap_2d(
        embeddings, labels, umap_path,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist
    )

    # Create per-class 2D UMAP plots if requested
    if args.per_class:
        per_class_dir = os.path.join(args.output_dir, f'umap_2d_per_class_{args.split}')
        plot_umap_2d_per_class(umap_embeddings, labels, per_class_dir)

    print("Done!")


if __name__ == '__main__':
    main()
