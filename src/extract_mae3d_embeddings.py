"""
Extract MAE3D Cross-Attention embeddings for all samples.

This script extracts and saves embeddings from a pretrained MAE3D encoder
for later use in fast classifier training (no image loading needed).

Usage:
    python src/extract_mae3d_embeddings.py \
        --config configs/opencell/opencell_localization_3d_cross_attention_subcell_fusion.yaml \
        --output_dir /path/to/save/embeddings

Output:
    - train_mae3d_embeddings.npy: [N_train, 768]
    - val_mae3d_embeddings.npy: [N_val, 768]
    - test_mae3d_embeddings.npy: [N_test, 768]
"""

import os
import sys
import argparse
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))

from utils.utils import get_conf
from lib.models import ViT3DCrossAttentionSubCellClassifier
from data.opencell.localization_fusion_dataset import OpenCellLocalizationFusionDataset
from data.opencell.transforms import get_opencell_val_transforms


def extract_embeddings(model, dataloader, device, desc="Extracting"):
    """Extract MAE3D embeddings for all samples in dataloader."""
    model.eval()
    all_embeddings = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=desc):
            images = batch['image'].to(device)

            with torch.cuda.amp.autocast(True):
                # Extract only MAE3D features (not the full forward)
                mae_features = model.forward_mae_features(images)

            all_embeddings.append(mae_features.cpu().numpy())

    return np.concatenate(all_embeddings, axis=0)


def main():
    parser = argparse.ArgumentParser(description='Extract MAE3D embeddings')
    parser.add_argument('--config', type=str,
                        default='configs/opencell/opencell_localization_3d_cross_attention_subcell_fusion.yaml',
                        help='Path to config file')
    parser.add_argument('--pretrain', type=str, default=None,
                        help='Path to pretrained MAE3D checkpoint (overrides config)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for embeddings (default: same as subcell embeddings)')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for extraction')
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of data loader workers')
    cmd_args = parser.parse_args()

    # Load config
    code_base_path = '/lustre/groups/aih/amirhossein.kardoost/codes/single-cell-foundation-model/single-cell-foundation-model/'
    config_path = cmd_args.config
    if not os.path.isabs(config_path):
        config_path = os.path.join(code_base_path, config_path)

    print(f'Loading config from: {config_path}')
    args = get_conf(config_path)

    # Override pretrain if specified
    if cmd_args.pretrain is not None:
        args.pretrain = cmd_args.pretrain

    # Output directory
    if cmd_args.output_dir is not None:
        output_dir = cmd_args.output_dir
    else:
        # Save alongside subcell embeddings
        output_dir = os.path.dirname(args.embedding_path)
        output_dir = os.path.join(output_dir, 'mae3d_cross_attention')

    os.makedirs(output_dir, exist_ok=True)
    print(f'Output directory: {output_dir}')

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Build model
    print(f'\n{"="*60}')
    print('Building MAE3D Cross-Attention model...')
    print(f'{"="*60}')

    model_params = {
        'input_size': tuple(args.input_size),
        'patch_size': tuple(args.patch_size),
        'in_chans': args.in_chans,
        'num_classes': args.num_classes,
        'embed_dim': args.encoder_embed_dim,
        'depth': args.encoder_depth,
        'num_heads': args.encoder_num_heads,
        'drop_path_rate': getattr(args, 'drop_path', 0.0),
        'pos_embed_type': getattr(args, 'pos_embed_type', 'sincos'),
        'use_global_pool': getattr(args, 'use_global_pool', True),
        'cross_attention_type': getattr(args, 'cross_attention_type', 'position_wise'),
        'pool_mode': getattr(args, 'pool_mode', 'concat'),
        'subcell_embed_dim': getattr(args, 'subcell_embed_dim', 1536),
        'subcell_proj_dim': getattr(args, 'subcell_proj_dim', None),
        'fusion_type': getattr(args, 'fusion_type', 'concat'),
    }

    model = ViT3DCrossAttentionSubCellClassifier(**model_params)

    # Load pretrained weights
    if args.pretrain is not None and os.path.exists(args.pretrain):
        print(f'Loading pretrained weights from {args.pretrain}')
        checkpoint = torch.load(args.pretrain, map_location='cpu')
        state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
        model.load_mae_encoder(state_dict, strict=False)
        print('Successfully loaded pretrained MAE encoder weights')
    else:
        raise ValueError(f'Pretrained checkpoint required: {args.pretrain}')

    model = model.to(device)
    model.eval()

    # Freeze model (we're only extracting, not training)
    for param in model.parameters():
        param.requires_grad = False

    print(f'\nMAE3D embedding dimension: {model.mae_feature_dim}')

    # Create transforms (no augmentation for embedding extraction)
    transform = get_opencell_val_transforms()

    # Get embedding paths for SubCell (to match sample order)
    embedding_base_path = args.embedding_path

    # Process each split
    splits = ['train', 'val', 'test']

    for split in splits:
        csv_path = os.path.join(args.csv_path, f'{split}.csv')
        subcell_emb_path = os.path.join(embedding_base_path, f'{split}.npy')

        if not os.path.exists(csv_path):
            print(f'\nSkipping {split}: {csv_path} not found')
            continue

        if not os.path.exists(subcell_emb_path):
            print(f'\nSkipping {split}: {subcell_emb_path} not found')
            continue

        print(f'\n{"="*60}')
        print(f'Processing {split} split...')
        print(f'{"="*60}')

        # Create dataset
        dataset = OpenCellLocalizationFusionDataset(
            csv_path=csv_path,
            localization_csv_path=args.localization_csv_path,
            embedding_path=subcell_emb_path,
            split=split,
            transform=transform,
            cache_rate=0.0,
            num_workers=cmd_args.workers,
            use_max_projection=False,
            grade_weights=getattr(args, 'grade_weights', None),
        )

        print(f'  Samples: {len(dataset)}')

        # Create dataloader (no shuffling to preserve order)
        dataloader = DataLoader(
            dataset,
            batch_size=cmd_args.batch_size,
            shuffle=False,
            num_workers=cmd_args.workers,
            pin_memory=True,
        )

        # Extract embeddings
        embeddings = extract_embeddings(model, dataloader, device, desc=f'Extracting {split}')

        print(f'  Extracted embeddings shape: {embeddings.shape}')

        # Save embeddings
        output_path = os.path.join(output_dir, f'{split}.npy')
        np.save(output_path, embeddings)
        print(f'  Saved to: {output_path}')

        # Verify
        loaded = np.load(output_path)
        print(f'  Verified: {loaded.shape}')

    print(f'\n{"="*60}')
    print('Embedding extraction complete!')
    print(f'Embeddings saved to: {output_dir}')
    print(f'{"="*60}')


if __name__ == '__main__':
    main()
