"""
Extract embeddings from any trained MAE model.

Supports:
- MAE3D (standard)
- MAE3D Cross-Attention
- MAE2D

Usage:
    # MAE3D Cross-Attention
    python src/extract_mae_embeddings.py \
        --config configs/opencell/opencell_3d_cross_attention.yaml \
        --checkpoint /path/to/checkpoint.pth.tar \
        --output_dir /path/to/save/embeddings \
        --model_type mae3d_cross_attention

    # Standard MAE3D
    python src/extract_mae_embeddings.py \
        --config configs/opencell/opencell_3d.yaml \
        --checkpoint /path/to/checkpoint.pth.tar \
        --output_dir /path/to/save/embeddings \
        --model_type mae3d

    # MAE2D
    python src/extract_mae_embeddings.py \
        --config configs/opencell/opencell_2d.yaml \
        --checkpoint /path/to/checkpoint.pth.tar \
        --output_dir /path/to/save/embeddings \
        --model_type mae2d

Output:
    - train.npy: [N_train, embed_dim]
    - val.npy: [N_val, embed_dim]
    - test.npy: [N_test, embed_dim]
    - metadata.json: embedding info (model_type, embed_dim, checkpoint, etc.)
"""

import os
import sys
import json
import argparse
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))

from utils.utils import get_conf
from data.opencell.dataset import OpenCellDataset
from data.opencell.transforms import get_opencell_val_transforms, get_opencell_2d_val_transforms


def build_model(args, model_type):
    """Build MAE model based on type."""

    if model_type in ('mae3d_cross_attention', 'mae3d_cross_attention_esm2'):
        if model_type == 'mae3d_cross_attention_esm2':
            from lib.models.mae3d_cross_attention_esm2 import MAE3DChannelCrossAttentionESM2
            model = MAE3DChannelCrossAttentionESM2(args)
        else:
            from lib.models.mae3d_cross_attention import MAE3DChannelCrossAttention
            model = MAE3DChannelCrossAttention(args)

    elif model_type == 'mae3d':
        from lib.models.mae3d import MAE3D
        from lib.networks.mae_vit import MAEViTEncoder, MAEViTDecoder

        model = MAE3D(MAEViTEncoder, MAEViTDecoder, args)

    elif model_type == 'mae2d_cross_attention':
        from lib.models.mae2d_cross_attention_fft import MAE2DChannelCrossAttentionFFT
        model = MAE2DChannelCrossAttentionFFT(args)

    elif model_type == 'mae2d':
        from lib.models.mae2d import MAE2D
        from lib.networks.mae_vit import MAEViTEncoder, MAEViTDecoder

        model = MAE2D(MAEViTEncoder, MAEViTDecoder, args)

    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    return model


def load_checkpoint(model, checkpoint_path):
    """Load checkpoint into model."""
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    state_dict = checkpoint.get('state_dict', checkpoint)

    # Remove 'module.' prefix if present
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v

    msg = model.load_state_dict(new_state_dict, strict=False)
    print(f"  Missing keys: {len(msg.missing_keys)}")
    print(f"  Unexpected keys: {len(msg.unexpected_keys)}")

    return model


def extract_embeddings_mae3d_cross_attention(model, dataloader, device, pool_mode='concat', use_global_pool=True):
    """Extract embeddings from MAE3D Cross-Attention model."""
    model.eval()
    all_embeddings = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting"):
            images = batch['image'].to(device)

            with torch.cuda.amp.autocast(True):
                # Forward through encoder
                x_ch0, x_ch1 = model.forward_encoder_no_mask(images)

                # Pool features (same as ViT3DCrossAttentionClassifier)
                if use_global_pool:
                    # Global average pooling over spatial tokens (skip CLS)
                    feat_ch0 = x_ch0[:, 1:, :].mean(dim=1)
                    feat_ch1 = x_ch1[:, 1:, :].mean(dim=1)
                else:
                    # Use CLS tokens
                    feat_ch0 = x_ch0[:, 0]
                    feat_ch1 = x_ch1[:, 0]

                # Combine channels
                if pool_mode == 'concat':
                    features = torch.cat([feat_ch0, feat_ch1], dim=-1)
                elif pool_mode == 'mean':
                    features = (feat_ch0 + feat_ch1) / 2
                elif pool_mode == 'sum':
                    features = feat_ch0 + feat_ch1
                else:
                    raise ValueError(f"Unknown pool_mode: {pool_mode}")

            all_embeddings.append(features.cpu().numpy())

    return np.concatenate(all_embeddings, axis=0)


def extract_embeddings_mae3d(model, dataloader, device, use_global_pool=True):
    """Extract embeddings from standard MAE3D model."""
    from lib.models.mae3d import patchify_image

    model.eval()
    all_embeddings = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting"):
            images = batch['image'].to(device)
            B = images.shape[0]

            with torch.cuda.amp.autocast(True):
                # Patchify
                x = patchify_image(images, model.patch_size)

                # Get positional embeddings
                pos_embed = model.encoder_pos_embed.expand(B, -1, -1)

                # Forward through encoder
                features = model.encoder.forward_features(x, pos_embed)

                # Pool features
                if use_global_pool:
                    # Global average pooling (skip CLS if present)
                    if features.shape[1] > 1:
                        features = features[:, 1:, :].mean(dim=1)
                    else:
                        features = features.mean(dim=1)
                else:
                    # Use CLS token
                    features = features[:, 0]

            all_embeddings.append(features.cpu().numpy())

    return np.concatenate(all_embeddings, axis=0)


def extract_embeddings_mae2d(model, dataloader, device, use_global_pool=True):
    """Extract embeddings from MAE2D model."""
    from lib.models.mae2d import patchify_image

    model.eval()
    all_embeddings = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting"):
            images = batch['image'].to(device)
            B = images.shape[0]

            with torch.cuda.amp.autocast(True):
                # Patchify
                x = patchify_image(images, model.patch_size)

                # Get positional embeddings
                pos_embed = model.encoder_pos_embed.expand(B, -1, -1)

                # Forward through encoder
                features = model.encoder.forward_features(x, pos_embed)

                # Pool features
                if use_global_pool:
                    if features.shape[1] > 1:
                        features = features[:, 1:, :].mean(dim=1)
                    else:
                        features = features.mean(dim=1)
                else:
                    features = features[:, 0]

            all_embeddings.append(features.cpu().numpy())

    return np.concatenate(all_embeddings, axis=0)


def main():
    parser = argparse.ArgumentParser(description='Extract MAE embeddings')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for embeddings')
    parser.add_argument('--model_type', type=str, required=True,
                        choices=['mae3d', 'mae3d_cross_attention', 'mae3d_cross_attention_esm2',
                                 'mae2d_cross_attention', 'mae2d'],
                        help='Type of MAE model')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--workers', type=int, default=4, help='Number of workers')
    parser.add_argument('--pool_mode', type=str, default='concat',
                        choices=['concat', 'mean', 'sum'],
                        help='How to combine channel features (for cross-attention)')
    parser.add_argument('--use_global_pool', action='store_true', default=True,
                        help='Use global average pooling (vs CLS token)')
    parser.add_argument('--splits', type=str, default='train,val,test',
                        help='Comma-separated list of splits to process')
    cmd_args = parser.parse_args()

    # Load config
    code_base_path = '/path/to/repository/'
    config_path = cmd_args.config
    if not os.path.isabs(config_path):
        config_path = os.path.join(code_base_path, config_path)

    print(f'Loading config from: {config_path}')
    args = get_conf(config_path)

    # Create output directory
    os.makedirs(cmd_args.output_dir, exist_ok=True)
    print(f'Output directory: {cmd_args.output_dir}')

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Build model
    print(f'\n{"="*60}')
    print(f'Building {cmd_args.model_type} model...')
    print(f'{"="*60}')

    model = build_model(args, cmd_args.model_type)
    model = load_checkpoint(model, cmd_args.checkpoint)
    model = model.to(device)
    model.eval()

    # Freeze model
    for param in model.parameters():
        param.requires_grad = False

    # Determine embedding dimension
    if cmd_args.model_type in ('mae3d_cross_attention', 'mae3d_cross_attention_esm2',
                               'mae2d_cross_attention'):
        if cmd_args.pool_mode == 'concat':
            embed_dim = args.encoder_embed_dim * 2
        else:
            embed_dim = args.encoder_embed_dim
    else:
        embed_dim = args.encoder_embed_dim

    print(f'Embedding dimension: {embed_dim}')

    # Get transforms
    if cmd_args.model_type in ('mae2d', 'mae2d_cross_attention'):
        transform = get_opencell_2d_val_transforms()
        use_max_projection = True
    else:
        transform = get_opencell_val_transforms()
        use_max_projection = False

    # Process each split
    splits = cmd_args.splits.split(',')

    for split in splits:
        csv_path = os.path.join(args.csv_path, f'{split}.csv')

        if not os.path.exists(csv_path):
            print(f'\nSkipping {split}: {csv_path} not found')
            continue

        print(f'\n{"="*60}')
        print(f'Processing {split} split...')
        print(f'{"="*60}')

        # Create dataset
        dataset = OpenCellDataset(
            csv_path=csv_path,
            split=split,
            transform=transform,
            cache_rate=0.0,
            num_workers=cmd_args.workers,
            use_max_projection=use_max_projection,
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

        # Extract embeddings based on model type
        if cmd_args.model_type in ('mae3d_cross_attention', 'mae3d_cross_attention_esm2',
                                   'mae2d_cross_attention'):
            embeddings = extract_embeddings_mae3d_cross_attention(
                model, dataloader, device,
                pool_mode=cmd_args.pool_mode,
                use_global_pool=cmd_args.use_global_pool
            )
        elif cmd_args.model_type == 'mae3d':
            embeddings = extract_embeddings_mae3d(
                model, dataloader, device,
                use_global_pool=cmd_args.use_global_pool
            )
        elif cmd_args.model_type == 'mae2d':
            embeddings = extract_embeddings_mae2d(
                model, dataloader, device,
                use_global_pool=cmd_args.use_global_pool
            )

        print(f'  Extracted embeddings shape: {embeddings.shape}')

        # Save embeddings
        output_path = os.path.join(cmd_args.output_dir, f'{split}.npy')
        np.save(output_path, embeddings)
        print(f'  Saved to: {output_path}')

    # Save metadata
    metadata = {
        'model_type': cmd_args.model_type,
        'checkpoint': cmd_args.checkpoint,
        'config': config_path,
        'embed_dim': embed_dim,
        'pool_mode': cmd_args.pool_mode if cmd_args.model_type in (
            'mae3d_cross_attention', 'mae3d_cross_attention_esm2', 'mae2d_cross_attention'
        ) else None,
        'use_global_pool': cmd_args.use_global_pool,
    }

    metadata_path = os.path.join(cmd_args.output_dir, 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f'\nSaved metadata to: {metadata_path}')

    print(f'\n{"="*60}')
    print('Embedding extraction complete!')
    print(f'{"="*60}')


if __name__ == '__main__':
    main()
