"""
Extract MAE embeddings for WTC-11 fold-specific CSVs.

Same logic as extract_mae_embeddings.py but uses WTCDataset instead of
OpenCellDataset.  Processes train and val splits (no test in WTC kfold).

Usage
-----
    # 3D Cross-Attention (FFT or CLIP checkpoint)
    python src/extract_mae_embeddings_wtc.py \
        --config  configs/wtc/wtc_3d_cross_attention_fft_kfold.yaml \
        --checkpoint /path/to/checkpoint.pth.tar \
        --output_dir /path/to/mae3d_embeddings \
        --csv_path   /ictstr01/.../wtc11/kfold5/fold0 \
        --model_type mae3d_cross_attention

    # 2D Cross-Attention
    python src/extract_mae_embeddings_wtc.py \
        --config  configs/wtc/wtc_2d_cross_attention_fft_kfold.yaml \
        --checkpoint /path/to/checkpoint.pth.tar \
        --output_dir /path/to/mae2d_embeddings \
        --csv_path   /ictstr01/.../wtc11/kfold5/fold0 \
        --model_type mae2d_cross_attention
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))

from utils.utils import get_conf
from data.wtc.dataset import WTCDataset
from data.opencell.transforms import get_opencell_val_transforms, get_opencell_2d_val_transforms

# Re-use model building and extraction logic from the OpenCell script
from extract_mae_embeddings import (
    build_model,
    load_checkpoint,
    extract_embeddings_mae3d_cross_attention,
    extract_embeddings_mae2d,
    extract_embeddings_mae3d,
)


def main():
    parser = argparse.ArgumentParser(
        description='Extract MAE embeddings for WTC-11 (per-fold)'
    )
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--csv_path', type=str, required=True,
                        help='Directory containing train.csv and val.csv for this fold.')
    parser.add_argument(
        '--model_type', type=str, required=True,
        choices=['mae3d_cross_attention', 'mae2d_cross_attention', 'mae3d', 'mae2d'],
    )
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--pool_mode', type=str, default='concat',
                        choices=['concat', 'mean', 'sum'])
    parser.add_argument('--use_global_pool', action='store_true', default=True)
    parser.add_argument('--splits', type=str, default='train,val',
                        help='Comma-separated splits (default: train,val).')
    cmd = parser.parse_args()

    # ── Config ───────────────────────────────────────────────────────────────
    code_base = '/lustre/groups/aih/amirhossein.kardoost/codes/single-cell-foundation-model/single-cell-foundation-model/'
    config_path = cmd.config if os.path.isabs(cmd.config) else os.path.join(code_base, cmd.config)
    args = get_conf(config_path)

    os.makedirs(cmd.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('=' * 60)
    print(f'WTC-11 MAE embedding extraction')
    print(f'Model:      {cmd.model_type}')
    print(f'Checkpoint: {cmd.checkpoint}')
    print(f'CSV dir:    {cmd.csv_path}')
    print(f'Output:     {cmd.output_dir}')
    print(f'Splits:     {cmd.splits}')
    print('=' * 60)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_model(args, cmd.model_type)
    model = load_checkpoint(model, cmd.checkpoint)
    model = model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    if cmd.model_type in ('mae3d_cross_attention', 'mae2d_cross_attention'):
        embed_dim = args.encoder_embed_dim * 2 if cmd.pool_mode == 'concat' else args.encoder_embed_dim
    else:
        embed_dim = args.encoder_embed_dim
    print(f'Embedding dim: {embed_dim}')

    # ── Transforms ────────────────────────────────────────────────────────────
    use_max_projection = cmd.model_type in ('mae2d', 'mae2d_cross_attention')
    transform = get_opencell_2d_val_transforms() if use_max_projection else get_opencell_val_transforms()

    # ── Process splits ────────────────────────────────────────────────────────
    for split in cmd.splits.split(','):
        csv_file = os.path.join(cmd.csv_path, f'{split}.csv')
        if not os.path.exists(csv_file):
            print(f'\nSkipping {split}: {csv_file} not found')
            continue

        print(f'\n{"="*60}\nProcessing {split} split ...\n{"="*60}')

        dataset = WTCDataset(
            csv_path=csv_file,
            split=split,
            transform=transform,
            cache_rate=0.0,
            use_max_projection=use_max_projection,
        )
        print(f'  Samples: {len(dataset)}')

        dataloader = DataLoader(
            dataset,
            batch_size=cmd.batch_size,
            shuffle=False,
            num_workers=cmd.workers,
            pin_memory=True,
        )

        if cmd.model_type in ('mae3d_cross_attention', 'mae2d_cross_attention'):
            embeddings = extract_embeddings_mae3d_cross_attention(
                model, dataloader, device,
                pool_mode=cmd.pool_mode,
                use_global_pool=cmd.use_global_pool,
            )
        elif cmd.model_type == 'mae3d':
            embeddings = extract_embeddings_mae3d(model, dataloader, device,
                                                   use_global_pool=cmd.use_global_pool)
        else:
            embeddings = extract_embeddings_mae2d(model, dataloader, device,
                                                   use_global_pool=cmd.use_global_pool)

        out_path = os.path.join(cmd.output_dir, f'{split}.npy')
        np.save(out_path, embeddings)
        print(f'  Saved {embeddings.shape} → {out_path}')

    # ── Metadata ──────────────────────────────────────────────────────────────
    metadata = {
        'model_type': cmd.model_type,
        'checkpoint': cmd.checkpoint,
        'config': config_path,
        'embed_dim': embed_dim,
        'pool_mode': cmd.pool_mode,
        'use_global_pool': cmd.use_global_pool,
        'csv_path': cmd.csv_path,
    }
    with open(os.path.join(cmd.output_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f'\n{"="*60}\nExtraction complete!\n{"="*60}')


if __name__ == '__main__':
    main()
