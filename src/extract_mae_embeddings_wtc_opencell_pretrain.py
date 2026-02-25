"""
Extract WTC-11 MAE embeddings using an OpenCell-pretrained model (zero-shot).

The OpenCell model was trained on (100, 176, 176) with patch_size [10, 8, 8].
WTC images are (80, 224, 224) — same patch_size, different grid.

Position embeddings (nn.Parameter, size-dependent) are re-initialised via
sincos for the WTC grid (8, 28, 28).  All other encoder weights (patch
projection, transformer blocks, layer norms) transfer cleanly because they
are independent of the number of patches.

The key difference from extract_mae_embeddings_wtc.py is the checkpoint
loader: it explicitly filters keys whose shape does not match the current
model so that position-embedding size mismatches are silently skipped
rather than raising a RuntimeError.

Usage
-----
    python src/extract_mae_embeddings_wtc_opencell_pretrain.py \\
        --config   configs/wtc/wtc_3d_cross_attention_opencell_pretrain_kfold.yaml \\
        --checkpoint /ictstr01/.../opencell/mae_opencell_3d_cross_attention_clip_kfold5/run3/fold0/ckpts/checkpoint_0004.pth.tar \\
        --output_dir /ictstr01/.../wtc11/mae_wtc_3d_opencell_pretrain/fold0/mae3d_embeddings \\
        --csv_path   /ictstr01/.../wtc11/kfold5/fold0 \\
        --pool_mode  concat
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
from data.opencell.transforms import get_opencell_val_transforms
from lib.models.mae3d_cross_attention import MAE3DChannelCrossAttention


# ── Checkpoint loader ─────────────────────────────────────────────────────────

def load_checkpoint_cross_dataset(model, checkpoint_path):
    """
    Load a checkpoint, skipping keys whose tensor shape does not match the
    current model (e.g. position embeddings from a different grid size).

    This is intentionally separate from the shared load_checkpoint() so that
    cross-dataset transfer loading does not affect other pipelines.
    """
    print(f"Loading checkpoint from: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    state_dict = ckpt.get('state_dict', ckpt)

    # Strip 'module.' prefix (DDP checkpoints)
    stripped = {}
    for k, v in state_dict.items():
        stripped[k[7:] if k.startswith('module.') else k] = v

    # Filter keys whose shape mismatches the current model
    model_sd = model.state_dict()
    filtered, skipped = {}, []
    for k, v in stripped.items():
        if k in model_sd and v.shape != model_sd[k].shape:
            skipped.append(
                f"  SKIP  {k}: ckpt {tuple(v.shape)}  vs  model {tuple(model_sd[k].shape)}"
            )
        else:
            filtered[k] = v

    if skipped:
        print(f"Skipped {len(skipped)} key(s) due to shape mismatch "
              f"(expected for cross-dataset transfer):")
        for s in skipped:
            print(s)

    msg = model.load_state_dict(filtered, strict=False)
    print(f"  Missing keys  : {len(msg.missing_keys)}")
    print(f"  Unexpected keys: {len(msg.unexpected_keys)}")
    return model


# ── Embedding extraction ──────────────────────────────────────────────────────

def extract_embeddings(model, dataloader, device, pool_mode='concat', use_global_pool=True):
    """Extract embeddings from MAE3DChannelCrossAttention encoder (no masking)."""
    model.eval()
    all_embeddings = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting"):
            images = batch['image'].to(device)

            with torch.cuda.amp.autocast(True):
                x_ch0, x_ch1 = model.forward_encoder_no_mask(images)

                if use_global_pool:
                    feat_ch0 = x_ch0[:, 1:, :].mean(dim=1)
                    feat_ch1 = x_ch1[:, 1:, :].mean(dim=1)
                else:
                    feat_ch0 = x_ch0[:, 0]
                    feat_ch1 = x_ch1[:, 0]

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


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Extract WTC-11 MAE embeddings from an OpenCell-pretrained model (zero-shot)'
    )
    parser.add_argument('--config',     type=str, required=True,
                        help='WTC architecture config (e.g. configs/wtc/wtc_3d_cross_attention_opencell_pretrain_kfold.yaml)')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to the OpenCell-trained checkpoint (.pth.tar)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save train.npy / val.npy + metadata.json')
    parser.add_argument('--csv_path',   type=str, required=True,
                        help='Directory containing train.csv and val.csv for this fold')
    parser.add_argument('--pool_mode',  type=str, default='concat',
                        choices=['concat', 'mean', 'sum'])
    parser.add_argument('--use_global_pool', action='store_true', default=True)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--workers',    type=int, default=8)
    parser.add_argument('--splits',     type=str, default='train,val')
    cmd = parser.parse_args()

    # ── Config ───────────────────────────────────────────────────────────────
    code_base = '/lustre/groups/aih/amirhossein.kardoost/codes/single-cell-foundation-model/single-cell-foundation-model/'
    config_path = cmd.config if os.path.isabs(cmd.config) else os.path.join(code_base, cmd.config)
    args = get_conf(config_path)

    os.makedirs(cmd.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    embed_dim = args.encoder_embed_dim * 2 if cmd.pool_mode == 'concat' else args.encoder_embed_dim

    print('=' * 60)
    print('WTC-11  zero-shot embedding extraction (OpenCell pretrain)')
    print(f'Checkpoint : {cmd.checkpoint}')
    print(f'WTC config : {config_path}')
    print(f'Input size : {args.input_size}  patch {args.patch_size}')
    print(f'Pool mode  : {cmd.pool_mode}  →  embed_dim={embed_dim}')
    print(f'CSV dir    : {cmd.csv_path}')
    print(f'Output     : {cmd.output_dir}')
    print('=' * 60)

    # ── Model ─────────────────────────────────────────────────────────────────
    # Build with WTC input size (fresh sincos pos embeddings for WTC grid)
    model = MAE3DChannelCrossAttention(args)
    # Load OpenCell weights; pos-embed size mismatch is handled by skipping
    model = load_checkpoint_cross_dataset(model, cmd.checkpoint)
    model = model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    # ── Transforms ────────────────────────────────────────────────────────────
    transform = get_opencell_val_transforms()

    # ── Process splits ────────────────────────────────────────────────────────
    for split in cmd.splits.split(','):
        csv_file = os.path.join(cmd.csv_path, f'{split}.csv')
        if not os.path.exists(csv_file):
            print(f'\nSkipping {split}: {csv_file} not found')
            continue

        print(f'\n{"=" * 60}\nProcessing {split} split ...\n{"=" * 60}')

        dataset = WTCDataset(
            csv_path=csv_file,
            split=split,
            transform=transform,
            cache_rate=0.0,
        )
        print(f'  Samples: {len(dataset)}')

        dataloader = DataLoader(
            dataset,
            batch_size=cmd.batch_size,
            shuffle=False,
            num_workers=cmd.workers,
            pin_memory=True,
        )

        embeddings = extract_embeddings(
            model, dataloader, device,
            pool_mode=cmd.pool_mode,
            use_global_pool=cmd.use_global_pool,
        )

        out_path = os.path.join(cmd.output_dir, f'{split}.npy')
        np.save(out_path, embeddings)
        print(f'  Saved {embeddings.shape} → {out_path}')

    # ── Metadata ──────────────────────────────────────────────────────────────
    metadata = {
        'source': 'opencell_pretrain_zero_shot',
        'checkpoint': cmd.checkpoint,
        'config': config_path,
        'model_type': 'mae3d_cross_attention',
        'input_size': list(args.input_size),
        'patch_size': list(args.patch_size),
        'embed_dim': embed_dim,
        'pool_mode': cmd.pool_mode,
        'use_global_pool': cmd.use_global_pool,
        'csv_path': cmd.csv_path,
    }
    with open(os.path.join(cmd.output_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f'\n{"=" * 60}\nExtraction complete!\n{"=" * 60}')


if __name__ == '__main__':
    main()
