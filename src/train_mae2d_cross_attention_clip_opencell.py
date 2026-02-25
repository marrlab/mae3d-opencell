"""
Training script for MAE2D with Channel Cross-Attention and CLIP-based ESM2 Integration.

Trains a 2D Masked Autoencoder on max-projected OpenCell images with:
- Channel-wise tokens with cross-attention (nucleus and protein)
- ESM2 token concatenated in decoder
- InfoNCE (CLIP-style) contrastive loss between image and ESM2 embeddings

Usage:
    # Single GPU
    python src/train_mae2d_cross_attention_clip_opencell.py \
        --config configs/opencell/opencell_2d_cross_attention_clip_kfold.yaml

    # K-fold cross-validation (fold k)
    python src/train_mae2d_cross_attention_clip_opencell.py \
        --config configs/opencell/opencell_2d_cross_attention_clip_kfold.yaml \
        --resume /path/to/fft_kfold/fold2/ckpts/checkpoint_0009.pth.tar \
        --csv_path /path/to/kfold5/fold2 \
        --output_dir /path/to/output/fold2 \
        --esm2_embedding_path /path/to/esm2_kfold5/fold2 \
        --fold 2
"""

import os
import sys
import argparse
import torch
import torch.distributed as dist
import wandb
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(__file__))

from utils.utils import set_seed, get_conf
from lib.trainers import MAE2DChannelCrossAttentionCLIPTrainer


def setup_distributed():
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)
        is_distributed = True
        if rank == 0:
            print(f'Distributed training: rank {rank}/{world_size}, local_rank {local_rank}')
    else:
        rank = 0
        world_size = 1
        local_rank = 0
        is_distributed = False
        if rank == 0:
            print('Single GPU training')

    return rank, world_size, local_rank, is_distributed


def main():
    parser = argparse.ArgumentParser(description='MAE2D Cross-Attention CLIP Training')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config file (optional)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from (overrides config)')
    parser.add_argument('--csv_path', type=str, default=None,
                        help='Directory with train.csv/val.csv (overrides config; use for k-fold)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory (overrides config; use for k-fold)')
    parser.add_argument('--fold', type=int, default=None,
                        help='Fold index for k-fold cross-validation (used in run name)')
    parser.add_argument('--esm2_embedding_path', type=str, default=None,
                        help='Directory with train.npy/val.npy ESM2 embeddings (overrides config; use for k-fold)')
    parser.add_argument('--clip_weight', type=float, default=None,
                        help='CLIP loss weight (overrides config)')
    parser.add_argument('--clip_embed_dim', type=int, default=None,
                        help='CLIP embedding dimension (overrides config)')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of epochs (overrides config)')
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate (overrides config)')
    cmd_args = parser.parse_args()

    rank, world_size, local_rank, is_distributed = setup_distributed()

    code_base_path = '/lustre/groups/aih/amirhossein.kardoost/codes/single-cell-foundation-model/single-cell-foundation-model/'
    if cmd_args.config is not None:
        config_path = cmd_args.config
    else:
        config_path = os.path.join(code_base_path, 'configs/opencell/opencell_2d_cross_attention_clip_kfold.yaml')

    if rank == 0:
        print(f'Loading config from: {config_path}')

    args = get_conf(config_path)

    # Override config with command line arguments
    if cmd_args.resume is not None:
        args.resume = cmd_args.resume
    if cmd_args.csv_path is not None:
        args.csv_path = cmd_args.csv_path
    if cmd_args.output_dir is not None:
        args.output_dir = cmd_args.output_dir
        args.ckpt_dir = os.path.join(cmd_args.output_dir, 'ckpts')
    if cmd_args.fold is not None:
        args.run_name = f'{args.run_name}_fold{cmd_args.fold}'
    if cmd_args.esm2_embedding_path is not None:
        args.esm2_embedding_path = cmd_args.esm2_embedding_path
    if cmd_args.clip_weight is not None:
        args.clip_weight = cmd_args.clip_weight
    if cmd_args.clip_embed_dim is not None:
        args.clip_embed_dim = cmd_args.clip_embed_dim
    if cmd_args.epochs is not None:
        args.epochs = cmd_args.epochs
    if cmd_args.lr is not None:
        args.lr = cmd_args.lr

    args.rank = rank
    args.world_size = world_size
    args.gpu = local_rank

    set_seed(args.seed + rank)

    if rank == 0:
        print(f'\n{"="*60}')
        print(f'Creating {args.trainer_name}...')
        print(f'Model: MAE2D with Channel Cross-Attention + CLIP ESM2')
        print(f'Input: max-projected 2D  ({args.roi_y}x{args.roi_x})')
        print(f'Mask ratio: {args.mask_ratio}')
        print(f'CLIP embed dim: {getattr(args, "clip_embed_dim", 768)}')
        print(f'CLIP weight: {getattr(args, "clip_weight", 1.0)}')
        print(f'CLIP ramp-up epochs: {getattr(args, "clip_rampup_epochs", 1)}')
        if cmd_args.fold is not None:
            print(f'K-Fold: fold {cmd_args.fold}')
        print(f'{"="*60}\n')

    trainer = MAE2DChannelCrossAttentionCLIPTrainer(args)

    trainer.build_model()
    trainer.build_optimizer()
    trainer.build_dataloader()

    if args.resume is not None:
        trainer.resume()

    if rank == 0:
        wandb.init(
            project=f"{args.proj_name}_{args.dataset}",
            name=args.run_name,
            config=OmegaConf.to_container(args, resolve=True)
        )
        print(f'\nWandB initialized: {wandb.run.name}')
        print(f'{"="*60}\n')

    trainer.run()

    if rank == 0:
        wandb.finish()
        print('\nTraining completed successfully!')

    if is_distributed:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
