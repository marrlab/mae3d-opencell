"""
Training script for MAE3D with Channel Cross-Attention and CLIP-based ESM2 Integration.

This script trains a MAE3D model with:
- Standard encoder (no ESM2 conditioning)
- ESM2 token concatenated in decoder
- InfoNCE (CLIP-style) contrastive loss between image and ESM2 embeddings

Usage:
    # Single GPU
    python src/train_mae3d_cross_attention_clip_opencell.py

    # Multi-GPU with torchrun
    torchrun --nproc_per_node=4 src/train_mae3d_cross_attention_clip_opencell.py

    # With custom config
    python src/train_mae3d_cross_attention_clip_opencell.py --config configs/opencell/opencell_3d_cross_attention_clip.yaml
"""

import os
import sys
import argparse
import torch
import torch.distributed as dist
import wandb
from omegaconf import OmegaConf

# Add src to path
sys.path.insert(0, os.path.dirname(__file__))

from utils.utils import set_seed, get_conf
from lib.trainers import MAE3DChannelCrossAttentionCLIPTrainer


def setup_distributed():
    """
    Setup distributed training environment.
    Returns rank, world_size, local_rank, is_distributed.
    """
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
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='MAE3D Cross-Attention CLIP Training')
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

    # Setup distributed training
    rank, world_size, local_rank, is_distributed = setup_distributed()

    # Load config
    code_base_path = '/path/to/repository/'

    if cmd_args.config is not None:
        config_path = cmd_args.config
    else:
        config_path = os.path.join(code_base_path, 'configs/opencell/opencell_3d_cross_attention_clip.yaml')

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

    # Add distributed training info to args
    args.rank = rank
    args.world_size = world_size
    args.gpu = local_rank

    # Set seed (different seed per rank for data augmentation diversity)
    set_seed(args.seed + rank)

    # Create trainer
    if rank == 0:
        print(f'\n{"="*60}')
        print(f'Creating {args.trainer_name}...')
        print(f'Model: MAE3D with Channel Cross-Attention + CLIP ESM2')
        print(f'Channels: {args.in_chans} (processed as separate token streams)')
        print(f'Mask ratio: {args.mask_ratio}')
        print(f'CLIP embed dim: {getattr(args, "clip_embed_dim", 256)}')
        print(f'CLIP weight: {getattr(args, "clip_weight", 1.0)}')
        print(f'CLIP ramp-up epochs: {getattr(args, "clip_rampup_epochs", 1)}')
        if cmd_args.fold is not None:
            print(f'K-Fold: fold {cmd_args.fold}')
        print(f'{"="*60}\n')

    trainer = MAE3DChannelCrossAttentionCLIPTrainer(args)

    # Build model
    trainer.build_model()

    # Build optimizer
    trainer.build_optimizer()

    # Build dataloader
    trainer.build_dataloader()

    # Resume from checkpoint if specified
    if args.resume is not None:
        trainer.resume()

    # Setup wandb (only on rank 0)
    if rank == 0:
        wandb.init(
            project=f"{args.proj_name}_{args.dataset}",
            name=args.run_name,
            config=OmegaConf.to_container(args, resolve=True)
        )
        print(f'\nWandB initialized: {wandb.run.name}')
        print(f'{"="*60}\n')

    # Run training
    trainer.run()

    # Cleanup
    if rank == 0:
        wandb.finish()
        print('\nTraining completed successfully!')

    if is_distributed:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
