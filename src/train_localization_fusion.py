"""
Training script for Protein Localization with MAE3D + SubCell Fusion.

This script trains a classifier that combines MAE3D cross-attention encoder
features with precomputed SubCell embeddings for protein localization.

Architecture:
- MAE3D cross-attention encoder → global pool + concat → [768]
- SubCell embedding (precomputed) → [1536]
- Concatenate → [2304] → Linear → [17 classes]

Usage:
    # Single GPU
    python src/train_localization_fusion.py

    # Multi-GPU
    torchrun --nproc_per_node=4 src/train_localization_fusion.py

    # With custom config
    python src/train_localization_fusion.py --config configs/opencell/opencell_localization_3d_cross_attention_subcell_fusion.yaml
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
from lib.trainers import LocalizationFusionTrainer


def setup_distributed():
    """Setup distributed training environment."""
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
        print('Single GPU training')

    return rank, world_size, local_rank, is_distributed


def main():
    parser = argparse.ArgumentParser(description='MAE3D + SubCell Fusion Localization')
    parser.add_argument('--config', type=str, default=None, help='Path to config file')
    parser.add_argument('--pretrain', type=str, default=None, help='Path to pretrained MAE3D checkpoint')
    parser.add_argument('--freeze_encoder', action='store_true', help='Freeze MAE3D encoder (linear probing)')
    parser.add_argument('--epochs', type=int, default=None, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=None, help='Learning rate')
    parser.add_argument('--mae_embedding_path', type=str, default=None,
                        help='Path to precomputed MAE embeddings (for fast training)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Override output directory from config')
    cmd_args = parser.parse_args()

    # Setup distributed
    rank, world_size, local_rank, is_distributed = setup_distributed()

    # Load config
    code_base_path = '/path/to/repository/'

    if cmd_args.config is not None:
        config_path = cmd_args.config
    else:
        config_path = os.path.join(code_base_path, 'configs/opencell/opencell_localization_3d_cross_attention_subcell_fusion.yaml')

    if rank == 0:
        print(f'Loading config from: {config_path}')

    args = get_conf(config_path)

    # Override with command line args
    if cmd_args.pretrain is not None:
        args.pretrain = cmd_args.pretrain
    if cmd_args.freeze_encoder:
        args.freeze_encoder = True
    if cmd_args.epochs is not None:
        args.epochs = cmd_args.epochs
    if cmd_args.lr is not None:
        args.lr = cmd_args.lr
    if cmd_args.mae_embedding_path is not None:
        args.mae_embedding_path = cmd_args.mae_embedding_path
    if cmd_args.output_dir is not None:
        args.output_dir = cmd_args.output_dir
        args.ckpt_dir = os.path.join(cmd_args.output_dir, 'ckpts')

    # Add distributed info
    args.rank = rank
    args.world_size = world_size
    args.gpu = local_rank

    set_seed(args.seed + rank)

    # Create trainer
    if rank == 0:
        print(f'\n{"="*60}')
        print(f'Creating {args.trainer_name}...')
        print(f'Model: MAE3D Cross-Attention + SubCell Fusion')
        print(f'Fusion: {args.fusion_type}')
        print(f'MAE3D features: 768-dim (concat pool)')
        print(f'SubCell features: {args.subcell_embed_dim}-dim')
        print(f'{"="*60}\n')

    trainer = LocalizationFusionTrainer(args)

    # Build model
    trainer.build_model()

    # Build optimizer
    trainer.build_optimizer()

    # Build dataloader
    trainer.build_dataloader()

    # Resume if specified
    if args.resume is not None:
        trainer.resume()

    # Setup wandb
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
        print('\nTraining completed!')

    if is_distributed:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
