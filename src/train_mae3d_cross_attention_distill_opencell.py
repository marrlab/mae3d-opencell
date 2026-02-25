"""
Training script for MAE3D with Channel Cross-Attention and Distillation.

This script fine-tunes a pre-trained MAE3D with cross-attention model
using knowledge distillation from SubCell embeddings.

Key features:
- Loads pre-trained MAE3D cross-attention weights
- Adds distillation from SubCell teacher embeddings
- Supports MSE or Cosine similarity distillation loss
- Distillation loss ramp-up during first epoch

Usage:
    # Single GPU
    python src/train_mae3d_cross_attention_distill_opencell.py

    # Multi-GPU with torchrun
    torchrun --nproc_per_node=4 src/train_mae3d_cross_attention_distill_opencell.py

    # With custom config
    python src/train_mae3d_cross_attention_distill_opencell.py --config configs/opencell/opencell_3d_cross_attention_distill.yaml
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
from lib.trainers import MAE3DChannelCrossAttentionDistillTrainer


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
    parser = argparse.ArgumentParser(description='MAE3D Cross-Attention Distillation Training')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config file (optional)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from (overrides config)')
    parser.add_argument('--distill_loss_type', type=str, default=None,
                        choices=['mse', 'cosine'],
                        help='Distillation loss type (overrides config)')
    parser.add_argument('--distill_weight', type=float, default=None,
                        help='Distillation loss weight (overrides config)')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of epochs (overrides config)')
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate (overrides config)')
    cmd_args = parser.parse_args()

    # Setup distributed training
    rank, world_size, local_rank, is_distributed = setup_distributed()

    # Load config
    code_base_path = '/lustre/groups/aih/amirhossein.kardoost/codes/single-cell-foundation-model/single-cell-foundation-model/'

    if cmd_args.config is not None:
        config_path = cmd_args.config
    else:
        config_path = os.path.join(code_base_path, 'configs/opencell/opencell_3d_cross_attention_distill.yaml')

    if rank == 0:
        print(f'Loading config from: {config_path}')

    args = get_conf(config_path)

    # Override config with command line arguments
    if cmd_args.resume is not None:
        args.resume = cmd_args.resume
    if cmd_args.distill_loss_type is not None:
        args.distill_loss_type = cmd_args.distill_loss_type
    if cmd_args.distill_weight is not None:
        args.distill_weight = cmd_args.distill_weight
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
        print(f'Model: MAE3D with Channel Cross-Attention + Distillation')
        print(f'Channels: {args.in_chans} (processed as separate token streams)')
        print(f'Mask ratio: {args.mask_ratio}')
        print(f'Distillation loss: {args.distill_loss_type}')
        print(f'Distillation weight: {args.distill_weight}')
        print(f'Ramp-up epochs: {getattr(args, "distill_rampup_epochs", 1)}')
        print(f'{"="*60}\n')

    trainer = MAE3DChannelCrossAttentionDistillTrainer(args)

    # Build model
    trainer.build_model()

    # Build optimizer
    trainer.build_optimizer()

    # Build dataloader (with teacher embeddings)
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
        print('\nDistillation fine-tuning completed successfully!')

    if is_distributed:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
