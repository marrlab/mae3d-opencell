"""
Training script for MAE2D on OpenCell dataset with individual Z-slices.

This script trains a 2D Masked Autoencoder on individual slices from OpenCell volumes,
instead of using max-projection. This increases the number of training samples from
1 per volume to (slice_end - slice_start + 1) per volume.

Usage:
    # Single GPU
    python src/train_mae2d_slices_opencell.py

    # Multi-GPU with torchrun
    torchrun --nproc_per_node=4 src/train_mae2d_slices_opencell.py
"""

import os
import sys
import torch
import torch.distributed as dist
import wandb
from omegaconf import OmegaConf

# Add src to path
sys.path.insert(0, os.path.dirname(__file__))

from utils.utils import set_seed, get_conf
from lib.trainers import MAE2DTrainer


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
    # Setup distributed training
    rank, world_size, local_rank, is_distributed = setup_distributed()

    # Load config for slice-based training
    code_base_path = '/lustre/groups/aih/amirhossein.kardoost/codes/single-cell-foundation-model/single-cell-foundation-model/'
    config_path = os.path.join(code_base_path, 'configs/opencell/mask_ratio_experiments/opencell_2d_slices_mask0.75.yaml')

    if rank == 0:
        print(f'Loading config from: {config_path}')

    args = get_conf(config_path)

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
        print(f'Mode: Slice-based (Z-slices {args.slice_start} to {args.slice_end})')
        print(f'Mask ratio: {args.mask_ratio}')
        print(f'{"="*60}\n')

    trainer = MAE2DTrainer(args)

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
