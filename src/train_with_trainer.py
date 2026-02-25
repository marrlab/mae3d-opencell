"""
Training script using the Trainer pattern.

This script provides a cleaner, more modular approach to training compared to
the direct script approach in train_mae3d_opencell.py.

Usage:
    # Single GPU with specific config
    python src/train_with_trainer.py --config configs/opencell/mask_ratio_experiments/opencell_3d_mask0.75.yaml

    # Multi-GPU with torchrun
    torchrun --nproc_per_node=4 src/train_with_trainer.py --config configs/opencell/opencell_3d.yaml

    # Default config (opencell_3d.yaml)
    python src/train_with_trainer.py
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
from utils.logger import setup_logger, redirect_stdout_to_file, save_config_to_file
from lib.trainers import MAE3DTrainer, MAE2DTrainer


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
    parser = argparse.ArgumentParser(description='Train MAE with Trainer pattern')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to config file (relative to project root or absolute)')
    cmd_args = parser.parse_args()

    # Setup distributed training
    rank, world_size, local_rank, is_distributed = setup_distributed()

    # Load config
    code_base_path = '/path/to/repository/'

    # Use provided config or default
    if cmd_args.config is not None:
        # Support both absolute and relative paths
        if os.path.isabs(cmd_args.config):
            config_path = cmd_args.config
        else:
            config_path = os.path.join(code_base_path, cmd_args.config)
    else:
        # Default to opencell_3d.yaml
        config_path = os.path.join(code_base_path, 'configs/opencell/opencell_3d.yaml')

    if rank == 0:
        print(f'Loading config from: {config_path}')

    args = get_conf(config_path)

    # Add distributed training info to args
    args.rank = rank
    args.world_size = world_size
    args.gpu = local_rank

    # Set seed (different seed per rank for data augmentation diversity)
    set_seed(args.seed + rank)

    # Setup logging (redirect stdout to file)
    tee_logger = None
    if rank == 0:
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)

        # Redirect stdout to file (captures all print statements)
        tee_logger = redirect_stdout_to_file(args.output_dir, log_filename='training.log', rank=rank)

        # Save config to output directory
        save_config_to_file(args, args.output_dir, filename='config.yaml', rank=rank)

    # Create trainer
    if rank == 0:
        print(f'\n{"="*60}')
        print(f'Creating {args.trainer_name}...')
        print(f'{"="*60}\n')

    # Select appropriate trainer
    trainer_name = getattr(args, 'trainer_name', 'MAE3DTrainer')
    if trainer_name == 'MAE3DTrainer':
        trainer = MAE3DTrainer(args)
    elif trainer_name == 'MAE2DTrainer':
        trainer = MAE2DTrainer(args)
    else:
        raise ValueError(f"Unknown trainer: {trainer_name}")

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
        # Configure wandb to save locally in output_dir
        wandb_dir = os.path.join(args.output_dir, 'wandb')
        os.makedirs(wandb_dir, exist_ok=True)

        wandb.init(
            project=f"{args.proj_name}_{args.dataset}",
            name=args.run_name,
            config=OmegaConf.to_container(args, resolve=True),
            dir=wandb_dir,  # Save wandb files locally
            save_code=True   # Save code snapshot
        )
        print(f'\nWandB initialized: {wandb.run.name}')
        print(f'WandB directory: {wandb_dir}')
        print(f'{"="*60}\n')

    # Run training
    trainer.run()

    # Cleanup
    if rank == 0:
        wandb.finish()
        print('\nTraining completed successfully!')

        # Close logger
        if tee_logger is not None:
            tee_logger.close()
            # Restore stdout
            sys.stdout = tee_logger.terminal
            sys.stderr = tee_logger.terminal

    if is_distributed:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
