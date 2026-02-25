#!/usr/bin/env python3
"""
Training script for PPI (Protein-Protein Interaction) prediction via metric learning.

This script trains a Siamese-style network that learns protein embeddings
where interacting proteins have high cosine similarity.

Architecture:
    z = MLP(encoder(image))
    z_normalized = z / ||z||
    similarity = z_A · z_B

Usage:
    # Train 3D model
    python src/train_ppi.py configs/opencell/opencell_ppi_3d.yaml

    # Train 2D model
    python src/train_ppi.py configs/opencell/opencell_ppi_2d.yaml

    # Multi-GPU training
    torchrun --nproc_per_node=4 src/train_ppi.py configs/opencell/opencell_ppi_3d.yaml
"""

import os
import sys
import argparse
import random
import numpy as np
import torch
import torch.distributed as dist
import wandb
from pathlib import Path
from omegaconf import OmegaConf

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from utils.utils import get_conf
from lib.trainers import PPITrainer


def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_distributed():
    """Setup distributed training from environment variables."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        gpu = int(os.environ['LOCAL_RANK'])
    else:
        rank = 0
        world_size = 1
        gpu = 0

    if world_size > 1:
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=world_size,
            rank=rank
        )
        torch.cuda.set_device(gpu)

    return rank, world_size, gpu


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Train PPI Metric Learning Model')
    parser.add_argument('config', type=str, help='Path to config file')
    parser.add_argument('--mae_embedding_path', type=str, default=None,
                        help='Path to precomputed MAE embeddings directory (for fast training)')
    parser.add_argument('--mae_embedding_csv_path', type=str, default=None,
                        help='Directory of CSVs used during embedding extraction (e.g. dataset1/). '
                             'When set, builds a combined lookup from all splits so kfold '
                             'training CSVs can reference cells from any source split.')
    parser.add_argument('--opts', nargs=argparse.REMAINDER, default=None,
                       help='Override config options (e.g., --opts lr=1e-3 batch_size=8)')
    args_cmd = parser.parse_args()

    # Load config
    args = get_conf(args_cmd.config)

    # Override with command line args
    if args_cmd.mae_embedding_path is not None:
        args.mae_embedding_path = args_cmd.mae_embedding_path
    if args_cmd.mae_embedding_csv_path is not None:
        args.mae_embedding_csv_path = args_cmd.mae_embedding_csv_path

    # Handle config overrides
    if args_cmd.opts is not None:
        for opt in args_cmd.opts:
            if '=' in opt:
                key, value = opt.split('=')
                try:
                    value = eval(value)
                except:
                    pass
                setattr(args, key, value)

    # Recompute ckpt_dir if output_dir was overridden
    if args_cmd.opts is not None and any(opt.startswith('output_dir=') for opt in args_cmd.opts):
        args.ckpt_dir = str(Path(args.output_dir) / "ckpts")

    # Setup distributed training
    rank, world_size, gpu = setup_distributed()
    args.rank = rank
    args.world_size = world_size
    args.gpu = gpu
    args.distributed = world_size > 1
    args.ngpus_per_node = torch.cuda.device_count()

    # Set random seed
    set_seed(args.seed + rank)

    # Create output directories
    if rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(args.ckpt_dir, exist_ok=True)

    # Convert OmegaConf to plain dict for wandb and printing
    config_dict = OmegaConf.to_container(args, resolve=True) if rank == 0 else None

    # Initialize wandb
    if rank == 0:
        wandb.init(
            project=args.proj_name,
            name=args.run_name,
            config=config_dict,
            id=args.wandb_id if args.wandb_id != 'None' else None,
            resume='allow' if args.resume else False
        )

    # Print configuration
    if rank == 0:
        print("=" * 80)
        print("PPI Metric Learning Configuration:")
        print("=" * 80)
        for key, value in sorted(config_dict.items()):
            print(f"  {key}: {value}")
        print("=" * 80)

    # Create trainer
    trainer = PPITrainer(args)

    # Build model, optimizer, and dataloader
    trainer.build_model()
    trainer.build_optimizer()
    trainer.build_dataloader()

    # Resume from checkpoint if specified
    if args.resume is not None and os.path.exists(args.resume):
        trainer.resume()

    # Print model info
    if rank == 0:
        print("\n" + "=" * 80)
        print("Model Information:")
        print("=" * 80)
        total_params = sum(p.numel() for p in trainer.model.parameters())
        trainable_params = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Projection output dim: {trainer.model.proj_output_dim}")
        print("=" * 80 + "\n")

    # Start training
    try:
        trainer.run()
    except KeyboardInterrupt:
        if rank == 0:
            print("\nTraining interrupted by user")
    except Exception as e:
        if rank == 0:
            print(f"\nTraining failed with error: {e}")
        raise
    finally:
        if rank == 0:
            wandb.finish()
        if args.distributed:
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
