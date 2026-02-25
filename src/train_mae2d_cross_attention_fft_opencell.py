"""
Training script for MAE2D with Channel Cross-Attention and FFT Loss on OpenCell.

Trains on max-projected (2D) images with:
- Position-wise cross-attention between nucleus and protein channels
- FFT loss on full per-channel reconstructions

Usage:
    # Single GPU
    python src/train_mae2d_cross_attention_fft_opencell.py

    # With custom config
    python src/train_mae2d_cross_attention_fft_opencell.py \\
        --config configs/opencell/opencell_2d_cross_attention_fft_kfold.yaml

    # K-fold (fold k)
    python src/train_mae2d_cross_attention_fft_opencell.py \\
        --csv_path /path/to/kfold/fold2 \\
        --output_dir /path/to/output/fold2 \\
        --fold 2

    # Multi-GPU
    torchrun --nproc_per_node=4 src/train_mae2d_cross_attention_fft_opencell.py
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
from lib.trainers import MAE2DChannelCrossAttentionFFTTrainer


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
    parser = argparse.ArgumentParser(description='MAE2D Cross-Attention + FFT Training')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config file (optional)')
    parser.add_argument('--csv_path', type=str, default=None,
                        help='Directory with train.csv/val.csv (overrides config; use for k-fold)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory (overrides config; use for k-fold)')
    parser.add_argument('--fold', type=int, default=None,
                        help='Fold index for k-fold cross-validation (used in run name)')
    cmd_args = parser.parse_args()

    rank, world_size, local_rank, is_distributed = setup_distributed()

    code_base_path = '/lustre/groups/aih/amirhossein.kardoost/codes/single-cell-foundation-model/single-cell-foundation-model/'
    if cmd_args.config is not None:
        config_path = cmd_args.config
    else:
        config_path = os.path.join(code_base_path, 'configs/opencell/opencell_2d_cross_attention_fft_kfold.yaml')

    if rank == 0:
        print(f'Loading config from: {config_path}')

    args = get_conf(config_path)

    if cmd_args.csv_path is not None:
        args.csv_path = cmd_args.csv_path
    if cmd_args.output_dir is not None:
        args.output_dir = cmd_args.output_dir
        args.ckpt_dir = os.path.join(cmd_args.output_dir, 'ckpts')
    if cmd_args.fold is not None:
        args.run_name = f'{args.run_name}_fold{cmd_args.fold}'

    args.rank = rank
    args.world_size = world_size
    args.gpu = local_rank

    set_seed(args.seed + rank)

    if rank == 0:
        print(f'\n{"="*60}')
        print(f'Creating {args.trainer_name}...')
        print(f'Model: MAE2D with Channel Cross-Attention + FFT Loss')
        print(f'Input: max-projection 2D  {tuple(args.input_size)}')
        print(f'Channels: {args.in_chans}')
        print(f'Mask ratio: {args.mask_ratio}')
        print(f'FFT loss weight: {args.fft_loss_weight}')
        if cmd_args.fold is not None:
            print(f'K-Fold: fold {cmd_args.fold}')
        print(f'{"="*60}\n')

    trainer = MAE2DChannelCrossAttentionFFTTrainer(args)
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
