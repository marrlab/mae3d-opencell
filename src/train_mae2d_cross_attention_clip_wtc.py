"""
Training script for MAE2D with Channel Cross-Attention and CLIP ESM2 on WTC-11.

Fine-tunes from a 2D FFT k-fold checkpoint with InfoNCE contrastive loss
between the image embedding and per-protein ESM2 sequence embedding.
Operates on max-projected 2D images: (B, 2, 224, 224).

Usage:
    # K-fold CLIP fine-tuning (fold k)
    python src/train_mae2d_cross_attention_clip_wtc.py \
        --config configs/wtc/wtc_2d_cross_attention_clip_kfold.yaml \
        --resume /path/to/fft_kfold5/fold2/ckpts/checkpoint_0009.pth.tar \
        --csv_path /path/to/kfold5/fold2 \
        --output_dir /path/to/clip_kfold5/fold2 \
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
from lib.trainers.mae2d_cross_attention_clip_wtc_trainer import (
    MAE2DChannelCrossAttentionCLIPWTCTrainer,
)


def setup_distributed():
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank       = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)
        is_distributed = True
        if rank == 0:
            print(f'Distributed training: rank {rank}/{world_size}, local_rank {local_rank}')
    else:
        rank, world_size, local_rank = 0, 1, 0
        is_distributed = False
        print('Single GPU training')
    return rank, world_size, local_rank, is_distributed


def main():
    parser = argparse.ArgumentParser(description='MAE2D Cross-Attention + CLIP ESM2 Training on WTC-11')
    parser.add_argument('--config',               type=str,   default=None)
    parser.add_argument('--resume',               type=str,   default=None,
                        help='Checkpoint to resume from (overrides config)')
    parser.add_argument('--csv_path',             type=str,   default=None,
                        help='Directory with train.csv/val.csv (overrides config; use for k-fold)')
    parser.add_argument('--output_dir',           type=str,   default=None,
                        help='Output directory (overrides config; use for k-fold)')
    parser.add_argument('--fold',                 type=int,   default=None,
                        help='Fold index (used in WandB run name)')
    parser.add_argument('--esm2_embedding_path',  type=str,   default=None,
                        help='Directory with train.npy/val.npy ESM2 embeddings')
    parser.add_argument('--clip_weight',          type=float, default=None)
    parser.add_argument('--clip_embed_dim',       type=int,   default=None)
    parser.add_argument('--epochs',               type=int,   default=None)
    parser.add_argument('--lr',                   type=float, default=None)
    cmd_args = parser.parse_args()

    rank, world_size, local_rank, is_distributed = setup_distributed()

    code_base_path = '/path/to/repository/'
    config_path = cmd_args.config if cmd_args.config is not None else \
        os.path.join(code_base_path, 'configs/wtc/wtc_2d_cross_attention_clip_kfold.yaml')

    if rank == 0:
        print(f'Loading config from: {config_path}')

    args = get_conf(config_path)

    if cmd_args.resume is not None:
        args.resume = cmd_args.resume
    if cmd_args.csv_path is not None:
        args.csv_path = cmd_args.csv_path
    if cmd_args.output_dir is not None:
        args.output_dir = cmd_args.output_dir
        args.ckpt_dir   = os.path.join(cmd_args.output_dir, 'ckpts')
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

    args.rank       = rank
    args.world_size = world_size
    args.gpu        = local_rank

    set_seed(args.seed + rank)

    if rank == 0:
        print(f'\n{"="*60}')
        print(f'MAE2D Cross-Attention + CLIP ESM2 — WTC-11')
        print(f'Input: (B, 2, 224, 224)  [max-projection]')
        print(f'Mask ratio: {args.mask_ratio}')
        print(f'CLIP embed dim: {getattr(args, "clip_embed_dim", 768)}')
        print(f'CLIP weight: {getattr(args, "clip_weight", 1.0)}')
        if cmd_args.fold is not None:
            print(f'K-Fold: fold {cmd_args.fold}')
        print(f'{"="*60}\n')

    trainer = MAE2DChannelCrossAttentionCLIPWTCTrainer(args)
    trainer.build_model()
    trainer.build_optimizer()
    trainer.build_dataloader()

    if args.resume is not None:
        trainer.resume()

    if rank == 0:
        wandb.init(
            project=f"{args.proj_name}_{args.dataset}",
            name=args.run_name,
            config=OmegaConf.to_container(args, resolve=True),
        )
        print(f'\nWandB: {wandb.run.name}')
        print(f'{"="*60}\n')

    trainer.run()

    if rank == 0:
        wandb.finish()
        print('\nTraining completed successfully!')

    if is_distributed:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
