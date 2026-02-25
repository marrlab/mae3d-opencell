"""
Train WTC-11 protein localization classifier (linear probe on MAE embeddings).

Usage
-----
    python src/train_localization_wtc.py configs/wtc/wtc_localization_emb_3d_fft_kfold.yaml \
        --mae_embedding_path /path/to/fold0/mae3d_embeddings \
        --opts csv_path=/path/to/kfold5/fold0 output_dir=/path/to/output/fold0
"""

import os
import sys
import argparse
import random

import numpy as np
import torch
import torch.distributed as dist
import wandb
from omegaconf import OmegaConf
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from utils.utils import get_conf
from lib.trainers.localization_wtc_trainer import LocalizationWTCTrainer


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_distributed():
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        gpu = int(os.environ['LOCAL_RANK'])
    else:
        rank, world_size, gpu = 0, 1, 0

    if world_size > 1:
        dist.init_process_group(backend='nccl', init_method='env://',
                                world_size=world_size, rank=rank)
        torch.cuda.set_device(gpu)

    return rank, world_size, gpu


def main():
    parser = argparse.ArgumentParser(description='Train WTC-11 Localization Classifier')
    parser.add_argument('config', type=str)
    parser.add_argument('--mae_embedding_path', type=str, default=None)
    parser.add_argument('--mae_embedding_csv_path', type=str, default=None)
    parser.add_argument('--opts', nargs=argparse.REMAINDER, default=None)
    args_cmd = parser.parse_args()

    args = get_conf(args_cmd.config)

    if args_cmd.mae_embedding_path is not None:
        args.mae_embedding_path = args_cmd.mae_embedding_path
    if args_cmd.mae_embedding_csv_path is not None:
        args.mae_embedding_csv_path = args_cmd.mae_embedding_csv_path

    if args_cmd.opts:
        for opt in args_cmd.opts:
            if '=' in opt:
                key, value = opt.split('=', 1)
                try:
                    value = eval(value)
                except Exception:
                    pass
                setattr(args, key, value)

    if args_cmd.opts and any(o.startswith('output_dir=') for o in args_cmd.opts):
        args.ckpt_dir = str(Path(args.output_dir) / 'ckpts')

    rank, world_size, gpu = setup_distributed()
    args.rank = rank
    args.world_size = world_size
    args.gpu = gpu
    args.distributed = world_size > 1
    args.ngpus_per_node = torch.cuda.device_count()

    set_seed(args.seed + rank)

    if rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(args.ckpt_dir, exist_ok=True)

    config_dict = OmegaConf.to_container(args, resolve=True) if rank == 0 else None

    if rank == 0:
        wandb.init(
            project=args.proj_name,
            name=args.run_name,
            config=config_dict,
            id=args.wandb_id if args.wandb_id != 'None' else None,
            resume='allow' if args.resume else False,
        )

    trainer = LocalizationWTCTrainer(args)
    trainer.build_model()
    trainer.build_optimizer()
    trainer.build_dataloader()

    if args.resume is not None and os.path.exists(args.resume):
        trainer.resume()

    try:
        trainer.run()
    except KeyboardInterrupt:
        if rank == 0:
            print('\nInterrupted')
    finally:
        if rank == 0:
            wandb.finish()
        if args.distributed:
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
