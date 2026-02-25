#!/usr/bin/env python3
"""
Training script for FOV (Field-of-View) images using MAE.

This script is a wrapper around train_with_trainer.py specifically for FOV datasets.
It handles FOV-specific dataset loading and configuration.

Usage:
    # Train MAE 3D on FOV images
    python src/train_fov.py --config configs/opencell/opencell_3d_fov.yaml

    # Train MAE 2D on FOV images (max-projected)
    python src/train_fov.py --config configs/opencell/opencell_2d_fov.yaml --use_2d

    # Mixed training (single cells + FOV)
    python src/train_fov.py \
        --config configs/opencell/opencell_3d_fov.yaml \
        --mixed_training \
        --single_cell_csv /path/to/train.csv \
        --fov_ratio 0.3
"""

import os
import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from utils.utils import get_conf
from data.opencell.fov_dataset import OpenCellFOVDataset, OpenCellMixedDataset
from data.opencell.dataset import OpenCellDataset
from data.opencell.transforms import (
    get_opencell_fov_train_transforms,
    get_opencell_fov_val_transforms,
    get_opencell_fov_2d_train_transforms,
    get_opencell_fov_2d_val_transforms,
    get_opencell_train_transforms,
    get_opencell_val_transforms,
)

# Import trainers
from lib.trainers import MAE3DTrainer, MAE2DTrainer


def build_fov_dataset(args, split='train', use_2d=False):
    """Build FOV dataset with appropriate transforms."""

    # Get transform based on split and dimensionality
    if split == 'train':
        if use_2d:
            transform = get_opencell_fov_2d_train_transforms(
                crop_size=None,  # No cropping for full FOV
                channel_wise_norm=getattr(args, 'channel_wise_norm', True),
                intensity_augmentation=getattr(args, 'intensity_augmentation', True),
                flip_prob=getattr(args, 'RandFlipd_prob', 0.2),
                rotate_prob=getattr(args, 'RandRotate90d_prob', 0.2),
                scale_intensity_prob=getattr(args, 'RandScaleIntensityd_prob', 0.1),
                shift_intensity_prob=getattr(args, 'RandShiftIntensityd_prob', 0.1),
            )
        else:
            transform = get_opencell_fov_train_transforms(
                crop_size=None,  # No cropping for full FOV
                channel_wise_norm=getattr(args, 'channel_wise_norm', True),
                intensity_augmentation=getattr(args, 'intensity_augmentation', True),
                flip_prob=getattr(args, 'RandFlipd_prob', 0.2),
                rotate_prob=getattr(args, 'RandRotate90d_prob', 0.2),
                scale_intensity_prob=getattr(args, 'RandScaleIntensityd_prob', 0.1),
                shift_intensity_prob=getattr(args, 'RandShiftIntensityd_prob', 0.1),
            )
    else:
        # Validation
        if use_2d:
            transform = get_opencell_fov_2d_val_transforms(
                crop_size=None,
                channel_wise_norm=getattr(args, 'channel_wise_norm', True),
            )
        else:
            transform = get_opencell_fov_val_transforms(
                crop_size=None,
                channel_wise_norm=getattr(args, 'channel_wise_norm', True),
            )

    # Create FOV dataset
    dataset = OpenCellFOVDataset(
        fov_dir=args.data_path,
        split=split,
        transform=transform,
        cache_rate=getattr(args, 'cache_rate', 0.0),
        use_max_projection=use_2d,
    )

    return dataset


def build_mixed_dataset(args, single_cell_csv, fov_ratio=0.3):
    """Build mixed dataset (single cells + FOV)."""

    # Build single cell dataset
    single_cell_transform = get_opencell_train_transforms(
        channel_wise_norm=getattr(args, 'channel_wise_norm', True),
        intensity_augmentation=getattr(args, 'intensity_augmentation', True),
    )

    single_cell_ds = OpenCellDataset(
        csv_path=single_cell_csv,
        split='train',
        transform=single_cell_transform,
        cache_rate=0.0,
    )

    # Build FOV dataset
    fov_ds = build_fov_dataset(args, split='train', use_2d=False)

    # Create mixed dataset
    mixed_ds = OpenCellMixedDataset(
        single_cell_dataset=single_cell_ds,
        fov_dataset=fov_ds,
        fov_ratio=fov_ratio,
    )

    return mixed_ds


def main():
    parser = argparse.ArgumentParser(description='Train MAE on FOV images')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--use_2d', action='store_true', help='Use 2D (max-projected) images')
    parser.add_argument('--mixed_training', action='store_true',
                       help='Train on mix of single cells and FOV')
    parser.add_argument('--single_cell_csv', type=str, default=None,
                       help='Path to single cell CSV for mixed training')
    parser.add_argument('--fov_ratio', type=float, default=0.3,
                       help='Ratio of FOV samples in mixed training (0.0-1.0)')

    # Allow config overrides
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--epochs', type=int, default=None)

    args_cmd = parser.parse_args()

    # Load config
    args = get_conf(args_cmd.config)

    # Override config with command line args
    if args_cmd.lr is not None:
        args.lr = args_cmd.lr
    if args_cmd.batch_size is not None:
        args.batch_size = args_cmd.batch_size
    if args_cmd.epochs is not None:
        args.epochs = args_cmd.epochs

    print("="*80)
    print("FOV Training Configuration")
    print("="*80)
    print(f"Config: {args_cmd.config}")
    print(f"Mode: {'2D (max-projection)' if args_cmd.use_2d else '3D (volumetric)'}")
    print(f"Mixed training: {args_cmd.mixed_training}")
    if args_cmd.mixed_training:
        print(f"FOV ratio: {args_cmd.fov_ratio}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print("="*80)

    # Select trainer
    if args_cmd.use_2d or getattr(args, 'use_2d', False):
        trainer_class = MAE2DTrainer
    else:
        trainer_class = MAE3DTrainer

    # Initialize trainer with custom dataset builder
    print("Initializing trainer...")
    trainer = trainer_class(args)

    # Build custom dataloaders for FOV
    print("Building FOV dataloaders...")

    if args_cmd.mixed_training:
        if args_cmd.single_cell_csv is None:
            raise ValueError("--single_cell_csv required for mixed training")

        train_dataset = build_mixed_dataset(
            args,
            single_cell_csv=args_cmd.single_cell_csv,
            fov_ratio=args_cmd.fov_ratio
        )
    else:
        train_dataset = build_fov_dataset(
            args,
            split='train',
            use_2d=args_cmd.use_2d
        )

    # Build dataloader
    from torch.utils.data import DataLoader

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=getattr(args, 'workers', 8),
        pin_memory=True,
    )

    # Replace trainer's dataloader
    trainer.train_loader = train_loader

    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Iterations per epoch: {len(train_loader)}")

    # Start training
    print("\nStarting training...")
    trainer.run()

    print("\nTraining complete!")


if __name__ == '__main__':
    main()
