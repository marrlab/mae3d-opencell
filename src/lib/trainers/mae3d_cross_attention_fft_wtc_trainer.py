"""
WTC-11 trainer for MAE3D with Channel Cross-Attention and FFT Loss.

Subclasses MAE3DChannelCrossAttentionFFTTrainer and overrides only
build_dataloader() to use WTCDataset instead of OpenCellDataset.
All training logic, model building, and FFT scheduling are unchanged.
"""

import os
import torch
from torch.utils.data import DistributedSampler

from .mae3d_cross_attention_fft_trainer import MAE3DChannelCrossAttentionFFTTrainer
from data.wtc.dataset import WTCDataset
from data.opencell.transforms import get_opencell_train_transforms, get_opencell_val_transforms


class MAE3DChannelCrossAttentionFFTWTCTrainer(MAE3DChannelCrossAttentionFFTTrainer):
    """
    MAE3D + Channel Cross-Attention + FFT Loss trainer for WTC-11.

    Identical to the OpenCell variant except that dataloaders are built
    using WTCDataset, which:
      - loads OME-TIFF crops (Z, C, Y, X)
      - selects channels [DNA=0, structure=2]
      - centre-crops / zero-pads to (80, 2, 224, 224)

    The same MONAI transform pipeline as OpenCell is applied afterwards
    (transpose + channel-wise normalisation + augmentation), so the
    encoder receives tensors of shape (B, C, Z, Y, X) = (B, 2, 80, 224, 224).
    """

    def build_dataloader(self):
        if self.train_loader is not None:
            raise ValueError("Dataloader has been created. Do not create twice.")

        print("=> creating WTC dataloaders (3D)")
        args = self.args

        train_csv_path = os.path.join(args.csv_path, "train.csv")
        train_transform = get_opencell_train_transforms(
            flip_prob=args.RandFlipd_prob,
            rotate_prob=args.RandRotate90d_prob,
            channel_wise_norm=getattr(args, 'channel_wise_norm', True),
            intensity_augmentation=getattr(args, 'intensity_augmentation', True),
            scale_intensity_prob=getattr(args, 'RandScaleIntensityd_prob', 0.1),
            scale_intensity_factor=getattr(args, 'scale_intensity_factor', 0.1),
            shift_intensity_prob=getattr(args, 'RandShiftIntensityd_prob', 0.1),
            shift_intensity_offset=getattr(args, 'shift_intensity_offset', 0.1),
        )

        train_dataset = WTCDataset(
            csv_path=train_csv_path,
            split='train',
            transform=train_transform,
            cache_rate=args.cache_rate,
            num_workers=args.workers,
            z_slice_start=getattr(args, 'z_slice_start', None),
            z_slice_end=getattr(args, 'z_slice_end', None),
        )

        if args.cache_rate > 0 and self.rank == 0:
            cache_stats = train_dataset.get_cache_stats()
            print(f"  Cache: {cache_stats['cached_images']}/{cache_stats['total_images']} images")

        train_workers = 0 if self.is_distributed else args.workers

        if self.is_distributed:
            sampler = DistributedSampler(
                train_dataset, num_replicas=self.world_size,
                rank=self.rank, shuffle=True
            )
            self.train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=args.batch_size,
                sampler=sampler, num_workers=train_workers, pin_memory=True
            )
        else:
            self.train_loader = train_dataset.get_dataloader(
                batch_size=args.batch_size, shuffle=True, num_workers=train_workers
            )

        print(f"   Dataset: WTC-11 (3D)")
        print(f"   Train samples: {len(train_dataset)}")
        print(f"   Batch size: {args.batch_size}")
        print(f"   Workers: {train_workers} (config: {args.workers})")
        print(f"   Iterations per epoch: {len(self.train_loader)}")

        val_csv_path = os.path.join(args.csv_path, "val.csv")
        if os.path.exists(val_csv_path):
            val_transform = get_opencell_val_transforms(
                channel_wise_norm=getattr(args, 'channel_wise_norm', True),
            )
            val_dataset = WTCDataset(
                csv_path=val_csv_path,
                split='val',
                transform=val_transform,
                cache_rate=0.0,
                num_workers=args.workers,
                z_slice_start=getattr(args, 'z_slice_start', None),
                z_slice_end=getattr(args, 'z_slice_end', None),
            )

            val_workers = 0
            if self.is_distributed:
                val_sampler = DistributedSampler(
                    val_dataset, num_replicas=self.world_size,
                    rank=self.rank, shuffle=False
                )
                self.val_loader = torch.utils.data.DataLoader(
                    val_dataset, batch_size=args.batch_size,
                    sampler=val_sampler, num_workers=val_workers, pin_memory=True
                )
            else:
                self.val_loader = val_dataset.get_dataloader(
                    batch_size=args.batch_size, shuffle=False, num_workers=val_workers
                )

            print(f"   Val samples: {len(val_dataset)}")
            print(f"   Validation enabled: True\n")
        else:
            print(f"   Validation file not found: {val_csv_path}")
            print(f"   Validation enabled: False\n")

        if self.rank == 0:
            self._store_fixed_vis_sample()
