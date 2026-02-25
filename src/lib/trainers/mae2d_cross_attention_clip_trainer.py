"""
Trainer for MAE2D with Channel Cross-Attention and CLIP-based ESM2 Integration.

Adapts MAE3DChannelCrossAttentionCLIPTrainer for max-projected 2D images:
- 2D model (MAE2DChannelCrossAttentionCLIP)
- Max-projection dataloader (use_max_projection=True)
- 2D reconstruction visualisation (no z-slices)
"""

import os
import torch

from .mae3d_cross_attention_clip_trainer import MAE3DChannelCrossAttentionCLIPTrainer
from lib.models.mae2d_cross_attention_clip import MAE2DChannelCrossAttentionCLIP
from data.opencell.dataset import OpenCellDataset
from data.opencell.transforms import get_opencell_2d_train_transforms, get_opencell_2d_val_transforms
from torch.utils.data import DistributedSampler


class MAE2DChannelCrossAttentionCLIPTrainer(MAE3DChannelCrossAttentionCLIPTrainer):
    """
    Trainer for MAE2D CLIP — inherits all CLIP loss logic from the 3D trainer,
    overrides build_model (2D model) and build_dataloader (max-projection).
    """

    def __init__(self, args):
        super().__init__(args)
        self.model_name = 'MAE2DChannelCrossAttentionCLIP'

    def build_model(self):
        """Build MAE2D CLIP model."""
        if self.model is not None:
            raise ValueError("Model has been created. Do not create twice")

        args = self.args
        print(f"=> creating model {self.model_name}")
        print(f"   Channels: {args.in_chans}")
        print(f"   Encoder depth: {args.encoder_depth}")
        print(f"   Decoder depth: {args.decoder_depth}")
        print(f"   Encoder embed dim: {args.encoder_embed_dim}")
        print(f"   Decoder embed dim: {args.decoder_embed_dim}")
        print(f"   ESM2 embed dim: {getattr(args, 'esm2_embed_dim', 1280)}")
        print(f"   CLIP embed dim: {getattr(args, 'clip_embed_dim', 768)}")

        self.model = MAE2DChannelCrossAttentionCLIP(args=args)

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"   Total parameters: {total_params:,}")
        print(f"   Trainable parameters: {trainable_params:,}")

        self.wrap_model()

    def build_dataloader(self):
        """Build max-projected 2D OpenCell dataloaders with ESM2 embeddings."""
        if self.train_loader is not None:
            raise ValueError("Dataloader has been created. Do not create twice.")

        print("=> creating dataloaders")
        args = self.args

        esm2_embedding_path = getattr(args, 'esm2_embedding_path', None)
        train_esm2_path = os.path.join(esm2_embedding_path, "train.npy") if esm2_embedding_path else None
        val_esm2_path = os.path.join(esm2_embedding_path, "val.npy") if esm2_embedding_path else None

        train_csv_path = os.path.join(args.csv_path, "train.csv")
        train_transform = get_opencell_2d_train_transforms(
            flip_prob=args.RandFlipd_prob,
            rotate_prob=args.RandRotate90d_prob
        )

        train_dataset = OpenCellDataset(
            csv_path=train_csv_path,
            split='train',
            transform=train_transform,
            cache_rate=args.cache_rate,
            num_workers=args.workers,
            use_max_projection=True,
            esm2_embedding_path=train_esm2_path
        )

        if self.rank == 0 and train_dataset.esm2_embeddings is not None:
            print(f"  ESM2 embedding dimension: {train_dataset.esm2_embeddings.shape[1]}")

        train_workers = 0 if self.is_distributed else args.workers

        if self.is_distributed:
            sampler = DistributedSampler(
                train_dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=True
            )
            self.train_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=args.batch_size,
                sampler=sampler,
                num_workers=train_workers,
                pin_memory=True
            )
        else:
            self.train_loader = train_dataset.get_dataloader(
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=train_workers
            )

        print(f"   Dataset: OpenCell (max-projected 2D)")
        print(f"   Train samples: {len(train_dataset)}")
        print(f"   Batch size: {args.batch_size}")
        print(f"   Workers: {train_workers}")
        print(f"   Iterations per epoch: {len(self.train_loader)}")

        val_csv_path = os.path.join(args.csv_path, "val.csv")
        if os.path.exists(val_csv_path):
            val_transform = get_opencell_2d_val_transforms()

            val_dataset = OpenCellDataset(
                csv_path=val_csv_path,
                split='val',
                transform=val_transform,
                cache_rate=0.0,
                num_workers=args.workers,
                use_max_projection=True,
                esm2_embedding_path=val_esm2_path
            )

            if self.is_distributed:
                val_sampler = DistributedSampler(
                    val_dataset,
                    num_replicas=self.world_size,
                    rank=self.rank,
                    shuffle=False
                )
                self.val_loader = torch.utils.data.DataLoader(
                    val_dataset,
                    batch_size=args.batch_size,
                    sampler=val_sampler,
                    num_workers=0,
                    pin_memory=True
                )
            else:
                self.val_loader = val_dataset.get_dataloader(
                    batch_size=args.batch_size,
                    shuffle=False,
                    num_workers=0
                )

            print(f"   Val samples: {len(val_dataset)}")
            print(f"   Validation enabled: True\n")
        else:
            print(f"   Validation enabled: False\n")

        if self.rank == 0:
            self._store_fixed_vis_sample()
