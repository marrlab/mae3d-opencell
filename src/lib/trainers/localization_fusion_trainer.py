"""
Trainer for Protein Localization with MAE3D + SubCell Fusion.

Trains a classifier that combines MAE3D cross-attention features with
precomputed SubCell embeddings for protein localization.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score
from collections import defaultdict
from torch.utils.data import DistributedSampler

from lib.trainers.base_trainer import BaseTrainer
from lib.models import ViT3DCrossAttentionSubCellClassifier
from data.opencell.localization_fusion_dataset import OpenCellLocalizationFusionDataset
from data.opencell.localization_dataset import LOCALIZATION_LABELS
from data.opencell.transforms import get_opencell_train_transforms, get_opencell_val_transforms


class WeightedBCEWithLogitsLoss(nn.Module):
    """Weighted BCE loss for multi-label classification with grade weights."""

    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, logits, targets):
        loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pos_weight = (targets > 0).float() * 2.0 + 1.0
        loss = loss * pos_weight

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class LocalizationFusionTrainer(BaseTrainer):
    """
    Trainer for protein localization using MAE3D + SubCell fusion.

    Supports two modes:
    1. Image mode (default): Load images, run through MAE encoder
    2. Embedding mode: Use precomputed MAE embeddings (much faster when encoder is frozen)

    To use embedding mode, set `mae_embedding_path` in config pointing to a directory
    with train.npy, val.npy, test.npy files containing precomputed MAE embeddings.
    """

    def __init__(self, args):
        super().__init__(args)
        self.model_name = 'ViT3DCrossAttentionSubCellClassifier'
        self.scaler = torch.cuda.amp.GradScaler()

        # Gradient accumulation
        self.gradient_accumulation_steps = getattr(args, 'gradient_accumulation_steps', 1)
        self.accum_iter = 0

        # Additional attributes
        self.batch_size = args.batch_size
        self.workers = args.workers
        self.global_step = 0

        # Check for embedding mode
        self.mae_embedding_path = getattr(args, 'mae_embedding_path', None)
        self.use_embedding_mode = self.mae_embedding_path is not None

    def build_model(self):
        if self.model is not None:
            raise ValueError("Model has been created. Do not create twice")

        args = self.args
        print(f"=> Creating model {self.model_name}")

        # Model parameters
        model_params = {
            'input_size': tuple(args.input_size),
            'patch_size': tuple(args.patch_size),
            'in_chans': args.in_chans,
            'num_classes': args.num_classes,
            'embed_dim': args.encoder_embed_dim,
            'depth': args.encoder_depth,
            'num_heads': args.encoder_num_heads,
            'drop_path_rate': getattr(args, 'drop_path', 0.0),
            'pos_embed_type': getattr(args, 'pos_embed_type', 'sincos'),
            'use_global_pool': getattr(args, 'use_global_pool', True),
            'cross_attention_type': getattr(args, 'cross_attention_type', 'position_wise'),
            'pool_mode': getattr(args, 'pool_mode', 'concat'),
            'subcell_embed_dim': getattr(args, 'subcell_embed_dim', 1536),
            'subcell_proj_dim': getattr(args, 'subcell_proj_dim', None),
            'fusion_type': getattr(args, 'fusion_type', 'concat'),
        }

        self.model = ViT3DCrossAttentionSubCellClassifier(**model_params)

        print(f"   Cross-attention type: {model_params['cross_attention_type']}")
        print(f"   Pool mode: {model_params['pool_mode']}")
        print(f"   SubCell embed dim: {model_params['subcell_embed_dim']}")
        print(f"   Fusion type: {model_params['fusion_type']}")

        # Load pretrained MAE weights
        if args.pretrain is not None and os.path.exists(args.pretrain):
            print(f"=> Loading pretrained weights from {args.pretrain}")
            checkpoint = torch.load(args.pretrain, map_location='cpu')
            state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
            self.model.load_mae_encoder(state_dict, strict=False)
            print("=> Successfully loaded pretrained MAE encoder weights")

        # Freeze encoder if specified
        freeze_encoder = getattr(args, 'freeze_encoder', False)
        if freeze_encoder:
            print("=> Freezing MAE encoder (linear probing mode)")
            for name, param in self.model.named_parameters():
                if 'head' not in name and 'subcell_proj' not in name:
                    param.requires_grad = False

            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in self.model.parameters())
            print(f"   Trainable parameters: {trainable_params:,} / {total_params:,} "
                  f"({100.0 * trainable_params / total_params:.2f}%)")
        else:
            print("=> Training entire model (full fine-tuning)")

        # Loss function
        self.loss_fn = WeightedBCEWithLogitsLoss()

        self.wrap_model()

    def build_dataloader(self):
        if self.train_loader is not None:
            raise ValueError("Dataloader has been created. Do not create twice")

        print("=> Creating dataloaders with SubCell embeddings")
        args = self.args

        # Get SubCell embedding paths
        embedding_base_path = args.embedding_path
        train_embedding_path = os.path.join(embedding_base_path, "train.npy")
        val_embedding_path = os.path.join(embedding_base_path, "val.npy")
        test_embedding_path = os.path.join(embedding_base_path, "test.npy")

        # Get MAE embedding paths (optional - for fast training mode)
        train_mae_emb_path = None
        val_mae_emb_path = None
        test_mae_emb_path = None

        if self.use_embedding_mode:
            train_mae_emb_path = os.path.join(self.mae_embedding_path, "train.npy")
            val_mae_emb_path = os.path.join(self.mae_embedding_path, "val.npy")
            test_mae_emb_path = os.path.join(self.mae_embedding_path, "test.npy")
            print(f"   Mode: Embedding-only (fast training)")
            print(f"   MAE embedding path: {self.mae_embedding_path}")
        else:
            print(f"   Mode: Image loading (full forward pass)")

        # Create transforms (only needed for image mode)
        if self.use_embedding_mode:
            train_transform = None  # No transforms needed for embeddings
            val_transform = None
        else:
            train_transform = get_opencell_train_transforms(
                flip_prob=args.RandFlipd_prob,
                rotate_prob=args.RandRotate90d_prob
            )
            val_transform = get_opencell_val_transforms()

        # Create train dataset
        train_csv_path = os.path.join(args.csv_path, "train.csv")
        train_dataset = OpenCellLocalizationFusionDataset(
            csv_path=train_csv_path,
            localization_csv_path=args.localization_csv_path,
            embedding_path=train_embedding_path,
            split='train',
            transform=train_transform,
            cache_rate=0.0 if self.use_embedding_mode else args.cache_rate,
            num_workers=args.workers,
            use_max_projection=getattr(args, 'use_2d', False),
            grade_weights=getattr(args, 'grade_weights', None),
            z_slice_start=getattr(args, 'z_slice_start', None),
            z_slice_end=getattr(args, 'z_slice_end', None),
            mae_embedding_path=train_mae_emb_path,
        )

        print(f"   SubCell embedding dimension: {train_dataset.get_embedding_dim()}")
        if train_dataset.is_embedding_mode():
            print(f"   MAE embedding dimension: {train_dataset.get_mae_embedding_dim()}")

        # Workers: embedding mode can use more workers (just loading numpy arrays)
        # Image mode uses 0 workers in distributed mode to avoid GPU memory issues
        if self.use_embedding_mode:
            train_workers = args.workers
        else:
            train_workers = 0 if self.is_distributed else args.workers

        if self.is_distributed:
            sampler = DistributedSampler(train_dataset, num_replicas=self.world_size, rank=self.rank, shuffle=True)
            self.train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=args.batch_size, sampler=sampler,
                num_workers=train_workers, pin_memory=True
            )
        else:
            self.train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=args.batch_size, shuffle=True,
                num_workers=train_workers, pin_memory=True
            )

        print(f"   Train samples: {len(train_dataset)}")
        print(f"   Batch size: {args.batch_size}")
        print(f"   Iterations per epoch: {len(self.train_loader)}")

        # Create validation dataset
        val_csv_path = os.path.join(args.csv_path, "val.csv")
        if os.path.exists(val_csv_path) and os.path.exists(val_embedding_path):
            val_dataset = OpenCellLocalizationFusionDataset(
                csv_path=val_csv_path,
                localization_csv_path=args.localization_csv_path,
                embedding_path=val_embedding_path,
                split='val',
                transform=val_transform,
                cache_rate=0.0,
                num_workers=args.workers,
                use_max_projection=getattr(args, 'use_2d', False),
                grade_weights=getattr(args, 'grade_weights', None),
                z_slice_start=getattr(args, 'z_slice_start', None),
                z_slice_end=getattr(args, 'z_slice_end', None),
                mae_embedding_path=val_mae_emb_path,
            )

            val_workers = args.workers if self.use_embedding_mode else 0
            if self.is_distributed:
                val_sampler = DistributedSampler(val_dataset, num_replicas=self.world_size, rank=self.rank, shuffle=False)
                self.val_loader = torch.utils.data.DataLoader(
                    val_dataset, batch_size=args.batch_size, sampler=val_sampler,
                    num_workers=val_workers, pin_memory=True
                )
            else:
                self.val_loader = torch.utils.data.DataLoader(
                    val_dataset, batch_size=args.batch_size, shuffle=False,
                    num_workers=val_workers, pin_memory=True
                )

            print(f"   Val samples: {len(val_dataset)}")
        else:
            print("   Validation disabled (files not found)")

        # Create test dataset
        test_csv_path = os.path.join(args.csv_path, "test.csv")
        if os.path.exists(test_csv_path) and os.path.exists(test_embedding_path):
            test_dataset = OpenCellLocalizationFusionDataset(
                csv_path=test_csv_path,
                localization_csv_path=args.localization_csv_path,
                embedding_path=test_embedding_path,
                split='test',
                transform=val_transform,
                cache_rate=0.0,
                num_workers=args.workers,
                use_max_projection=getattr(args, 'use_2d', False),
                grade_weights=getattr(args, 'grade_weights', None),
                z_slice_start=getattr(args, 'z_slice_start', None),
                z_slice_end=getattr(args, 'z_slice_end', None),
                mae_embedding_path=test_mae_emb_path,
            )

            test_workers = args.workers if self.use_embedding_mode else 0
            self.test_loader = torch.utils.data.DataLoader(
                test_dataset, batch_size=args.batch_size, shuffle=False,
                num_workers=test_workers, pin_memory=True
            )
            print(f"   Test samples: {len(test_dataset)}")
        else:
            self.test_loader = None
            print("   Test disabled (files not found)")

    def build_optimizer(self):
        assert self.model is not None
        print("=> Creating optimizer")
        args = self.args

        # Scale LR by world size
        self.lr = args.lr * self.world_size

        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.wrapped_model.parameters()),
            lr=self.lr,
            betas=(args.beta1, args.beta2),
            weight_decay=args.weight_decay
        )

        print(f"   Optimizer: AdamW")
        print(f"   Learning rate: {self.lr:.6f}")
        print(f"   Weight decay: {args.weight_decay}")

    def epoch_train(self, epoch):
        args = self.args
        model = self.wrapped_model
        optimizer = self.optimizer
        scaler = self.scaler

        model.train()

        # Warmup
        warmup_steps = args.warmup_epochs * len(self.train_loader)
        warmup_start_lr = self.lr * 0.01

        for i, batch in enumerate(self.train_loader):
            subcell_emb = batch['subcell_embedding'].cuda(self.local_rank, non_blocking=True)
            labels = batch['label'].cuda(self.local_rank, non_blocking=True)

            # Warmup LR
            if self.global_step < warmup_steps:
                progress = min(self.global_step / warmup_steps, 1.0)
                current_lr = warmup_start_lr + (self.lr - warmup_start_lr) * progress
                for param_group in optimizer.param_groups:
                    param_group['lr'] = current_lr

            # Zero gradients at start of accumulation
            if self.accum_iter == 0:
                optimizer.zero_grad()

            # Forward pass - choose mode based on batch contents
            with torch.cuda.amp.autocast(True):
                if 'mae_embedding' in batch:
                    # Embedding mode (fast) - get underlying model for custom method
                    mae_emb = batch['mae_embedding'].cuda(self.local_rank, non_blocking=True)
                    underlying_model = model.module if hasattr(model, 'module') else model
                    logits = underlying_model.forward_from_embeddings(mae_emb, subcell_emb)
                else:
                    # Image mode (full forward)
                    images = batch['image'].cuda(self.local_rank, non_blocking=True)
                    logits = model(images, subcell_emb)

                loss = self.loss_fn(logits, labels)
                loss = loss / self.gradient_accumulation_steps

            # Backward
            scaler.scale(loss).backward()

            self.accum_iter += 1

            # Update weights
            if self.accum_iter >= self.gradient_accumulation_steps:
                scaler.step(optimizer)
                scaler.update()
                self.accum_iter = 0

            # Logging
            if self.rank == 0:
                log_dict = {
                    "loss": loss.item() * self.gradient_accumulation_steps,
                    "epoch": epoch,
                    "step": self.global_step,
                    "lr": optimizer.param_groups[0]['lr']
                }
                wandb.log(log_dict)

                if i % args.print_freq == 0:
                    print(f"Epoch {epoch}/{args.epochs} | "
                          f"Iter {i}/{len(self.train_loader)} | "
                          f"Loss: {loss.item() * self.gradient_accumulation_steps:.4f}", flush=True)

            self.global_step += 1

    def validate_epoch(self, epoch, loader=None, split='val'):
        if loader is None:
            loader = self.val_loader
        if loader is None:
            return None

        args = self.args
        model = self.wrapped_model
        model.eval()

        all_logits = []
        all_labels = []

        print(f"\n=> Running {split} evaluation...", flush=True)

        with torch.no_grad():
            for i, batch in enumerate(loader):
                subcell_emb = batch['subcell_embedding'].cuda(self.local_rank, non_blocking=True)
                labels = batch['label'].cuda(self.local_rank, non_blocking=True)

                with torch.cuda.amp.autocast(True):
                    if 'mae_embedding' in batch:
                        # Embedding mode (fast) - get underlying model for custom method
                        mae_emb = batch['mae_embedding'].cuda(self.local_rank, non_blocking=True)
                        underlying_model = model.module if hasattr(model, 'module') else model
                        logits = underlying_model.forward_from_embeddings(mae_emb, subcell_emb)
                    else:
                        # Image mode (full forward)
                        images = batch['image'].cuda(self.local_rank, non_blocking=True)
                        logits = model(images, subcell_emb)

                all_logits.append(logits.cpu())
                all_labels.append(labels.cpu())

        # Compute metrics
        all_logits = torch.cat(all_logits, dim=0).numpy()
        all_labels = torch.cat(all_labels, dim=0).numpy()
        all_probs = 1 / (1 + np.exp(-all_logits))  # Sigmoid

        # Binary predictions
        binary_labels = (all_labels > 0).astype(np.float32)

        # Compute metrics per class
        metrics = {}
        aps = []
        aucs = []

        for i, label_name in enumerate(LOCALIZATION_LABELS):
            y_true = binary_labels[:, i]
            y_prob = all_probs[:, i]

            if y_true.sum() > 0:
                ap = average_precision_score(y_true, y_prob)
                try:
                    auc = roc_auc_score(y_true, y_prob)
                except:
                    auc = 0.5
                aps.append(ap)
                aucs.append(auc)

        metrics['mAP'] = np.mean(aps) if aps else 0.0
        metrics['mAUC'] = np.mean(aucs) if aucs else 0.0

        # Loss
        loss = F.binary_cross_entropy_with_logits(
            torch.from_numpy(all_logits),
            torch.from_numpy(all_labels)
        ).item()
        metrics['loss'] = loss

        if self.rank == 0:
            print(f"=> {split.capitalize()} | Loss: {loss:.4f} | mAP: {metrics['mAP']:.4f} | mAUC: {metrics['mAUC']:.4f}\n")

            wandb.log({
                f"{split}_loss": loss,
                f"{split}_mAP": metrics['mAP'],
                f"{split}_mAUC": metrics['mAUC'],
                "epoch": epoch,
            })

        return metrics

    def run(self):
        args = self.args

        for epoch in range(args.start_epoch, args.epochs):
            if self.is_distributed and hasattr(self.train_loader, 'sampler'):
                if hasattr(self.train_loader.sampler, 'set_epoch'):
                    self.train_loader.sampler.set_epoch(epoch)

            current_lr = self.adjust_learning_rate(epoch)

            if self.rank == 0:
                print(f"\n{'='*60}")
                print(f"Epoch {epoch}/{args.epochs} | LR: {current_lr:.6f}")
                print(f"{'='*60}")

            # Train
            self.epoch_train(epoch)

            # Validate
            eval_freq = getattr(args, 'eval_freq', 1)
            if (epoch + 1) % eval_freq == 0:
                val_metrics = self.validate_epoch(epoch, self.val_loader, 'val')

                # Test
                if self.test_loader is not None:
                    test_metrics = self.validate_epoch(epoch, self.test_loader, 'test')

            # Save checkpoint
            if (epoch + 1) % args.save_freq == 0:
                self.save_checkpoint(epoch)

        if self.rank == 0:
            print("\nTraining completed!")
