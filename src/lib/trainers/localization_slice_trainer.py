"""
Trainer for protein localization classification using slice aggregation.

Uses a 2D ViT encoder to process individual slices and aggregates
embeddings across slices for volume-level classification.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score

from lib.trainers.base_trainer import BaseTrainer
from lib.models import ViT2DSliceAggregateClassifier
from data.opencell.localization_slice_dataset import (
    OpenCellLocalizationSliceDataset,
    collate_slices
)
from data.opencell.localization_dataset import LOCALIZATION_LABELS
from data.opencell.transforms import get_opencell_2d_train_transforms, get_opencell_2d_val_transforms


class WeightedBCEWithLogitsLoss(nn.Module):
    """Weighted BCE loss for multi-label classification with grade weights."""

    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, logits, targets):
        loss = F.binary_cross_entropy_with_logits(
            logits, targets, reduction='none'
        )
        pos_weight = (targets > 0).float() * 2.0 + 1.0
        loss = loss * pos_weight

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class LocalizationSliceTrainer(BaseTrainer):
    """
    Trainer for protein localization using slice aggregation.

    Key differences from standard LocalizationTrainer:
    - Uses OpenCellLocalizationSliceDataset (returns multiple slices per sample)
    - Uses ViT2DSliceAggregateClassifier (aggregates slice embeddings)
    - Custom collate function for variable-length slice sequences
    """

    def __init__(self, args):
        super().__init__(args)
        self.model_name = args.arch
        self.scaler = torch.cuda.amp.GradScaler()

        # Slice parameters
        self.slice_start = getattr(args, 'slice_start', 45)
        self.slice_end = getattr(args, 'slice_end', 55)
        self.aggregation = getattr(args, 'aggregation', 'mean')

        # Gradient accumulation
        self.gradient_accumulation_steps = getattr(args, 'gradient_accumulation_steps', 1)
        self.accum_iter = 0

        # Additional attributes
        self.batch_size = args.batch_size
        self.workers = args.workers
        self.iters_per_epoch = 0
        self.val_iters = 0

    def build_model(self):
        if self.model is None:
            args = self.args
            print(f"=> Creating model {self.model_name}")
            print(f"   Using slice aggregation: slices {self.slice_start}-{self.slice_end}")
            print(f"   Aggregation method: {self.aggregation}")

            # Model parameters
            model_params = {
                'input_size': tuple(args.input_size) if hasattr(args, 'input_size') else (176, 176),
                'patch_size': tuple(args.patch_size) if hasattr(args, 'patch_size') else (8, 8),
                'in_chans': args.in_chans,
                'num_classes': args.num_classes,
                'embed_dim': args.encoder_embed_dim,
                'depth': args.encoder_depth,
                'num_heads': args.encoder_num_heads,
                'drop_path_rate': getattr(args, 'drop_path', 0.0),
                'pos_embed_type': getattr(args, 'pos_embed_type', 'sincos'),
                'use_global_pool': getattr(args, 'use_global_pool', True),
                'aggregation': self.aggregation,
            }

            self.model = ViT2DSliceAggregateClassifier(**model_params)

            # Load pretrained MAE2D weights
            if args.pretrain is not None and os.path.exists(args.pretrain):
                print(f"=> Loading pretrained weights from {args.pretrain}")
                checkpoint = torch.load(args.pretrain, map_location='cpu')

                if 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint

                self.model.load_mae_encoder(state_dict, strict=False)
                print("=> Successfully loaded pretrained MAE2D encoder weights")

            # Freeze encoder if specified (linear probing)
            freeze_encoder = getattr(args, 'freeze_encoder', False)
            if freeze_encoder:
                print("=> Freezing encoder (linear probing mode)")
                for name, param in self.model.named_parameters():
                    # Keep head and slice_attention trainable
                    if 'head' not in name and 'slice_attention' not in name:
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
        else:
            raise ValueError("=> Model has been created. Do not create twice")

    def build_optimizer(self):
        assert self.model is not None, "Model is not created yet."
        print("=> Creating optimizer")
        args = self.args

        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        print(f"   Optimizing {len(trainable_params)} parameter groups")

        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            weight_decay=args.weight_decay
        )

        print(f"   Optimizer: AdamW")
        print(f"   Learning rate: {args.lr:.6f}")
        print(f"   Weight decay: {args.weight_decay}")
        print(f"   Gradient accumulation steps: {self.gradient_accumulation_steps}")

    def build_dataloader(self):
        if self.train_loader is None:
            print("=> Creating dataloaders")
            args = self.args

            # Get transforms (2D transforms for individual slices)
            train_transform = get_opencell_2d_train_transforms(
                flip_prob=args.RandFlipd_prob,
                rotate_prob=args.RandRotate90d_prob
            )
            val_transform = get_opencell_2d_val_transforms()

            # Create slice datasets
            train_dataset = OpenCellLocalizationSliceDataset(
                csv_path=os.path.join(args.csv_path, 'train.csv'),
                localization_csv_path=args.localization_csv_path,
                split='train',
                transform=train_transform,
                slice_start=self.slice_start,
                slice_end=self.slice_end,
                grade_weights=getattr(args, 'grade_weights', None)
            )

            val_dataset = OpenCellLocalizationSliceDataset(
                csv_path=os.path.join(args.csv_path, 'val.csv'),
                localization_csv_path=args.localization_csv_path,
                split='val',
                transform=val_transform,
                slice_start=self.slice_start,
                slice_end=self.slice_end,
                grade_weights=getattr(args, 'grade_weights', None)
            )

            # Print info
            print(f"\n==> Training set: {len(train_dataset)} samples")
            print(f"==> Validation set: {len(val_dataset)} samples")
            print(f"==> Slices per sample: {self.slice_end - self.slice_start + 1}")

            # Print label distribution
            print("\n==> Training set label distribution:")
            train_dist = train_dataset.get_label_distribution()
            for label, count in sorted(train_dist.items(), key=lambda x: -x[1]):
                print(f"  {label}: {count}")

            # Samplers
            if args.distributed:
                train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
                val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)
            else:
                train_sampler = None
                val_sampler = None

            # Create dataloaders with custom collate function
            self.train_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=self.batch_size,
                shuffle=(train_sampler is None),
                num_workers=self.workers,
                pin_memory=True,
                sampler=train_sampler,
                drop_last=True,
                collate_fn=collate_slices
            )
            self.iters_per_epoch = len(self.train_loader)

            self.val_loader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.workers,
                pin_memory=True,
                sampler=val_sampler,
                drop_last=False,
                collate_fn=collate_slices
            )
            self.val_iters = len(self.val_loader)
        else:
            raise ValueError("Dataloader has been created. Do not create twice.")

    def run(self):
        args = self.args
        niters = args.start_epoch * self.iters_per_epoch

        for epoch in range(args.start_epoch, args.epochs):
            if args.distributed:
                self.train_loader.sampler.set_epoch(epoch)

            niters = self.epoch_train(epoch, niters)

            if epoch == 0 or (epoch + 1) % args.eval_freq == 0:
                self.evaluate(epoch=epoch, niters=niters)

            if epoch == 0 or (epoch + 1) % args.save_freq == 0:
                self.save_checkpoint(epoch, scaler=self.scaler.state_dict())

    def epoch_train(self, epoch, niters):
        args = self.args
        train_loader = self.train_loader
        model = self.wrapped_model
        optimizer = self.optimizer
        scaler = self.scaler
        loss_fn = self.loss_fn

        model.train()

        for i, data in enumerate(train_loader):
            self.adjust_learning_rate(epoch + i / self.iters_per_epoch, args)

            # Get data: slices [B, num_slices, C, H, W], mask [B, num_slices]
            slices = data['slices']
            mask = data['mask']
            target = data['label']

            if args.gpu is not None:
                slices = slices.cuda(args.gpu, non_blocking=True)
                mask = mask.cuda(args.gpu, non_blocking=True)
                target = target.cuda(args.gpu, non_blocking=True)

            if self.accum_iter == 0:
                optimizer.zero_grad()

            # Forward pass
            with torch.cuda.amp.autocast(True):
                logits = model(slices, mask=mask)
                loss = loss_fn(logits, target)
                loss = loss / self.gradient_accumulation_steps

            scaler.scale(loss).backward()

            self.accum_iter += 1

            if self.accum_iter >= self.gradient_accumulation_steps:
                scaler.step(optimizer)
                scaler.update()
                self.accum_iter = 0

            # Logging
            if i % args.print_freq == 0:
                lr = optimizer.param_groups[0]['lr']
                unscaled_loss = loss.item() * self.gradient_accumulation_steps
                print(f"Epoch: {epoch:03d}/{args.epochs} | "
                      f"Iter: {i:05d}/{self.iters_per_epoch} | "
                      f"TotalIter: {niters:06d} | "
                      f"Lr: {lr:.06f} | "
                      f"Loss: {unscaled_loss:.04f}")

                if args.rank == 0:
                    wandb.log({'lr': lr, 'train_loss': unscaled_loss}, step=niters)

            niters += 1

        return niters

    @torch.no_grad()
    def evaluate(self, epoch, niters):
        args = self.args
        model = self.wrapped_model
        val_loader = self.val_loader
        loss_fn = self.loss_fn

        model.eval()

        all_logits = []
        all_targets = []
        total_loss = 0.0
        num_samples = 0

        for i, data in enumerate(val_loader):
            slices = data['slices']
            mask = data['mask']
            target = data['label']

            if args.gpu is not None:
                slices = slices.cuda(args.gpu, non_blocking=True)
                mask = mask.cuda(args.gpu, non_blocking=True)
                target = target.cuda(args.gpu, non_blocking=True)

            with torch.cuda.amp.autocast(True):
                logits = model(slices, mask=mask)
                loss = loss_fn(logits, target)

            all_logits.append(logits.cpu())
            all_targets.append(target.cpu())
            total_loss += loss.item() * slices.size(0)
            num_samples += slices.size(0)

        # Compute metrics
        all_logits = torch.cat(all_logits, dim=0).numpy()
        all_targets = torch.cat(all_targets, dim=0).numpy()
        avg_loss = total_loss / num_samples

        all_probs = torch.sigmoid(torch.from_numpy(all_logits)).numpy()
        binary_targets = (all_targets > 0).astype(float)

        try:
            mAP = average_precision_score(binary_targets, all_probs, average='macro')
        except:
            mAP = 0.0

        try:
            macro_auc = roc_auc_score(binary_targets, all_probs, average='macro')
        except:
            macro_auc = 0.0

        binary_preds = (all_probs > 0.5).astype(float)
        macro_f1 = f1_score(binary_targets, binary_preds, average='macro', zero_division=0)
        micro_f1 = f1_score(binary_targets, binary_preds, average='micro', zero_division=0)

        # Per-class AP
        per_class_ap = {}
        for i, label in enumerate(LOCALIZATION_LABELS):
            if binary_targets[:, i].sum() > 0:
                try:
                    ap = average_precision_score(binary_targets[:, i], all_probs[:, i])
                    per_class_ap[label] = ap
                except:
                    per_class_ap[label] = 0.0

        # Print results
        print(f"\n==> Epoch {epoch:04d} Evaluation Results:")
        print(f"  Loss: {avg_loss:.04f}")
        print(f"  mAP: {mAP:.04f}")
        print(f"  Macro AUC: {macro_auc:.04f}")
        print(f"  Macro F1: {macro_f1:.04f}")
        print(f"  Micro F1: {micro_f1:.04f}")
        print(f"\n  Per-class AP:")
        for label, ap in sorted(per_class_ap.items(), key=lambda x: -x[1]):
            print(f"    {label}: {ap:.04f}")

        # Log to wandb
        if args.rank == 0:
            wandb.log({
                'val_loss': avg_loss,
                'mAP': mAP,
                'macro_auc': macro_auc,
                'macro_f1': macro_f1,
                'micro_f1': micro_f1,
            }, step=niters)

            for label, ap in per_class_ap.items():
                wandb.log({f'AP/{label}': ap}, step=niters)

    def adjust_learning_rate(self, epoch, args):
        """Cosine learning rate schedule with warmup."""
        init_lr = self.lr
        if epoch < args.warmup_epochs:
            cur_lr = init_lr * epoch / args.warmup_epochs
        else:
            cur_lr = init_lr * 0.5 * (1. + torch.cos(
                torch.tensor((epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs) * 3.14159)
            ).item())

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = cur_lr
