"""
Trainer for protein abundance prediction (regression task).

Supports both 2D and 3D models with pretrained MAE weights.
"""

import os
import torch
import torch.nn as nn
import wandb
import numpy as np
from scipy.stats import pearsonr, spearmanr

from lib.trainers.base_trainer import BaseTrainer
from lib.models import ViT3DClassifier, ViT2DClassifier, ViT3DCrossAttentionClassifier
from data.opencell.abundance_dataset import OpenCellAbundanceDataset
from data.opencell.transforms import (
    get_opencell_train_transforms,
    get_opencell_val_transforms,
    get_opencell_2d_train_transforms,
    get_opencell_2d_val_transforms
)


class AbundanceTrainer(BaseTrainer):
    """
    Trainer for protein abundance prediction using ViT.
    Supports both 2D and 3D models.
    """

    def __init__(self, args):
        super().__init__(args)
        self.model_name = args.arch
        self.scaler = torch.cuda.amp.GradScaler()
        self.use_2d = getattr(args, 'use_2d', False)

        # Gradient accumulation
        self.gradient_accumulation_steps = getattr(args, 'gradient_accumulation_steps', 1)
        self.accum_iter = 0

        # Additional attributes
        self.batch_size = args.batch_size
        self.workers = args.workers
        self.iters_per_epoch = 0
        self.val_iters = 0

        # For tracking best model
        self.best_pearson = -1.0

    def build_model(self):
        if self.model is None:
            args = self.args
            print(f"=> Creating model {self.model_name}")

            # Model parameters from config
            # Note: num_classes=1 for regression
            model_params = {
                'input_size': tuple(args.input_size) if hasattr(args, 'input_size') else (100, 176, 176),
                'patch_size': tuple(args.patch_size) if hasattr(args, 'patch_size') else (10, 8, 8),
                'in_chans': args.in_chans,
                'num_classes': 1,  # Single output for regression
                'embed_dim': args.encoder_embed_dim,
                'depth': args.encoder_depth,
                'num_heads': args.encoder_num_heads,
                'drop_path_rate': getattr(args, 'drop_path', 0.0),
                'pos_embed_type': getattr(args, 'pos_embed_type', 'sincos'),
                'use_global_pool': getattr(args, 'use_global_pool', True),
            }

            # Create model based on architecture
            if self.model_name == 'ViT3DCrossAttentionClassifier':
                model_params['cross_attention_type'] = getattr(args, 'cross_attention_type', 'position_wise')
                model_params['pool_mode'] = getattr(args, 'pool_mode', 'concat')
                self.model = ViT3DCrossAttentionClassifier(**model_params)
                print(f"   Cross-attention type: {model_params['cross_attention_type']}")
                print(f"   Pool mode: {model_params['pool_mode']}")
            elif self.use_2d:
                model_params['input_size'] = model_params['input_size'][1:] if len(model_params['input_size']) == 3 else model_params['input_size']
                model_params['patch_size'] = model_params['patch_size'][1:] if len(model_params['patch_size']) == 3 else model_params['patch_size']
                self.model = ViT2DClassifier(**model_params)
            else:
                self.model = ViT3DClassifier(**model_params)

            # Load pretrained MAE weights if provided
            if args.pretrain is not None and os.path.exists(args.pretrain):
                print(f"=> Loading pretrained weights from {args.pretrain}")
                checkpoint = torch.load(args.pretrain, map_location='cpu')

                if 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint

                # Load encoder weights
                self.model.load_mae_encoder(state_dict, strict=False)
                print("=> Successfully loaded pretrained MAE encoder weights")

            # Freeze encoder if specified (linear probing)
            freeze_encoder = getattr(args, 'freeze_encoder', False)
            if freeze_encoder:
                print("=> Freezing encoder (linear probing mode)")
                for name, param in self.model.named_parameters():
                    if 'head' not in name:
                        param.requires_grad = False

                trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
                total_params = sum(p.numel() for p in self.model.parameters())
                print(f"   Trainable parameters: {trainable_params:,} / {total_params:,} "
                      f"({100.0 * trainable_params / total_params:.2f}%)")
            else:
                print("=> Training entire model (full fine-tuning)")

            # Setup loss function (MSE for regression)
            self.loss_fn = nn.MSELoss()

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

    def build_dataloader(self):
        if self.train_loader is None:
            print("=> Creating dataloaders")
            args = self.args

            # Get transforms
            if self.use_2d:
                train_transform = get_opencell_2d_train_transforms(
                    flip_prob=args.RandFlipd_prob,
                    rotate_prob=args.RandRotate90d_prob
                )
                val_transform = get_opencell_2d_val_transforms()
            else:
                train_transform = get_opencell_train_transforms(
                    flip_prob=args.RandFlipd_prob,
                    rotate_prob=args.RandRotate90d_prob
                )
                val_transform = get_opencell_val_transforms()

            # Create datasets
            train_dataset = OpenCellAbundanceDataset(
                csv_path=os.path.join(args.csv_path, 'train.csv'),
                abundance_csv_path=args.abundance_csv_path,
                split='train',
                transform=train_transform,
                cache_rate=args.cache_rate,
                use_max_projection=self.use_2d,
                target_column=args.target_column,
                log_transform=getattr(args, 'log_transform', True),
                normalize_target=getattr(args, 'normalize_target', True)
            )

            # Store normalization stats for later
            self.norm_stats = train_dataset.get_normalization_stats()

            val_dataset = OpenCellAbundanceDataset(
                csv_path=os.path.join(args.csv_path, 'val.csv'),
                abundance_csv_path=args.abundance_csv_path,
                split='val',
                transform=val_transform,
                cache_rate=args.cache_rate,
                use_max_projection=self.use_2d,
                target_column=args.target_column,
                log_transform=getattr(args, 'log_transform', True),
                normalize_target=getattr(args, 'normalize_target', True)
            )

            # Print target statistics
            print("\n==> Training set target statistics:")
            train_stats = train_dataset.get_target_statistics()
            for key, value in train_stats.items():
                print(f"  {key}: {value}")

            # Create samplers
            if args.distributed:
                train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
                val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)
            else:
                train_sampler = None
                val_sampler = None

            # Create dataloaders
            self.train_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=self.batch_size,
                shuffle=(train_sampler is None),
                num_workers=self.workers,
                pin_memory=True,
                sampler=train_sampler,
                drop_last=True
            )
            self.iters_per_epoch = len(self.train_loader)

            self.val_loader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.workers,
                pin_memory=True,
                sampler=val_sampler,
                drop_last=False
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

            # Train for one epoch
            niters = self.epoch_train(epoch, niters)

            # Evaluate
            if epoch == 0 or (epoch + 1) % args.eval_freq == 0:
                metrics = self.evaluate(epoch=epoch, niters=niters)

                # Save best model based on Pearson correlation
                if metrics['pearson'] > self.best_pearson:
                    self.best_pearson = metrics['pearson']
                    if args.rank == 0:
                        self.save_checkpoint(epoch, is_best=True, scaler=self.scaler.state_dict())

            # Save checkpoint periodically
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
            # Adjust learning rate
            self.adjust_learning_rate(epoch + i / self.iters_per_epoch, args)

            # Get data
            image = data['image']
            target = data['label']

            if args.gpu is not None:
                image = image.cuda(args.gpu, non_blocking=True)
                target = target.cuda(args.gpu, non_blocking=True)

            # Ensure target has correct shape for MSE loss
            if target.dim() == 1:
                target = target.unsqueeze(1)

            # Zero gradients at the start of accumulation
            if self.accum_iter == 0:
                optimizer.zero_grad()

            # Forward pass with mixed precision
            with torch.cuda.amp.autocast(True):
                predictions = model(image)
                loss = loss_fn(predictions, target)
                loss = loss / self.gradient_accumulation_steps

            # Backward pass
            scaler.scale(loss).backward()

            # Increment accumulation counter
            self.accum_iter += 1

            # Update weights after accumulation
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

        all_predictions = []
        all_targets = []
        total_loss = 0.0
        num_samples = 0

        for i, data in enumerate(val_loader):
            image = data['image']
            target = data['label']

            if args.gpu is not None:
                image = image.cuda(args.gpu, non_blocking=True)
                target = target.cuda(args.gpu, non_blocking=True)

            if target.dim() == 1:
                target = target.unsqueeze(1)

            # Forward pass
            with torch.cuda.amp.autocast(True):
                predictions = model(image)
                loss = loss_fn(predictions, target)

            # Collect predictions and targets
            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(target.cpu().numpy())
            total_loss += loss.item() * image.size(0)
            num_samples += image.size(0)

        # Concatenate all predictions and targets
        all_predictions = np.concatenate(all_predictions, axis=0).flatten()
        all_targets = np.concatenate(all_targets, axis=0).flatten()
        avg_loss = total_loss / num_samples

        # Compute metrics
        mse = np.mean((all_predictions - all_targets) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(all_predictions - all_targets))

        # Correlation metrics
        try:
            pearson_corr, pearson_p = pearsonr(all_predictions, all_targets)
        except:
            pearson_corr, pearson_p = 0.0, 1.0

        try:
            spearman_corr, spearman_p = spearmanr(all_predictions, all_targets)
        except:
            spearman_corr, spearman_p = 0.0, 1.0

        # R-squared
        ss_res = np.sum((all_targets - all_predictions) ** 2)
        ss_tot = np.sum((all_targets - np.mean(all_targets)) ** 2)
        r_squared = 1 - (ss_res / (ss_tot + 1e-8))

        metrics = {
            'loss': avg_loss,
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'pearson': pearson_corr,
            'spearman': spearman_corr,
            'r_squared': r_squared
        }

        # Print results
        print(f"\n==> Epoch {epoch:04d} Evaluation Results:")
        print(f"  Loss (MSE): {avg_loss:.04f}")
        print(f"  RMSE: {rmse:.04f}")
        print(f"  MAE: {mae:.04f}")
        print(f"  Pearson r: {pearson_corr:.04f} (p={pearson_p:.2e})")
        print(f"  Spearman rho: {spearman_corr:.04f} (p={spearman_p:.2e})")
        print(f"  R-squared: {r_squared:.04f}")
        print(f"  Best Pearson so far: {self.best_pearson:.04f}")

        # Log to wandb
        if args.rank == 0:
            wandb.log({
                'val_loss': avg_loss,
                'val_mse': mse,
                'val_rmse': rmse,
                'val_mae': mae,
                'val_pearson': pearson_corr,
                'val_spearman': spearman_corr,
                'val_r_squared': r_squared,
            }, step=niters)

        return metrics

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
