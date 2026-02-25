import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score
from collections import defaultdict

from lib.trainers.base_trainer import BaseTrainer
from lib.models import ViT3DClassifier, ViT2DClassifier, ViT3DCrossAttentionClassifier
from data.opencell.localization_dataset import OpenCellLocalizationDataset, LOCALIZATION_LABELS
from data.opencell.transforms import (
    get_opencell_train_transforms,
    get_opencell_val_transforms,
    get_opencell_2d_train_transforms,
    get_opencell_2d_val_transforms
)


class WeightedBCEWithLogitsLoss(nn.Module):
    """
    Weighted BCE loss for multi-label classification with grade weights.
    Each label can have a different weight based on annotation grade.
    """
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        Args:
            logits: [B, num_classes] predicted logits
            targets: [B, num_classes] target labels with weights (0.0 to 1.0)

        The targets contain weighted labels where:
        - 0.0 means the label is not present
        - 0.25, 0.5, 1.0 represent grade 1, 2, 3 annotations respectively
        """
        # Standard BCE with logits
        loss = F.binary_cross_entropy_with_logits(
            logits, targets, reduction='none'
        )

        # Apply sample weighting: give more weight to positive samples
        # Weight positive samples more heavily to handle class imbalance
        pos_weight = (targets > 0).float() * 2.0 + 1.0
        loss = loss * pos_weight

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class LocalizationTrainer(BaseTrainer):
    """
    Trainer for protein localization classification using ViT.
    Supports both 2D and 3D models.

    Supports two modes:
    1. Image mode (default): Load images, run through MAE encoder
    2. Embedding mode: Use precomputed MAE embeddings (much faster when encoder is frozen)

    To use embedding mode, set `mae_embedding_path` in config pointing to a directory
    with train.npy, val.npy, test.npy files containing precomputed MAE embeddings.
    """
    def __init__(self, args):
        super().__init__(args)
        self.model_name = args.arch
        self.scaler = torch.cuda.amp.GradScaler()
        self.use_2d = getattr(args, 'use_2d', False)  # Whether to use 2D max projection

        # Gradient accumulation
        self.gradient_accumulation_steps = getattr(args, 'gradient_accumulation_steps', 1)
        self.accum_iter = 0  # Track accumulation iterations

        # Additional attributes for compatibility
        self.batch_size = args.batch_size
        self.workers = args.workers
        self.iters_per_epoch = 0
        self.val_iters = 0

        # Check for embedding mode
        self.mae_embedding_path = getattr(args, 'mae_embedding_path', None)
        self.use_embedding_mode = self.mae_embedding_path is not None
        # Optional source CSV directory for image_path-based embedding lookup
        self.mae_embedding_csv_path = getattr(args, 'mae_embedding_csv_path', None)

    def build_model(self):
        if self.model is None:
            args = self.args
            print(f"=> Creating model {self.model_name}")

            # Model parameters from config
            model_params = {
                'input_size': tuple(args.input_size) if hasattr(args, 'input_size') else (100, 176, 176),
                'patch_size': tuple(args.patch_size) if hasattr(args, 'patch_size') else (10, 8, 8),
                'in_chans': args.in_chans,
                'num_classes': args.num_classes,
                'embed_dim': args.encoder_embed_dim,
                'depth': args.encoder_depth,
                'num_heads': args.encoder_num_heads,
                'drop_path_rate': getattr(args, 'drop_path', 0.0),
                'pos_embed_type': getattr(args, 'pos_embed_type', 'sincos'),
                'use_global_pool': getattr(args, 'use_global_pool', True),
            }

            # Create model based on architecture
            if self.model_name == 'ViT3DCrossAttentionClassifier':
                # Cross-attention classifier for dual-channel processing
                model_params['cross_attention_type'] = getattr(args, 'cross_attention_type', 'position_wise')
                model_params['pool_mode'] = getattr(args, 'pool_mode', 'concat')
                self.model = ViT3DCrossAttentionClassifier(**model_params)
                print(f"   Cross-attention type: {model_params['cross_attention_type']}")
                print(f"   Pool mode: {model_params['pool_mode']}")
            elif self.use_2d:
                # For 2D, input_size and patch_size should be 2D
                model_params['input_size'] = model_params['input_size'][:2] if len(model_params['input_size']) == 3 else model_params['input_size']
                model_params['patch_size'] = model_params['patch_size'][:2] if len(model_params['patch_size']) == 3 else model_params['patch_size']
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
                # Freeze all parameters except the classification head
                for name, param in self.model.named_parameters():
                    if 'head' not in name:  # Only keep head trainable
                        param.requires_grad = False

                # Print number of trainable vs frozen parameters
                trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
                total_params = sum(p.numel() for p in self.model.parameters())
                print(f"   Trainable parameters: {trainable_params:,} / {total_params:,} "
                      f"({100.0 * trainable_params / total_params:.2f}%)")
            else:
                print("=> Training entire model (full fine-tuning)")

            # Setup loss function
            self.loss_fn = WeightedBCEWithLogitsLoss()

            self.wrap_model()
        else:
            raise ValueError("=> Model has been created. Do not create twice")

    def build_optimizer(self):
        assert self.model is not None, "Model is not created yet. Please create model first."
        print("=> Creating optimizer")
        args = self.args

        # Only optimize parameters that require gradients
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
        print(f"   Betas: ({args.beta1}, {args.beta2})")
        print(f"   Weight decay: {args.weight_decay}")
        print(f"   Gradient accumulation steps: {self.gradient_accumulation_steps}")

    def build_dataloader(self):
        if self.train_loader is None:
            print("=> Creating dataloaders")
            args = self.args

            # Get MAE embedding paths (optional - for fast training mode)
            train_mae_emb_path = None
            val_mae_emb_path = None
            train_mae_emb_csv = None
            val_mae_emb_csv = None
            combined_emb = None
            combined_lookup = None

            if self.use_embedding_mode:
                print(f"   Mode: Embedding-only (fast training)")
                print(f"   MAE embedding path: {self.mae_embedding_path}")

                if self.mae_embedding_csv_path:
                    # Kfold mode: a localization fold's train/val CSVs may reference cells that
                    # were in any of the extraction-source splits (train/val/test of dataset1).
                    # Build one combined embedding array + lookup covering ALL source splits so
                    # every cell can be found regardless of which split it came from.
                    print(f"   MAE embedding CSV:  {self.mae_embedding_csv_path}")
                    print(f"   Building combined embedding lookup from all source splits...")
                    all_embs = []
                    combined_lookup = {}
                    offset = 0
                    for sname in ('train', 'val', 'test'):
                        npy = os.path.join(self.mae_embedding_path, f'{sname}.npy')
                        csv_s = os.path.join(self.mae_embedding_csv_path, f'{sname}.csv')
                        if os.path.exists(npy) and os.path.exists(csv_s):
                            arr = np.load(npy)
                            sdf = pd.read_csv(csv_s)
                            assert len(arr) == len(sdf), (
                                f"{sname}: npy rows ({len(arr)}) != csv rows ({len(sdf)})"
                            )
                            for local_i, (_, row) in enumerate(sdf.iterrows()):
                                combined_lookup[row['image_path']] = offset + local_i
                            all_embs.append(arr)
                            offset += len(arr)
                            print(f"     {sname}: {len(arr)} embeddings from {csv_s}")
                    if all_embs:
                        combined_emb = np.concatenate(all_embs, axis=0)
                        print(f"   Combined: {combined_emb.shape}, {len(combined_lookup)} unique paths")
                else:
                    # Legacy / non-kfold: separate per-split npy files aligned with csv_path.
                    train_mae_emb_path = os.path.join(self.mae_embedding_path, "train.npy")
                    val_mae_emb_path = os.path.join(self.mae_embedding_path, "val.npy")
            else:
                print(f"   Mode: Image loading (full forward pass)")

            # Get transforms (only needed for image mode)
            if self.use_embedding_mode:
                train_transform = None  # No transforms needed for embeddings
                val_transform = None
            elif self.use_2d:
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
            train_dataset = OpenCellLocalizationDataset(
                csv_path=os.path.join(args.csv_path, 'train.csv'),
                localization_csv_path=args.localization_csv_path,
                split='train',
                transform=train_transform,
                cache_rate=0.0 if self.use_embedding_mode else args.cache_rate,
                use_max_projection=self.use_2d,
                grade_weights=getattr(args, 'grade_weights', None),
                z_slice_start=getattr(args, 'z_slice_start', None),
                z_slice_end=getattr(args, 'z_slice_end', None),
                mae_embedding_path=train_mae_emb_path,
                mae_embedding_csv_path=train_mae_emb_csv,
                mae_embedding_array=combined_emb,
                mae_embedding_lookup_dict=combined_lookup,
            )

            val_dataset = OpenCellLocalizationDataset(
                csv_path=os.path.join(args.csv_path, 'val.csv'),
                localization_csv_path=args.localization_csv_path,
                split='val',
                transform=val_transform,
                cache_rate=0.0,
                use_max_projection=self.use_2d,
                grade_weights=getattr(args, 'grade_weights', None),
                z_slice_start=getattr(args, 'z_slice_start', None),
                z_slice_end=getattr(args, 'z_slice_end', None),
                mae_embedding_path=val_mae_emb_path,
                mae_embedding_csv_path=val_mae_emb_csv,
                mae_embedding_array=combined_emb,
                mae_embedding_lookup_dict=combined_lookup,
            )

            # Print embedding info if in embedding mode
            if train_dataset.is_embedding_mode():
                print(f"   MAE embedding dimension: {train_dataset.get_mae_embedding_dim()}")

            # Print label distribution
            print("\n==> Training set label distribution:")
            train_dist = train_dataset.get_label_distribution()
            for label, count in sorted(train_dist.items(), key=lambda x: -x[1]):
                print(f"  {label}: {count}")

            # Create samplers
            if args.distributed:
                train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
                val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)
            else:
                train_sampler = None
                val_sampler = None

            # Workers: embedding mode can use more workers (just loading numpy arrays)
            # Image mode uses fewer workers in distributed mode to avoid GPU memory issues
            if self.use_embedding_mode:
                train_workers = self.workers
                val_workers = self.workers
            else:
                train_workers = self.workers
                val_workers = self.workers

            # Create dataloaders
            self.train_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=self.batch_size,
                shuffle=(train_sampler is None),
                num_workers=train_workers,
                pin_memory=True,
                sampler=train_sampler,
                drop_last=True
            )
            self.iters_per_epoch = len(self.train_loader)

            self.val_loader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=val_workers,
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
                self.evaluate(epoch=epoch, niters=niters)

            # Save checkpoint
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
            target = data['label']

            if args.gpu is not None:
                target = target.cuda(args.gpu, non_blocking=True)

            # Zero gradients at the start of accumulation
            if self.accum_iter == 0:
                optimizer.zero_grad()

            # Forward pass - choose mode based on batch contents
            with torch.cuda.amp.autocast(True):
                if 'mae_embedding' in data:
                    # Embedding mode (fast) - get underlying model for custom method
                    mae_emb = data['mae_embedding'].cuda(args.gpu, non_blocking=True)
                    underlying_model = model.module if hasattr(model, 'module') else model
                    logits = underlying_model.forward_from_embeddings(mae_emb)
                else:
                    # Image mode (full forward)
                    image = data['image'].cuda(args.gpu, non_blocking=True)
                    logits = model(image)

                loss = loss_fn(logits, target)
                # Scale loss for gradient accumulation
                loss = loss / self.gradient_accumulation_steps

            # Backward pass
            scaler.scale(loss).backward()

            # Increment accumulation counter
            self.accum_iter += 1

            # Only update weights after accumulating enough gradients
            if self.accum_iter >= self.gradient_accumulation_steps:
                scaler.step(optimizer)
                scaler.update()
                # Reset accumulation counter
                self.accum_iter = 0

            # Logging
            if i % args.print_freq == 0:
                lr = optimizer.param_groups[0]['lr']
                # Unscale loss for logging
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
            target = data['label']

            if args.gpu is not None:
                target = target.cuda(args.gpu, non_blocking=True)

            # Forward pass - choose mode based on batch contents
            with torch.cuda.amp.autocast(True):
                if 'mae_embedding' in data:
                    # Embedding mode (fast) - get underlying model for custom method
                    mae_emb = data['mae_embedding'].cuda(args.gpu, non_blocking=True)
                    underlying_model = model.module if hasattr(model, 'module') else model
                    logits = underlying_model.forward_from_embeddings(mae_emb)
                else:
                    # Image mode (full forward)
                    image = data['image'].cuda(args.gpu, non_blocking=True)
                    logits = model(image)

                loss = loss_fn(logits, target)

            # Collect predictions and targets
            all_logits.append(logits.cpu())
            all_targets.append(target.cpu())
            total_loss += loss.item() * target.size(0)
            num_samples += target.size(0)

        # Concatenate all predictions and targets
        all_logits = torch.cat(all_logits, dim=0).numpy()
        all_targets = torch.cat(all_targets, dim=0).numpy()
        avg_loss = total_loss / num_samples

        # Compute metrics
        # Convert logits to probabilities
        all_probs = torch.sigmoid(torch.from_numpy(all_logits)).numpy()

        # Binarize targets for metrics (any weight > 0 means the label is present)
        binary_targets = (all_targets > 0).astype(float)

        # Compute mAP (mean average precision)
        try:
            mAP = average_precision_score(binary_targets, all_probs, average='macro')
        except:
            mAP = 0.0

        # Compute AUC
        try:
            macro_auc = roc_auc_score(binary_targets, all_probs, average='macro')
        except:
            macro_auc = 0.0

        # Compute F1 (with threshold 0.5)
        binary_preds = (all_probs > 0.5).astype(float)
        macro_f1 = f1_score(binary_targets, binary_preds, average='macro', zero_division=0)
        micro_f1 = f1_score(binary_targets, binary_preds, average='micro', zero_division=0)

        # Per-class AP
        per_class_ap = {}
        for i, label in enumerate(LOCALIZATION_LABELS):
            if binary_targets[:, i].sum() > 0:  # Only compute if label exists
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

            # Log per-class metrics
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
