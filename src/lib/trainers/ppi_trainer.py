"""
Trainer for PPI (Protein-Protein Interaction) prediction via metric learning.

Trains a Siamese-style network with contrastive loss on protein pairs.
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from collections import defaultdict
from sklearn.metrics import roc_auc_score, average_precision_score

from lib.trainers.base_trainer import BaseTrainer
from lib.models.ppi_metric import PPIMetric3D, PPIMetric2D, PPIMetric3DCrossAttention
from data.opencell.ppi_dataset import OpenCellPPIDataset, OpenCellPPITestDataset
from data.opencell.transforms import (
    get_opencell_train_transforms,
    get_opencell_val_transforms,
    get_opencell_2d_train_transforms,
    get_opencell_2d_val_transforms
)


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for metric learning.

    L = y * d^2 + (1-y) * max(0, margin - d)^2

    where d is the distance (1 - cosine_similarity for normalized embeddings).
    """

    def __init__(self, margin=0.5):
        super().__init__()
        self.margin = margin

    def forward(self, z1, z2, labels):
        """
        Args:
            z1: Embeddings of first protein [B, D]
            z2: Embeddings of second protein [B, D]
            labels: 1 for positive pairs, 0 for negative pairs [B]

        Returns:
            Scalar loss
        """
        # Cosine similarity (embeddings are already L2 normalized)
        cos_sim = (z1 * z2).sum(dim=-1)

        # Distance = 1 - similarity
        distance = 1 - cos_sim

        # Contrastive loss
        pos_loss = labels * distance.pow(2)
        neg_loss = (1 - labels) * F.relu(self.margin - distance).pow(2)

        loss = (pos_loss + neg_loss).mean()
        return loss


class CosineContrastiveLoss(nn.Module):
    """
    Cosine-based contrastive loss.

    Positive pairs should have high similarity.
    Negative pairs should have low similarity.
    """

    def __init__(self, margin=0.3, temperature=0.1):
        super().__init__()
        self.margin = margin
        self.temperature = temperature

    def forward(self, z1, z2, labels):
        """
        Args:
            z1: Embeddings of first protein [B, D]
            z2: Embeddings of second protein [B, D]
            labels: 1 for positive pairs, 0 for negative pairs [B]
        """
        # Cosine similarity
        cos_sim = (z1 * z2).sum(dim=-1)

        # Positive: maximize similarity (loss = 1 - sim)
        # Negative: minimize similarity with margin (loss = max(0, sim - margin))
        pos_loss = labels * (1 - cos_sim)
        neg_loss = (1 - labels) * F.relu(cos_sim - (-1 + self.margin))

        loss = (pos_loss + neg_loss).mean()
        return loss


class BCEWithSimilarityLoss(nn.Module):
    """
    Binary cross-entropy loss using similarity as logits.

    Treats PPI prediction as binary classification where
    similarity score is the logit.
    """

    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, z1, z2, labels):
        """
        Args:
            z1: Embeddings [B, D]
            z2: Embeddings [B, D]
            labels: Binary labels [B]
        """
        # Cosine similarity as logits (scaled by temperature)
        cos_sim = (z1 * z2).sum(dim=-1)
        logits = cos_sim / self.temperature

        loss = self.bce(logits, labels)
        return loss


class PPITrainer(BaseTrainer):
    """
    Trainer for PPI prediction using metric learning.
    """

    def __init__(self, args):
        super().__init__(args)
        self.model_name = args.arch
        self.scaler = torch.cuda.amp.GradScaler()
        self.use_2d = getattr(args, 'use_2d', False)

        # Gradient accumulation
        self.gradient_accumulation_steps = getattr(args, 'gradient_accumulation_steps', 1)
        self.accum_iter = 0

        # Training params
        self.batch_size = args.batch_size
        self.workers = args.workers
        self.iters_per_epoch = 0
        self.val_iters = 0

        # Best model tracking
        self.best_roc_auc = 0.0

        # Embedding mode
        self.mae_embedding_path = getattr(args, 'mae_embedding_path', None)
        self.mae_embedding_csv_path = getattr(args, 'mae_embedding_csv_path', None)
        self.use_embedding_mode = self.mae_embedding_path is not None

        # Loss function
        loss_type = getattr(args, 'loss_type', 'contrastive')
        margin = getattr(args, 'margin', 0.5)
        temperature = getattr(args, 'temperature', 0.1)

        if loss_type == 'contrastive':
            self.loss_fn = ContrastiveLoss(margin=margin)
        elif loss_type == 'cosine':
            self.loss_fn = CosineContrastiveLoss(margin=margin, temperature=temperature)
        elif loss_type == 'bce':
            self.loss_fn = BCEWithSimilarityLoss(temperature=temperature)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

        print(f"Using {loss_type} loss with margin={margin}, temperature={temperature}")

    def build_model(self):
        if self.model is None:
            args = self.args
            print(f"=> Creating model {self.model_name}")

            model_params = {
                'input_size': tuple(args.input_size) if hasattr(args, 'input_size') else (100, 176, 176),
                'patch_size': tuple(args.patch_size) if hasattr(args, 'patch_size') else (10, 8, 8),
                'in_chans': args.in_chans,
                'embed_dim': args.encoder_embed_dim,
                'depth': args.encoder_depth,
                'num_heads': args.encoder_num_heads,
                'drop_path_rate': getattr(args, 'drop_path', 0.0),
                'pos_embed_type': getattr(args, 'pos_embed_type', 'sincos'),
                'use_global_pool': getattr(args, 'use_global_pool', True),
                'proj_hidden_dim': getattr(args, 'proj_hidden_dim', 512),
                'proj_output_dim': getattr(args, 'proj_output_dim', 128),
                'proj_num_layers': getattr(args, 'proj_num_layers', 2),
            }

            if self.model_name == 'PPIMetric3DCrossAttention':
                model_params['cross_attention_type'] = getattr(args, 'cross_attention_type', 'position_wise')
                model_params['pool_mode'] = getattr(args, 'pool_mode', 'concat')
                self.model = PPIMetric3DCrossAttention(**model_params)
                print(f"   Cross-attention type: {model_params['cross_attention_type']}")
                print(f"   Pool mode: {model_params['pool_mode']}")
            elif self.use_2d:
                model_params['input_size'] = model_params['input_size'][1:] if len(model_params['input_size']) == 3 else model_params['input_size']
                model_params['patch_size'] = model_params['patch_size'][1:] if len(model_params['patch_size']) == 3 else model_params['patch_size']
                self.model = PPIMetric2D(**model_params)
            else:
                self.model = PPIMetric3D(**model_params)

            print(f"   Projection head: {model_params['embed_dim']} -> {model_params['proj_hidden_dim']} -> {model_params['proj_output_dim']}")

            # Load pretrained MAE weights
            if args.pretrain is not None and os.path.exists(args.pretrain):
                print(f"=> Loading pretrained weights from {args.pretrain}")
                checkpoint = torch.load(args.pretrain, map_location='cpu')

                if 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint

                self.model.load_mae_encoder(state_dict, strict=False)
                print("=> Successfully loaded pretrained MAE encoder weights")

            # Freeze encoder if specified (linear probing)
            freeze_encoder = getattr(args, 'freeze_encoder', False)
            if freeze_encoder:
                print("=> Freezing encoder (linear probing mode)")
                for name, param in self.model.named_parameters():
                    if 'projection_head' not in name:
                        param.requires_grad = False

                trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
                total_params = sum(p.numel() for p in self.model.parameters())
                print(f"   Trainable parameters: {trainable_params:,} / {total_params:,} "
                      f"({100.0 * trainable_params / total_params:.2f}%)")
            else:
                print("=> Training entire model (full fine-tuning)")

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

            combined_emb = None
            combined_lookup = None
            train_emb_path = None
            val_emb_path = None

            if self.use_embedding_mode:
                train_transform = None
                val_transform = None
                print(f"   Mode: Embedding-only (fast training)")
                print(f"   MAE embedding path: {self.mae_embedding_path}")

                if self.mae_embedding_csv_path:
                    # Kfold mode: build combined lookup from all source splits so proteins'
                    # cells from any dataset1 split are found correctly.
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
                            print(f"     {sname}: {len(arr)} embeddings")
                    if all_embs:
                        combined_emb = np.concatenate(all_embs, axis=0)
                        print(f"   Combined: {combined_emb.shape}, {len(combined_lookup)} unique paths")
                else:
                    # Legacy / non-kfold: per-split npy files aligned positionally with csv_path.
                    train_emb_path = os.path.join(self.mae_embedding_path, 'train.npy')
                    val_emb_path = os.path.join(self.mae_embedding_path, 'val.npy')
            else:
                train_emb_path = None
                val_emb_path = None
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
                print(f"   Mode: Image loading (full forward pass)")

            # Create datasets
            train_dataset = OpenCellPPIDataset(
                csv_path=os.path.join(args.csv_path, 'train.csv'),
                ppi_csv_path=args.ppi_csv_path,
                abundance_csv_path=getattr(args, 'abundance_csv_path', None),
                split='train',
                transform=train_transform,
                cache_rate=0.0 if self.use_embedding_mode else args.cache_rate,
                use_max_projection=self.use_2d,
                pval_threshold=getattr(args, 'pval_threshold', 5.0),
                enrichment_threshold=getattr(args, 'enrichment_threshold', 2.5),
                stoichiometry_threshold=getattr(args, 'stoichiometry_threshold', 0.05),
                n_abundance_buckets=getattr(args, 'n_abundance_buckets', 10),
                n_negatives_per_positive=getattr(args, 'n_negatives_per_positive', 1),
                seed=args.seed,
                mae_embedding_path=train_emb_path,
                mae_embedding_array=combined_emb,
                mae_embedding_lookup_dict=combined_lookup,
            )

            val_dataset = OpenCellPPIDataset(
                csv_path=os.path.join(args.csv_path, 'val.csv'),
                ppi_csv_path=args.ppi_csv_path,
                abundance_csv_path=getattr(args, 'abundance_csv_path', None),
                split='val',
                transform=val_transform,
                cache_rate=0.0,
                use_max_projection=self.use_2d,
                pval_threshold=getattr(args, 'pval_threshold', 5.0),
                enrichment_threshold=getattr(args, 'enrichment_threshold', 2.5),
                stoichiometry_threshold=getattr(args, 'stoichiometry_threshold', 0.05),
                n_abundance_buckets=getattr(args, 'n_abundance_buckets', 10),
                n_negatives_per_positive=getattr(args, 'n_negatives_per_positive', 1),
                seed=args.seed + 1,
                mae_embedding_path=val_emb_path,
                mae_embedding_array=combined_emb,
                mae_embedding_lookup_dict=combined_lookup,
            )

            # Print dataset statistics
            print("\n==> Training set statistics:")
            train_stats = train_dataset.get_statistics()
            for key, value in train_stats.items():
                print(f"  {key}: {value}")

            print("\n==> Validation set statistics:")
            val_stats = val_dataset.get_statistics()
            for key, value in val_stats.items():
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

                # Save best model based on ROC-AUC
                if metrics['roc_auc'] > self.best_roc_auc:
                    self.best_roc_auc = metrics['roc_auc']
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

            labels = data['label']
            if args.gpu is not None:
                labels = labels.cuda(args.gpu, non_blocking=True)

            # Zero gradients
            if self.accum_iter == 0:
                optimizer.zero_grad()

            # Forward pass
            with torch.cuda.amp.autocast(True):
                if 'embedding1' in data:
                    # Embedding mode (fast)
                    emb1 = data['embedding1'].cuda(args.gpu, non_blocking=True)
                    emb2 = data['embedding2'].cuda(args.gpu, non_blocking=True)
                    underlying = model.module if hasattr(model, 'module') else model
                    z1, z2, similarity = underlying.forward_from_embeddings(emb1, emb2)
                else:
                    # Image mode
                    image1 = data['image1'].cuda(args.gpu, non_blocking=True)
                    image2 = data['image2'].cuda(args.gpu, non_blocking=True)
                    z1, z2, similarity = model(image1, image2)
                loss = loss_fn(z1, z2, labels)
                loss = loss / self.gradient_accumulation_steps

            # Backward pass
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

                # Compute batch accuracy
                with torch.no_grad():
                    predictions = (similarity > 0).float()
                    accuracy = (predictions == labels).float().mean().item()

                print(f"Epoch: {epoch:03d}/{args.epochs} | "
                      f"Iter: {i:05d}/{self.iters_per_epoch} | "
                      f"Lr: {lr:.06f} | "
                      f"Loss: {unscaled_loss:.04f} | "
                      f"Acc: {accuracy:.04f}")

                if args.rank == 0:
                    wandb.log({
                        'lr': lr,
                        'train_loss': unscaled_loss,
                        'train_accuracy': accuracy
                    }, step=niters)

            niters += 1

        return niters

    @torch.no_grad()
    def evaluate(self, epoch, niters):
        args = self.args
        model = self.wrapped_model
        val_loader = self.val_loader
        loss_fn = self.loss_fn

        model.eval()

        all_similarities = []
        all_labels = []
        total_loss = 0.0
        num_samples = 0

        for i, data in enumerate(val_loader):
            labels = data['label']
            if args.gpu is not None:
                labels = labels.cuda(args.gpu, non_blocking=True)

            with torch.cuda.amp.autocast(True):
                if 'embedding1' in data:
                    # Embedding mode (fast)
                    emb1 = data['embedding1'].cuda(args.gpu, non_blocking=True)
                    emb2 = data['embedding2'].cuda(args.gpu, non_blocking=True)
                    underlying = model.module if hasattr(model, 'module') else model
                    z1, z2, similarity = underlying.forward_from_embeddings(emb1, emb2)
                else:
                    # Image mode
                    image1 = data['image1'].cuda(args.gpu, non_blocking=True)
                    image2 = data['image2'].cuda(args.gpu, non_blocking=True)
                    z1, z2, similarity = model(image1, image2)
                loss = loss_fn(z1, z2, labels)

            all_similarities.append(similarity.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            total_loss += loss.item() * labels.size(0)
            num_samples += labels.size(0)

        # Concatenate results
        all_similarities = np.concatenate(all_similarities)
        all_labels = np.concatenate(all_labels)
        avg_loss = total_loss / num_samples

        # Compute metrics
        try:
            roc_auc = roc_auc_score(all_labels, all_similarities)
        except:
            roc_auc = 0.5

        try:
            avg_precision = average_precision_score(all_labels, all_similarities)
        except:
            avg_precision = 0.0

        # Accuracy at threshold 0
        predictions = (all_similarities > 0).astype(float)
        accuracy = (predictions == all_labels).mean()

        # Mean similarity by class
        pos_mask = all_labels == 1
        neg_mask = all_labels == 0
        mean_pos_sim = all_similarities[pos_mask].mean() if pos_mask.any() else 0.0
        mean_neg_sim = all_similarities[neg_mask].mean() if neg_mask.any() else 0.0

        metrics = {
            'loss': avg_loss,
            'roc_auc': roc_auc,
            'average_precision': avg_precision,
            'accuracy': accuracy,
            'mean_pos_similarity': mean_pos_sim,
            'mean_neg_similarity': mean_neg_sim,
        }

        # Print results
        print(f"\n==> Epoch {epoch:04d} Evaluation Results:")
        print(f"  Loss: {avg_loss:.04f}")
        print(f"  ROC-AUC: {roc_auc:.04f}")
        print(f"  Average Precision: {avg_precision:.04f}")
        print(f"  Accuracy: {accuracy:.04f}")
        print(f"  Mean Pos Similarity: {mean_pos_sim:.04f}")
        print(f"  Mean Neg Similarity: {mean_neg_sim:.04f}")
        print(f"  Best ROC-AUC so far: {self.best_roc_auc:.04f}")

        # Log to wandb
        if args.rank == 0:
            wandb.log({
                'val_loss': avg_loss,
                'val_roc_auc': roc_auc,
                'val_avg_precision': avg_precision,
                'val_accuracy': accuracy,
                'val_mean_pos_similarity': mean_pos_sim,
                'val_mean_neg_similarity': mean_neg_sim,
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
