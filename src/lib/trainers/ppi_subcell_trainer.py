"""
Trainer for PPI prediction using SubCell embeddings via metric learning.

This trainer uses precomputed SubCell embeddings, so training is fast
(no encoder forward pass required).
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

from lib.trainers.base_trainer import BaseTrainer
from lib.models.ppi_metric_subcell import PPIMetricSubCell
from data.opencell.subcell_ppi_dataset import SubCellPPIDataset


class ContrastiveLoss(nn.Module):
    """Contrastive loss for metric learning."""

    def __init__(self, margin=0.5):
        super().__init__()
        self.margin = margin

    def forward(self, z1, z2, labels):
        cos_sim = (z1 * z2).sum(dim=-1)
        distance = 1 - cos_sim
        pos_loss = labels * distance.pow(2)
        neg_loss = (1 - labels) * F.relu(self.margin - distance).pow(2)
        return (pos_loss + neg_loss).mean()


class CosineContrastiveLoss(nn.Module):
    """Cosine-based contrastive loss."""

    def __init__(self, margin=0.3, temperature=0.1):
        super().__init__()
        self.margin = margin
        self.temperature = temperature

    def forward(self, z1, z2, labels):
        cos_sim = (z1 * z2).sum(dim=-1)
        pos_loss = labels * (1 - cos_sim)
        neg_loss = (1 - labels) * F.relu(cos_sim - (-1 + self.margin))
        return (pos_loss + neg_loss).mean()


class BCEWithSimilarityLoss(nn.Module):
    """BCE loss using similarity as logits."""

    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, z1, z2, labels):
        cos_sim = (z1 * z2).sum(dim=-1)
        logits = cos_sim / self.temperature
        return self.bce(logits, labels)


class PPISubCellTrainer(BaseTrainer):
    """Trainer for PPI prediction using SubCell embeddings."""

    def __init__(self, args):
        super().__init__(args)
        self.model_name = args.arch

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

            self.model = PPIMetricSubCell(
                embed_dim=args.embed_dim,
                proj_hidden_dim=getattr(args, 'proj_hidden_dim', 512),
                proj_output_dim=getattr(args, 'proj_output_dim', 128),
                proj_num_layers=getattr(args, 'proj_num_layers', 2),
            )

            print(f"   Embedding dimension: {args.embed_dim}")
            print(f"   Projection: {args.embed_dim} -> {args.proj_hidden_dim} -> {args.proj_output_dim}")
            print(f"   Total parameters: {self.model.get_num_params():,}")

            self.wrap_model()
        else:
            raise ValueError("=> Model has been created. Do not create twice")

    def build_optimizer(self):
        assert self.model is not None, "Model is not created yet."
        print("=> Creating optimizer")
        args = self.args

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
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

            # Create training dataset
            train_dataset = SubCellPPIDataset(
                embedding_path=os.path.join(args.embedding_dir, 'train.npy'),
                csv_path=os.path.join(args.csv_path, 'train.csv'),
                ppi_csv_path=args.ppi_csv_path,
                abundance_csv_path=getattr(args, 'abundance_csv_path', None),
                split='train',
                pval_threshold=getattr(args, 'pval_threshold', 5.0),
                enrichment_threshold=getattr(args, 'enrichment_threshold', 2.5),
                stoichiometry_threshold=getattr(args, 'stoichiometry_threshold', 0.05),
                n_abundance_buckets=getattr(args, 'n_abundance_buckets', 10),
                n_negatives_per_positive=getattr(args, 'n_negatives_per_positive', 1),
                seed=args.seed
            )

            # Create validation dataset
            val_dataset = SubCellPPIDataset(
                embedding_path=os.path.join(args.embedding_dir, 'val.npy'),
                csv_path=os.path.join(args.csv_path, 'val.csv'),
                ppi_csv_path=args.ppi_csv_path,
                abundance_csv_path=getattr(args, 'abundance_csv_path', None),
                split='val',
                pval_threshold=getattr(args, 'pval_threshold', 5.0),
                enrichment_threshold=getattr(args, 'enrichment_threshold', 2.5),
                stoichiometry_threshold=getattr(args, 'stoichiometry_threshold', 0.05),
                n_abundance_buckets=getattr(args, 'n_abundance_buckets', 10),
                n_negatives_per_positive=getattr(args, 'n_negatives_per_positive', 1),
                seed=args.seed + 1
            )

            # Print statistics
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
                        self.save_checkpoint(epoch, is_best=True)

            # Save checkpoint periodically
            if epoch == 0 or (epoch + 1) % args.save_freq == 0:
                self.save_checkpoint(epoch)

    def epoch_train(self, epoch, niters):
        args = self.args
        train_loader = self.train_loader
        model = self.wrapped_model
        optimizer = self.optimizer
        loss_fn = self.loss_fn

        model.train()

        for i, data in enumerate(train_loader):
            # Adjust learning rate
            self.adjust_learning_rate(epoch + i / self.iters_per_epoch, args)

            # Get data
            emb1 = data['embedding1']
            emb2 = data['embedding2']
            labels = data['label']

            if args.gpu is not None:
                emb1 = emb1.cuda(args.gpu, non_blocking=True)
                emb2 = emb2.cuda(args.gpu, non_blocking=True)
                labels = labels.cuda(args.gpu, non_blocking=True)

            # Zero gradients
            if self.accum_iter == 0:
                optimizer.zero_grad()

            # Forward pass
            z1, z2, similarity = model(emb1, emb2)
            loss = loss_fn(z1, z2, labels)
            loss = loss / self.gradient_accumulation_steps

            # Backward pass
            loss.backward()

            self.accum_iter += 1

            if self.accum_iter >= self.gradient_accumulation_steps:
                optimizer.step()
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

        for data in val_loader:
            emb1 = data['embedding1']
            emb2 = data['embedding2']
            labels = data['label']

            if args.gpu is not None:
                emb1 = emb1.cuda(args.gpu, non_blocking=True)
                emb2 = emb2.cuda(args.gpu, non_blocking=True)
                labels = labels.cuda(args.gpu, non_blocking=True)

            z1, z2, similarity = model(emb1, emb2)
            loss = loss_fn(z1, z2, labels)

            all_similarities.append(similarity.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            total_loss += loss.item() * emb1.size(0)
            num_samples += emb1.size(0)

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
