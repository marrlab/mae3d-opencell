"""
WTC-11 localization trainer.

Subclasses LocalizationTrainer and overrides only build_dataloader()
to use WTCLocalizationDataset instead of OpenCellLocalizationDataset.

No localization_csv_path is needed — labels come from the built-in
WTC_PROTEIN_TO_LOCALIZATION mapping in the dataset.
"""

import os
import numpy as np
import pandas as pd
import torch
import wandb
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score

from lib.trainers.localization_trainer import LocalizationTrainer
from data.wtc.localization_dataset import WTCLocalizationDataset, WTC_LOCALIZATION_LABELS
from data.opencell.transforms import (
    get_opencell_train_transforms,
    get_opencell_val_transforms,
    get_opencell_2d_train_transforms,
    get_opencell_2d_val_transforms,
)


class LocalizationWTCTrainer(LocalizationTrainer):
    """LocalizationTrainer configured for WTC-11."""

    def build_dataloader(self):
        if self.train_loader is not None:
            raise ValueError("Dataloaders already built.")

        print("=> Creating WTC-11 localization dataloaders")
        args = self.args

        # ── Resolve embedding paths ───────────────────────────────────────────
        train_mae_emb_path = None
        val_mae_emb_path = None
        combined_emb = None
        combined_lookup = None

        if self.use_embedding_mode:
            print(f"   Mode: Embedding-only (fast training)")
            print(f"   MAE embedding path: {self.mae_embedding_path}")

            if self.mae_embedding_csv_path:
                # Combined lookup mode (train+val from extraction CSVs)
                print(f"   Building combined embedding lookup ...")
                all_embs = []
                combined_lookup = {}
                offset = 0
                for sname in ('train', 'val'):
                    npy = os.path.join(self.mae_embedding_path, f'{sname}.npy')
                    csv_s = os.path.join(self.mae_embedding_csv_path, f'{sname}.csv')
                    if os.path.exists(npy) and os.path.exists(csv_s):
                        arr = np.load(npy)
                        sdf = pd.read_csv(csv_s)
                        assert len(arr) == len(sdf)
                        for local_i, (_, row) in enumerate(sdf.iterrows()):
                            combined_lookup[row['image_path']] = offset + local_i
                        all_embs.append(arr)
                        offset += len(arr)
                        print(f"     {sname}: {len(arr)} embeddings")
                if all_embs:
                    combined_emb = np.concatenate(all_embs, axis=0)
                    print(f"   Combined: {combined_emb.shape}")
            else:
                # Simple positional mode — per-fold npy aligned with fold CSV
                train_mae_emb_path = os.path.join(self.mae_embedding_path, 'train.npy')
                val_mae_emb_path   = os.path.join(self.mae_embedding_path, 'val.npy')
        else:
            print(f"   Mode: Image loading")

        # ── Transforms (only used in image mode) ─────────────────────────────
        if self.use_embedding_mode:
            train_transform = val_transform = None
        elif self.use_2d:
            train_transform = get_opencell_2d_train_transforms(
                flip_prob=args.RandFlipd_prob,
                rotate_prob=args.RandRotate90d_prob,
            )
            val_transform = get_opencell_2d_val_transforms()
        else:
            train_transform = get_opencell_train_transforms(
                flip_prob=args.RandFlipd_prob,
                rotate_prob=args.RandRotate90d_prob,
            )
            val_transform = get_opencell_val_transforms()

        # ── Datasets ──────────────────────────────────────────────────────────
        def _make_dataset(split, mae_path, transform):
            return WTCLocalizationDataset(
                csv_path=os.path.join(args.csv_path, f'{split}.csv'),
                split=split,
                transform=transform,
                use_max_projection=self.use_2d,
                mae_embedding_path=mae_path,
                mae_embedding_array=combined_emb,
                mae_embedding_lookup_dict=combined_lookup,
            )

        train_dataset = _make_dataset('train', train_mae_emb_path, train_transform)
        val_dataset   = _make_dataset('val',   val_mae_emb_path,   val_transform)

        if train_dataset.is_embedding_mode():
            print(f"   MAE embedding dim: {train_dataset.get_mae_embedding_dim()}")

        print("\n==> Training set label distribution:")
        for label, count in sorted(train_dataset.get_label_distribution().items(),
                                   key=lambda x: -x[1]):
            print(f"  {label}: {count}")

        # ── Samplers ──────────────────────────────────────────────────────────
        if args.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                train_dataset, shuffle=True)
            val_sampler = torch.utils.data.distributed.DistributedSampler(
                val_dataset, shuffle=False)
        else:
            train_sampler = val_sampler = None

        # ── DataLoaders ───────────────────────────────────────────────────────
        self.train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=(train_sampler is None),
            num_workers=self.workers,
            pin_memory=True,
            sampler=train_sampler,
            drop_last=True,
        )
        self.val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.workers,
            pin_memory=True,
            sampler=val_sampler,
        )

        self.iters_per_epoch = len(self.train_loader)
        self.val_iters = len(self.val_loader)

        print(f"\n   Train: {len(train_dataset)} cells, {self.iters_per_epoch} iters/epoch")
        print(f"   Val:   {len(val_dataset)} cells, {self.val_iters} iters/epoch")

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

            with torch.cuda.amp.autocast(True):
                if 'mae_embedding' in data:
                    mae_emb = data['mae_embedding'].cuda(args.gpu, non_blocking=True)
                    underlying_model = model.module if hasattr(model, 'module') else model
                    logits = underlying_model.forward_from_embeddings(mae_emb)
                else:
                    image = data['image'].cuda(args.gpu, non_blocking=True)
                    logits = model(image)

                loss = loss_fn(logits, target)

            all_logits.append(logits.cpu())
            all_targets.append(target.cpu())
            total_loss += loss.item() * target.size(0)
            num_samples += target.size(0)

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

        per_class_ap = {}
        for i, label in enumerate(WTC_LOCALIZATION_LABELS):
            if binary_targets[:, i].sum() > 0:
                try:
                    ap = average_precision_score(binary_targets[:, i], all_probs[:, i])
                    per_class_ap[label] = ap
                except:
                    per_class_ap[label] = 0.0

        print(f"\n==> Epoch {epoch:04d} Evaluation Results:")
        print(f"  Loss: {avg_loss:.04f}")
        print(f"  mAP: {mAP:.04f}")
        print(f"  Macro AUC: {macro_auc:.04f}")
        print(f"  Macro F1: {macro_f1:.04f}")
        print(f"  Micro F1: {micro_f1:.04f}")
        print(f"\n  Per-class AP:")
        for label, ap in sorted(per_class_ap.items(), key=lambda x: -x[1]):
            print(f"    {label}: {ap:.04f}")

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
