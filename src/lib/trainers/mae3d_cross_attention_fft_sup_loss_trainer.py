"""
Trainer for MAE3D with Channel Cross-Attention, FFT Loss, and Supervised Classification Loss.

Extends the FFT trainer with:
- Protein classification loss on encoder features (train only)
- Dataset returns protein_label for supervised training
- Validation only reports MSE + FFT (val proteins unseen during train)
"""

import os
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import wandb

from .mae3d_cross_attention_fft_trainer import MAE3DChannelCrossAttentionFFTTrainer
from lib.models.mae3d_cross_attention_fft_sup_loss import MAE3DChannelCrossAttentionFFTSupLoss


class MAE3DChannelCrossAttentionFFTSupLossTrainer(MAE3DChannelCrossAttentionFFTTrainer):
    """
    Trainer for MAE3D with Channel Cross-Attention, FFT Loss,
    and Supervised Protein Classification Loss.

    The supervised loss is only applied during training since validation
    proteins have zero overlap with training proteins.
    """

    def __init__(self, args):
        super().__init__(args)
        self.model_name = 'MAE3DChannelCrossAttentionFFTSupLoss'
        self.sup_loss_weight = getattr(args, 'sup_loss_weight', 0.1)

        print(f"=> Supervised Loss Settings:")
        print(f"   Sup loss weight: {self.sup_loss_weight}")
        print(f"   Num classes: {getattr(args, 'num_classes', 1048)}")

    def build_model(self):
        """Build MAE3D model with channel cross-attention, FFT loss, and supervised loss."""
        if self.model is not None:
            raise ValueError("Model has been created. Do not create twice")

        args = self.args
        print(f"=> creating model {self.model_name}")
        print(f"   Cross-attention enabled: True")
        print(f"   FFT loss enabled: True")
        print(f"   Supervised loss enabled: True")
        print(f"   Channels: {args.in_chans}")
        print(f"   Encoder depth: {args.encoder_depth}")
        print(f"   Decoder depth: {args.decoder_depth}")
        print(f"   Encoder embed dim: {args.encoder_embed_dim}")
        print(f"   Decoder embed dim: {args.decoder_embed_dim}")

        self.model = MAE3DChannelCrossAttentionFFTSupLoss(args=args)

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"   Total parameters: {total_params:,}")
        print(f"   Trainable parameters: {trainable_params:,}")

        self.wrap_model()

    def build_dataloader(self):
        """
        Build dataloaders with protein label support for supervised training.
        """
        if self.train_loader is not None:
            raise ValueError("Dataloader has been created. Do not create twice.")

        print("=> creating dataloaders")
        args = self.args

        from data.opencell.transforms import get_opencell_train_transforms, get_opencell_val_transforms
        from data.opencell.dataset import OpenCellDataset
        from torch.utils.data.distributed import DistributedSampler

        return_protein_label = getattr(args, 'return_protein_label', True)

        # Create train dataset
        train_csv_path = os.path.join(args.csv_path, "train.csv")
        train_transform = get_opencell_train_transforms(
            flip_prob=args.RandFlipd_prob,
            rotate_prob=args.RandRotate90d_prob
        )

        train_dataset = OpenCellDataset(
            csv_path=train_csv_path,
            split='train',
            transform=train_transform,
            cache_rate=args.cache_rate,
            num_workers=args.workers,
            z_slice_start=getattr(args, 'z_slice_start', None),
            z_slice_end=getattr(args, 'z_slice_end', None),
            return_protein_label=return_protein_label,
        )

        if return_protein_label:
            print(f"   Protein labels enabled: {train_dataset.num_classes} classes")

        # Display cache statistics
        if args.cache_rate > 0 and self.rank == 0:
            cache_stats = train_dataset.get_cache_stats()
            print(f"\n  Train Dataset Cache Statistics:")
            print(f"    Total images: {cache_stats['total_images']}")
            print(f"    Cached images: {cache_stats['cached_images']}")
            print(f"    Cache rate: {cache_stats['cache_rate']:.2%}")
            print(f"    Cache hit rate: {cache_stats['cache_hit_rate']:.2%}\n")

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

        print(f"   Dataset: OpenCell")
        print(f"   Train samples: {len(train_dataset)}")
        print(f"   Batch size: {args.batch_size}")
        print(f"   Workers: {train_workers} (config: {args.workers})")
        print(f"   Iterations per epoch: {len(self.train_loader)}")

        # Create validation dataset (no protein labels needed)
        val_csv_path = os.path.join(args.csv_path, "val.csv")
        if os.path.exists(val_csv_path):
            val_transform = get_opencell_val_transforms()

            val_dataset = OpenCellDataset(
                csv_path=val_csv_path,
                split='val',
                transform=val_transform,
                cache_rate=0.0,
                num_workers=args.workers,
                z_slice_start=getattr(args, 'z_slice_start', None),
                z_slice_end=getattr(args, 'z_slice_end', None),
                return_protein_label=False,  # Val proteins are unseen
            )

            val_workers = 0 if self.is_distributed else args.workers

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
                    num_workers=val_workers,
                    pin_memory=True
                )
            else:
                self.val_loader = val_dataset.get_dataloader(
                    batch_size=args.batch_size,
                    shuffle=False,
                    num_workers=val_workers
                )

            print(f"   Val samples: {len(val_dataset)}")
        else:
            self.val_loader = None
            print("   No validation set found.")

        # Store a fixed visualization sample
        vis_batch_size = getattr(args, 'vis_batch_size', None)
        if vis_batch_size is None or str(vis_batch_size) == 'None':
            vis_batch_size = args.batch_size
        vis_batch_size = int(vis_batch_size)
        self.fixed_vis_sample = None
        if self.rank == 0:
            for batch in self.train_loader:
                self.fixed_vis_sample = batch['image'][:vis_batch_size].cuda(self.local_rank)
                break

    def epoch_train(self, epoch):
        """Training logic for one epoch with FFT + supervised loss."""
        args = self.args
        model = self.wrapped_model
        optimizer = self.optimizer
        scaler = self.scaler

        model.train()

        warmup_steps = args.warmup_epochs * len(self.train_loader)
        warmup_start_lr = self.lr * 0.01
        steps_per_epoch = len(self.train_loader)

        correct = 0
        total = 0

        for i, batch in enumerate(self.train_loader):
            images = batch['image'].cuda(self.local_rank, non_blocking=True)
            protein_labels = batch.get('protein_label', None)
            if protein_labels is not None:
                protein_labels = protein_labels.cuda(self.local_rank, non_blocking=True)

            # Per-step LR adjustment during warmup
            if self.global_step < warmup_steps:
                progress = min(self.global_step / warmup_steps, 1.0)
                current_lr = warmup_start_lr + (self.lr - warmup_start_lr) * progress
                for param_group in optimizer.param_groups:
                    param_group['lr'] = current_lr

            # Update FFT weight based on schedule
            current_fft_weight = self._get_fft_weight_for_step(epoch, i, steps_per_epoch)
            if hasattr(model, 'module'):
                model.module.set_fft_weight(current_fft_weight)
            else:
                model.set_fft_weight(current_fft_weight)

            # Determine if we should visualize this step
            should_visualize = (
                (self.global_step % args.vis_freq == 0) and
                (self.rank == 0) and
                (self.fixed_vis_sample is not None)
            )

            # Zero gradients at the start of accumulation
            if self.accum_iter == 0:
                optimizer.zero_grad()

            # Forward pass with mixed precision
            with torch.cuda.amp.autocast(True):
                loss, loss_mse, loss_fft, loss_sup = model(
                    images, return_image=False, protein_labels=protein_labels
                )
                loss = loss / self.gradient_accumulation_steps

            # Backward pass
            scaler.scale(loss).backward()

            # Increment accumulation counter
            self.accum_iter += 1

            # Only update weights after accumulating enough gradients
            if self.accum_iter >= self.gradient_accumulation_steps:
                if hasattr(args, 'grad_clip') and args.grad_clip is not None and args.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

                scaler.step(optimizer)
                scaler.update()
                self.accum_iter = 0

            # Track accuracy
            if protein_labels is not None:
                with torch.no_grad():
                    # Re-compute logits for accuracy (use model's cls_head)
                    # We can get this from the loss_sup being non-zero
                    inner_model = model.module if hasattr(model, 'module') else model
                    # Use a simple forward through cls_head on cached features
                    # Instead, track from the forward pass output
                    # For efficiency, just track the loss and compute accuracy periodically
                    pass

            # Generate visualization on fixed sample
            vis_outputs = None
            if should_visualize:
                with torch.no_grad():
                    with torch.cuda.amp.autocast(True):
                        _, _, _, _, original_patches, recon_patches, masked_patches = model(
                            self.fixed_vis_sample, return_image=True
                        )
                vis_outputs = (original_patches, recon_patches, masked_patches)

            # Log metrics
            if self.rank == 0:
                log_dict = {
                    "loss": loss.item() * self.gradient_accumulation_steps,
                    "loss_mse": loss_mse.item(),
                    "loss_fft": loss_fft.item(),
                    "loss_fft_weighted": (current_fft_weight * loss_fft).item(),
                    "fft_weight": current_fft_weight,
                    "loss_sup": loss_sup.item(),
                    "loss_sup_weighted": (self.sup_loss_weight * loss_sup).item(),
                    "sup_weight": self.sup_loss_weight,
                    "epoch": epoch,
                    "step": self.global_step,
                    "lr": optimizer.param_groups[0]['lr']
                }

                # Add GPU metrics
                gpu_metrics = self.get_gpu_metrics()
                log_dict.update(gpu_metrics)

                # Add visualization if needed
                if should_visualize and vis_outputs is not None:
                    try:
                        print(f"  Generating visualization at step {self.global_step}...", flush=True)
                        original_patches, recon_patches, masked_patches = vis_outputs
                        vis_z_slices = getattr(args, 'vis_z_slices', None)
                        if vis_z_slices is None:
                            z_dim = args.input_size[0]
                            vis_z_slices = [z_dim // 5, z_dim // 2, z_dim * 4 // 5]
                        vis_image = self.visualize_mae_reconstruction(
                            original_patches=original_patches,
                            masked_patches=masked_patches,
                            recon_patches=recon_patches,
                            z_slices=vis_z_slices
                        )
                        log_dict["reconstruction"] = vis_image
                        print(f"  Visualization logged successfully!", flush=True)
                    except Exception as e:
                        print(f"  WARNING: Visualization failed: {e}", flush=True)

                wandb.log(log_dict)

                # Print progress
                if i % args.print_freq == 0:
                    print(f"Epoch {epoch}/{args.epochs} | "
                          f"Iter {i}/{len(self.train_loader)} | "
                          f"Step {self.global_step} | "
                          f"Loss: {loss.item() * self.gradient_accumulation_steps:.4f} | "
                          f"MSE: {loss_mse.item():.4f} | "
                          f"FFT: {loss_fft.item():.4f} | "
                          f"Sup: {loss_sup.item():.4f} | "
                          f"FFT_w: {current_fft_weight:.3f}", flush=True)

            self.global_step += 1

    def validate_epoch(self, epoch):
        """
        Validation logic - only MSE + FFT losses (val proteins are unseen).
        """
        if self.val_loader is None:
            return None

        args = self.args
        model = self.wrapped_model

        model.eval()
        val_losses = []
        val_mse_losses = []
        val_fft_losses = []

        # Use full FFT weight for validation
        if hasattr(model, 'module'):
            model.module.set_fft_weight(self.fft_loss_weight)
        else:
            model.set_fft_weight(self.fft_loss_weight)

        print(f"\n=> Running validation...", flush=True)

        with torch.no_grad():
            for i, batch in enumerate(self.val_loader):
                images = batch['image'].cuda(self.local_rank, non_blocking=True)

                with torch.cuda.amp.autocast(True):
                    # No protein_labels -> loss_sup = 0
                    loss, loss_mse, loss_fft, loss_sup = model(images, return_image=False)

                # loss already includes 0 sup loss; recompute MSE+FFT only
                val_loss = loss_mse.item() + self.fft_loss_weight * loss_fft.item()
                val_losses.append(val_loss)
                val_mse_losses.append(loss_mse.item())
                val_fft_losses.append(loss_fft.item())

        avg_val_loss = np.mean(val_losses)
        avg_val_mse = np.mean(val_mse_losses)
        avg_val_fft = np.mean(val_fft_losses)

        # Sync across GPUs if distributed
        if args.distributed:
            val_loss_tensor = torch.tensor([avg_val_loss, avg_val_mse, avg_val_fft],
                                           device=f'cuda:{self.local_rank}')
            torch.distributed.all_reduce(val_loss_tensor)
            val_loss_tensor /= args.world_size
            avg_val_loss, avg_val_mse, avg_val_fft = val_loss_tensor.tolist()

        if self.rank == 0:
            print(f"=> Validation Loss: {avg_val_loss:.4f} | "
                  f"MSE: {avg_val_mse:.4f} | FFT: {avg_val_fft:.4f}", flush=True)
            wandb.log({
                "val_loss": avg_val_loss,
                "val_loss_mse": avg_val_mse,
                "val_loss_fft": avg_val_fft,
                "epoch": epoch
            })

        return avg_val_loss
