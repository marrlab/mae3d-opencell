"""
Trainer for MAE3D with Channel Cross-Attention, FFT Loss, and
External Embedding Concatenation in Decoder.

Extends MAE3DChannelCrossAttentionFFTTrainer to:
- Load external embeddings (ESM2, SubCell) alongside images
- Pass embeddings to model forward for decoder concatenation
- Handle visualization with embeddings
- Support loading pretrained weights with strict=False
"""

import os
import torch
import numpy as np
import wandb

from .mae3d_cross_attention_fft_trainer import MAE3DChannelCrossAttentionFFTTrainer
from lib.models.mae3d_cross_attention_concat import MAE3DChannelCrossAttentionConcat
from data.opencell.dataset import OpenCellDataset
from data.opencell.transforms import get_opencell_train_transforms, get_opencell_val_transforms
from torch.utils.data import DistributedSampler


class MAE3DChannelCrossAttentionConcatTrainer(MAE3DChannelCrossAttentionFFTTrainer):
    """
    Trainer for MAE3D with Channel Cross-Attention, FFT Loss, and
    External Embedding Concatenation in Decoder.

    Key differences from MAE3DChannelCrossAttentionFFTTrainer:
    - Uses MAE3DChannelCrossAttentionConcat model
    - Loads ESM2 and/or SubCell embeddings from dataset
    - Passes embeddings to model forward
    - Handles loading pretrained MAE weights (without projection layers)
    """

    def __init__(self, args):
        super().__init__(args)
        self.model_name = 'MAE3DChannelCrossAttentionConcat'

        # Feature flags
        self.concat_esm2 = getattr(args, 'concat_esm2', False)
        self.concat_subcell = getattr(args, 'concat_subcell', False)

        print(f"\n=> Concatenation Configuration:")
        print(f"   Concat ESM2: {self.concat_esm2}")
        if self.concat_esm2:
            print(f"   ESM2 embed dim: {getattr(args, 'esm2_embed_dim', 1280)}")
            print(f"   Num ESM2 tokens: {getattr(args, 'num_esm2_tokens', 1)}")
        print(f"   Concat SubCell: {self.concat_subcell}")
        if self.concat_subcell:
            print(f"   SubCell embed dim: {getattr(args, 'subcell_embed_dim', 1536)}")
            print(f"   Num SubCell tokens: {getattr(args, 'num_subcell_tokens', 1)}")

    def build_model(self):
        """Build MAE3D model with cross-attention, FFT loss, and external concatenation."""
        if self.model is not None:
            raise ValueError("Model has been created. Do not create twice")

        args = self.args
        print(f"=> creating model {self.model_name}")
        print(f"   Cross-attention enabled: True")
        print(f"   FFT loss enabled: {getattr(args, 'use_fft_loss', True)}")
        print(f"   Concat ESM2: {self.concat_esm2}")
        print(f"   Concat SubCell: {self.concat_subcell}")
        print(f"   Channels: {args.in_chans}")
        print(f"   Encoder depth: {args.encoder_depth}")
        print(f"   Decoder depth: {args.decoder_depth}")
        print(f"   Encoder embed dim: {args.encoder_embed_dim}")
        print(f"   Decoder embed dim: {args.decoder_embed_dim}")

        # Create model
        self.model = MAE3DChannelCrossAttentionConcat(args=args)

        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"   Total parameters: {total_params:,}")
        print(f"   Trainable parameters: {trainable_params:,}")

        # Wrap with DDP
        self.wrap_model()

    def build_dataloader(self):
        """Build OpenCell dataloaders with external embeddings."""
        if self.train_loader is not None:
            raise ValueError("Dataloader has been created. Do not create twice.")

        print("=> creating dataloaders with external embeddings")
        args = self.args

        # Get embedding paths
        esm2_embedding_path = getattr(args, 'esm2_embedding_path', None) if self.concat_esm2 else None
        concat_embedding_path = getattr(args, 'concat_embedding_path', None) if self.concat_subcell else None

        # Construct paths for train/val splits
        train_esm2_path = os.path.join(esm2_embedding_path, "train.npy") if esm2_embedding_path else None
        val_esm2_path = os.path.join(esm2_embedding_path, "val.npy") if esm2_embedding_path else None
        train_concat_path = os.path.join(concat_embedding_path, "train.npy") if concat_embedding_path else None
        val_concat_path = os.path.join(concat_embedding_path, "val.npy") if concat_embedding_path else None

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
            esm2_embedding_path=train_esm2_path,
            concat_embedding_path=train_concat_path
        )

        # Display cache statistics (only rank 0)
        if args.cache_rate > 0 and self.rank == 0:
            cache_stats = train_dataset.get_cache_stats()
            print(f"\n  Train Dataset Cache Statistics:")
            print(f"    Total images: {cache_stats['total_images']}")
            print(f"    Cached images: {cache_stats['cached_images']}")
            print(f"    Cache rate: {cache_stats['cache_rate']:.2%}")

        # Display embedding info
        if self.rank == 0:
            if self.concat_esm2 and train_dataset.esm2_embeddings is not None:
                print(f"  ESM2 embedding dimension: {train_dataset.esm2_embeddings.shape[1]}")
            if self.concat_subcell and train_dataset.concat_embeddings is not None:
                print(f"  Concat (SubCell) embedding dimension: {train_dataset.concat_embeddings.shape[1]}")

        # Use 0 workers in distributed mode
        train_workers = 0 if self.is_distributed else args.workers

        # Create train dataloader
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

        print(f"   Dataset: OpenCell with external embedding concatenation")
        print(f"   Train samples: {len(train_dataset)}")
        print(f"   Batch size: {args.batch_size}")
        print(f"   Workers: {train_workers} (config: {args.workers})")
        print(f"   Iterations per epoch: {len(self.train_loader)}")

        # Create validation dataset
        val_csv_path = os.path.join(args.csv_path, "val.csv")
        val_has_embeddings = True
        if self.concat_esm2 and val_esm2_path and not os.path.exists(val_esm2_path):
            val_has_embeddings = False
        if self.concat_subcell and val_concat_path and not os.path.exists(val_concat_path):
            val_has_embeddings = False

        if os.path.exists(val_csv_path) and val_has_embeddings:
            val_transform = get_opencell_val_transforms()

            val_dataset = OpenCellDataset(
                csv_path=val_csv_path,
                split='val',
                transform=val_transform,
                cache_rate=0.0,
                num_workers=args.workers,
                z_slice_start=getattr(args, 'z_slice_start', None),
                z_slice_end=getattr(args, 'z_slice_end', None),
                esm2_embedding_path=val_esm2_path,
                concat_embedding_path=val_concat_path
            )

            val_workers = 0

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
            print(f"   Val workers: {val_workers}")
            print(f"   Validation enabled: True\n")
        else:
            print(f"   Validation file or embeddings not found")
            print(f"   Validation enabled: False\n")

        # Store fixed sample for visualization (only rank 0)
        if self.rank == 0:
            self._store_fixed_vis_sample()

    def _store_fixed_vis_sample(self):
        """Store a fixed sample for consistent visualization, including embeddings."""
        print("\n=> Storing fixed visualization sample...")

        for batch in self.train_loader:
            self.fixed_vis_sample = {
                'image': batch['image'][0:1].cuda(self.local_rank, non_blocking=True)
            }
            if self.concat_esm2 and 'esm2_embedding' in batch:
                self.fixed_vis_sample['esm2_embedding'] = batch['esm2_embedding'][0:1].cuda(
                    self.local_rank, non_blocking=True
                )
            if self.concat_subcell and 'concat_embedding' in batch:
                self.fixed_vis_sample['concat_embedding'] = batch['concat_embedding'][0:1].cuda(
                    self.local_rank, non_blocking=True
                )

            print(f"   Fixed sample image shape: {self.fixed_vis_sample['image'].shape}")
            if 'esm2_embedding' in self.fixed_vis_sample:
                print(f"   Fixed sample ESM2 shape: {self.fixed_vis_sample['esm2_embedding'].shape}")
            if 'concat_embedding' in self.fixed_vis_sample:
                print(f"   Fixed sample SubCell shape: {self.fixed_vis_sample['concat_embedding'].shape}")
            break

    def _get_embeddings_from_batch(self, batch):
        """Extract ESM2 and SubCell embeddings from batch dict."""
        esm2_emb = None
        subcell_emb = None

        if self.concat_esm2 and 'esm2_embedding' in batch:
            esm2_emb = batch['esm2_embedding'].cuda(self.local_rank, non_blocking=True)
        if self.concat_subcell and 'concat_embedding' in batch:
            subcell_emb = batch['concat_embedding'].cuda(self.local_rank, non_blocking=True)

        return esm2_emb, subcell_emb

    def _get_embeddings_from_vis_sample(self):
        """Extract embeddings from fixed visualization sample."""
        esm2_emb = self.fixed_vis_sample.get('esm2_embedding', None)
        subcell_emb = self.fixed_vis_sample.get('concat_embedding', None)
        return esm2_emb, subcell_emb

    def epoch_train(self, epoch):
        """Training logic for one epoch with external embedding concatenation."""
        args = self.args
        model = self.wrapped_model
        optimizer = self.optimizer
        scaler = self.scaler

        model.train()

        # Calculate warmup settings
        warmup_steps = args.warmup_epochs * len(self.train_loader)
        warmup_start_lr = self.lr * 0.01

        steps_per_epoch = len(self.train_loader)

        for i, batch in enumerate(self.train_loader):
            images = batch['image'].cuda(self.local_rank, non_blocking=True)
            esm2_emb, subcell_emb = self._get_embeddings_from_batch(batch)

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
                loss, loss_mse, loss_fft = model(
                    images,
                    esm2_emb=esm2_emb,
                    subcell_emb=subcell_emb,
                    return_image=False
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

            # Generate visualization on fixed sample
            vis_outputs = None
            if should_visualize:
                with torch.no_grad():
                    with torch.cuda.amp.autocast(True):
                        vis_esm2, vis_subcell = self._get_embeddings_from_vis_sample()
                        _, _, _, original_patches, recon_patches, masked_patches = model(
                            self.fixed_vis_sample['image'],
                            esm2_emb=vis_esm2,
                            subcell_emb=vis_subcell,
                            return_image=True
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

                # Print progress with FFT info
                if i % args.print_freq == 0:
                    print(f"Epoch {epoch}/{args.epochs} | "
                          f"Iter {i}/{len(self.train_loader)} | "
                          f"Step {self.global_step} | "
                          f"Loss: {loss.item() * self.gradient_accumulation_steps:.4f} | "
                          f"MSE: {loss_mse.item():.4f} | "
                          f"FFT: {loss_fft.item():.4f} | "
                          f"FFT_w: {current_fft_weight:.3f}", flush=True)

            self.global_step += 1

    def validate_epoch(self, epoch):
        """Validation logic with separate MSE and FFT loss reporting."""
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
                esm2_emb, subcell_emb = self._get_embeddings_from_batch(batch)

                with torch.cuda.amp.autocast(True):
                    loss, loss_mse, loss_fft = model(
                        images,
                        esm2_emb=esm2_emb,
                        subcell_emb=subcell_emb,
                        return_image=False
                    )

                val_losses.append(loss.item())
                val_mse_losses.append(loss_mse.item())
                val_fft_losses.append(loss_fft.item())

        # Average losses
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

    def resume(self):
        """
        Load pretrained weights with strict=False.

        Handles the case where the checkpoint doesn't have external embedding
        projection layers (loading from a non-concat checkpoint for fine-tuning).
        """
        args = self.args

        if args.resume is None:
            return None

        if not os.path.isfile(args.resume):
            print(f"=> no checkpoint found at '{args.resume}'")
            return None

        print(f"=> loading pretrained checkpoint '{args.resume}'")

        if torch.cuda.is_available():
            loc = f'cuda:{self.local_rank}'
            checkpoint = torch.load(args.resume, map_location=loc)
        else:
            checkpoint = torch.load(args.resume)

        # Load model state with strict=False (allows missing projection keys)
        state_dict = checkpoint['state_dict']

        if self.is_distributed:
            msg = self.wrapped_model.module.load_state_dict(state_dict, strict=False)
        else:
            msg = self.model.load_state_dict(state_dict, strict=False)

        print(f"=> loaded pretrained weights (strict=False)")
        print(f"   Missing keys: {msg.missing_keys}")
        print(f"   Unexpected keys: {msg.unexpected_keys[:5] if len(msg.unexpected_keys) > 5 else msg.unexpected_keys}")

        # NOTE: We intentionally do NOT load optimizer state or set start_epoch
        # because we're fine-tuning from a different model (MAE+FFT -> MAE+FFT+Concat)
        print(f"=> pretrained checkpoint loaded (epoch {checkpoint.get('epoch', 'N/A')} in original training)")
        print(f"=> fine-tuning will start from epoch 0")

        return checkpoint
