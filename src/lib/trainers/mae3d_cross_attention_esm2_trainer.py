"""
Trainer for MAE3D with Channel Cross-Attention, ESM2 Conditioning, and Distillation.

Extends MAE3DChannelCrossAttentionTrainer with:
- Optional ESM2 protein embedding conditioning
- Optional SubCell distillation

Both features are independently toggleable via config flags.
"""

import os
import torch
import wandb

from .mae3d_cross_attention_trainer import MAE3DChannelCrossAttentionTrainer
from lib.models.mae3d_cross_attention_esm2 import MAE3DChannelCrossAttentionESM2
from data.opencell.dataset import OpenCellDataset
from data.opencell.transforms import get_opencell_train_transforms, get_opencell_val_transforms
from torch.utils.data import DistributedSampler


class MAE3DChannelCrossAttentionESM2Trainer(MAE3DChannelCrossAttentionTrainer):
    """
    Trainer for MAE3D with Channel Cross-Attention, ESM2 Conditioning, and Distillation.

    Supports:
    - ESM2 protein embedding conditioning (optional)
    - SubCell distillation (optional)

    When neither is enabled, behaves identically to MAE3DChannelCrossAttentionTrainer.
    """

    def __init__(self, args):
        super().__init__(args)
        self.model_name = 'MAE3DChannelCrossAttentionESM2'

        # Feature flags
        self.use_esm2_conditioning = getattr(args, 'use_esm2_conditioning', False)
        self.use_subcell_distillation = getattr(args, 'use_subcell_distillation', False)

        # Decoder freeze state (shared by warmup and distillation ramp-up)
        self.decoder_frozen = False
        self.freeze_decoder_during_warmup = getattr(args, 'freeze_decoder_during_warmup', False)

        # Distillation configuration (if enabled)
        if self.use_subcell_distillation:
            self.distill_weight = getattr(args, 'distill_weight', 1.0)
            self.distill_rampup_epochs = getattr(args, 'distill_rampup_epochs', 1)
            self.freeze_decoder_during_rampup = getattr(args, 'freeze_decoder_during_rampup', True)

        print(f"\n=> Feature Configuration:")
        print(f"   ESM2 conditioning: {self.use_esm2_conditioning}")
        print(f"   Freeze decoder during warmup: {self.freeze_decoder_during_warmup}")
        print(f"   SubCell distillation: {self.use_subcell_distillation}")
        if self.use_subcell_distillation:
            print(f"   Distillation weight: {self.distill_weight}")
            print(f"   Ramp-up epochs: {self.distill_rampup_epochs}")
            print(f"   Freeze decoder during ramp-up: {self.freeze_decoder_during_rampup}")

    def _get_decoder_params(self):
        """Get all decoder-related parameters for freezing."""
        model = self.model if not self.is_distributed else self.wrapped_model.module
        decoder_params = []

        decoder_modules = [
            'encoder_to_decoder',
            'mask_tokens',
            'decoder_blocks',
            'decoder_norms',
            'decoder_heads',
            'decoder_pos_embed',
        ]

        for name, param in model.named_parameters():
            for decoder_module in decoder_modules:
                if decoder_module in name:
                    decoder_params.append(param)
                    break

        return decoder_params

    def freeze_decoder(self):
        """Freeze decoder parameters."""
        if self.decoder_frozen:
            return

        decoder_params = self._get_decoder_params()
        for param in decoder_params:
            param.requires_grad = False

        self.decoder_frozen = True
        if self.rank == 0:
            print(f"   Decoder frozen ({len(decoder_params)} parameters)")

    def unfreeze_decoder(self):
        """Unfreeze decoder parameters."""
        if not self.decoder_frozen:
            return

        decoder_params = self._get_decoder_params()
        for param in decoder_params:
            param.requires_grad = True

        self.decoder_frozen = False
        if self.rank == 0:
            print(f"   Decoder unfrozen ({len(decoder_params)} parameters)")

    def build_model(self):
        """Build MAE3D model with ESM2 conditioning and distillation support."""
        if self.model is not None:
            raise ValueError("Model has been created. Do not create twice")

        args = self.args
        print(f"=> creating model {self.model_name}")
        print(f"   Channels: {args.in_chans}")
        print(f"   Encoder depth: {args.encoder_depth}")
        print(f"   Decoder depth: {args.decoder_depth}")
        print(f"   Encoder embed dim: {args.encoder_embed_dim}")
        print(f"   Decoder embed dim: {args.decoder_embed_dim}")
        if self.use_esm2_conditioning:
            print(f"   ESM2 embed dim: {getattr(args, 'esm2_embed_dim', 1280)}")
        if self.use_subcell_distillation:
            print(f"   Teacher embed dim: {getattr(args, 'teacher_embed_dim', 1536)}")
            print(f"   Distill loss type: {getattr(args, 'distill_loss_type', 'cosine')}")

        # Create model
        self.model = MAE3DChannelCrossAttentionESM2(args=args)

        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"   Total parameters: {total_params:,}")
        print(f"   Trainable parameters: {trainable_params:,}")

        # Wrap with DDP
        self.wrap_model()

    def build_dataloader(self):
        """Build OpenCell dataloaders with optional ESM2 and SubCell embeddings."""
        if self.train_loader is not None:
            raise ValueError("Dataloader has been created. Do not create twice.")

        print("=> creating dataloaders")
        args = self.args

        # Get embedding paths
        esm2_embedding_path = getattr(args, 'esm2_embedding_path', None) if self.use_esm2_conditioning else None
        subcell_embedding_path = getattr(args, 'embedding_path', None) if self.use_subcell_distillation else None

        # Construct paths for train/val splits
        train_esm2_path = os.path.join(esm2_embedding_path, "train.npy") if esm2_embedding_path else None
        val_esm2_path = os.path.join(esm2_embedding_path, "val.npy") if esm2_embedding_path else None
        train_subcell_path = os.path.join(subcell_embedding_path, "train.npy") if subcell_embedding_path else None
        val_subcell_path = os.path.join(subcell_embedding_path, "val.npy") if subcell_embedding_path else None

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
            embedding_path=train_subcell_path,
            esm2_embedding_path=train_esm2_path
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
            if self.use_subcell_distillation:
                embed_dim = train_dataset.get_embedding_dim()
                print(f"  Teacher (SubCell) embedding dimension: {embed_dim}")
            if self.use_esm2_conditioning and train_dataset.esm2_embeddings is not None:
                print(f"  ESM2 embedding dimension: {train_dataset.esm2_embeddings.shape[1]}")

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

        print(f"   Dataset: OpenCell")
        print(f"   Train samples: {len(train_dataset)}")
        print(f"   Batch size: {args.batch_size}")
        print(f"   Workers: {train_workers}")
        print(f"   Iterations per epoch: {len(self.train_loader)}")

        # Create validation dataset
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
                embedding_path=val_subcell_path,
                esm2_embedding_path=val_esm2_path
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
            print(f"   Validation enabled: True\n")
        else:
            print(f"   Validation file not found")
            print(f"   Validation enabled: False\n")

        # Store fixed sample for visualization (only rank 0)
        if self.rank == 0:
            self._store_fixed_vis_sample()

    def _store_fixed_vis_sample(self):
        """Store a fixed sample for consistent visualization."""
        print("\n=> Storing fixed visualization sample...")

        for batch in self.train_loader:
            self.fixed_vis_sample = {
                'image': batch['image'][0:1].cuda(self.local_rank, non_blocking=True)
            }
            if self.use_esm2_conditioning and 'esm2_embedding' in batch:
                self.fixed_vis_sample['esm2_embedding'] = batch['esm2_embedding'][0:1].cuda(
                    self.local_rank, non_blocking=True
                )
            if self.use_subcell_distillation and 'teacher_embedding' in batch:
                self.fixed_vis_sample['teacher_embedding'] = batch['teacher_embedding'][0:1].cuda(
                    self.local_rank, non_blocking=True
                )
            print(f"   Fixed sample image shape: {self.fixed_vis_sample['image'].shape}")
            if 'esm2_embedding' in self.fixed_vis_sample:
                print(f"   Fixed sample ESM2 shape: {self.fixed_vis_sample['esm2_embedding'].shape}")
            if 'teacher_embedding' in self.fixed_vis_sample:
                print(f"   Fixed sample SubCell shape: {self.fixed_vis_sample['teacher_embedding'].shape}")
            break

    def get_distill_weight(self, epoch, step_in_epoch, total_steps_in_epoch):
        """Compute distillation weight with ramp-up."""
        if not self.use_subcell_distillation:
            return 0.0

        if self.distill_rampup_epochs <= 0:
            return self.distill_weight

        if epoch >= self.distill_rampup_epochs:
            return self.distill_weight

        # Linear ramp-up
        total_rampup_steps = self.distill_rampup_epochs * total_steps_in_epoch
        current_step = epoch * total_steps_in_epoch + step_in_epoch
        progress = min(current_step / total_rampup_steps, 1.0)

        return self.distill_weight * progress

    def epoch_train(self, epoch):
        """Training logic for one epoch with ESM2 and distillation support."""
        args = self.args
        model = self.wrapped_model
        optimizer = self.optimizer
        scaler = self.scaler

        # Freeze decoder during warmup (for ESM2 fine-tuning)
        if self.freeze_decoder_during_warmup:
            if epoch < args.warmup_epochs:
                if not self.decoder_frozen:
                    self.freeze_decoder()
            else:
                if self.decoder_frozen and not (self.use_subcell_distillation
                        and self.freeze_decoder_during_rampup
                        and epoch < self.distill_rampup_epochs):
                    self.unfreeze_decoder()

        # Handle decoder freezing during ramp-up (only for distillation)
        if self.use_subcell_distillation and self.freeze_decoder_during_rampup:
            if epoch < self.distill_rampup_epochs:
                if not self.decoder_frozen:
                    self.freeze_decoder()
            else:
                if self.decoder_frozen:
                    self.unfreeze_decoder()

        model.train()

        # Calculate warmup settings
        warmup_steps = args.warmup_epochs * len(self.train_loader)
        warmup_start_lr = self.lr * 0.01

        total_steps_in_epoch = len(self.train_loader)

        for i, batch in enumerate(self.train_loader):
            images = batch['image'].cuda(self.local_rank, non_blocking=True)

            # Get optional embeddings
            esm2_emb = None
            teacher_emb = None
            if self.use_esm2_conditioning and 'esm2_embedding' in batch:
                esm2_emb = batch['esm2_embedding'].cuda(self.local_rank, non_blocking=True)
            if self.use_subcell_distillation and 'teacher_embedding' in batch:
                teacher_emb = batch['teacher_embedding'].cuda(self.local_rank, non_blocking=True)

            # Per-step LR adjustment during warmup
            if self.global_step < warmup_steps:
                progress = min(self.global_step / warmup_steps, 1.0)
                current_lr = warmup_start_lr + (self.lr - warmup_start_lr) * progress
                for param_group in optimizer.param_groups:
                    param_group['lr'] = current_lr

            # Get current distillation weight
            current_distill_weight = self.get_distill_weight(epoch, i, total_steps_in_epoch)

            # Determine if we should visualize
            should_visualize = (self.global_step % args.vis_freq == 0) and (self.rank == 0) and (self.fixed_vis_sample is not None)

            # Zero gradients at the start of accumulation
            if self.accum_iter == 0:
                optimizer.zero_grad()

            # Forward pass
            with torch.cuda.amp.autocast(True):
                if self.use_subcell_distillation and teacher_emb is not None:
                    recon_loss, distill_loss = model(
                        images,
                        esm2_emb=esm2_emb,
                        teacher_emb=teacher_emb,
                        return_image=False,
                        return_distill_loss=True
                    )
                    total_loss = recon_loss + current_distill_weight * distill_loss
                else:
                    recon_loss = model(
                        images,
                        esm2_emb=esm2_emb,
                        return_image=False
                    )
                    distill_loss = torch.tensor(0.0, device=images.device)
                    total_loss = recon_loss

                # Scale loss for gradient accumulation
                total_loss = total_loss / self.gradient_accumulation_steps

            # Backward pass
            scaler.scale(total_loss).backward()

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
                        vis_esm2 = self.fixed_vis_sample.get('esm2_embedding', None)
                        vis_teacher = self.fixed_vis_sample.get('teacher_embedding', None)

                        if self.use_subcell_distillation and vis_teacher is not None:
                            _, _, original_patches, recon_patches, masked_patches = model(
                                self.fixed_vis_sample['image'],
                                esm2_emb=vis_esm2,
                                teacher_emb=vis_teacher,
                                return_image=True,
                                return_distill_loss=True
                            )
                        else:
                            _, original_patches, recon_patches, masked_patches = model(
                                self.fixed_vis_sample['image'],
                                esm2_emb=vis_esm2,
                                return_image=True
                            )
                        vis_outputs = (original_patches, recon_patches, masked_patches)

            # Log metrics (only rank 0)
            if self.rank == 0:
                log_dict = {
                    "loss": total_loss.item() * self.gradient_accumulation_steps,
                    "recon_loss": recon_loss.item(),
                    "epoch": epoch,
                    "step": self.global_step,
                    "lr": optimizer.param_groups[0]['lr']
                }

                if self.use_subcell_distillation:
                    log_dict["distill_loss"] = distill_loss.item()
                    log_dict["distill_weight"] = current_distill_weight

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
                    if self.use_subcell_distillation:
                        print(f"Epoch {epoch}/{args.epochs} | "
                              f"Iter {i}/{len(self.train_loader)} | "
                              f"Step {self.global_step} | "
                              f"Loss: {total_loss.item() * self.gradient_accumulation_steps:.4f} | "
                              f"Recon: {recon_loss.item():.4f} | "
                              f"Distill: {distill_loss.item():.4f} | "
                              f"Weight: {current_distill_weight:.3f}", flush=True)
                    else:
                        print(f"Epoch {epoch}/{args.epochs} | "
                              f"Iter {i}/{len(self.train_loader)} | "
                              f"Step {self.global_step} | "
                              f"Loss: {total_loss.item() * self.gradient_accumulation_steps:.4f}", flush=True)

            self.global_step += 1

    def validate_epoch(self, epoch):
        """Validation logic for one epoch."""
        if self.val_loader is None:
            return None

        args = self.args
        model = self.wrapped_model

        model.eval()
        val_recon_losses = []
        val_distill_losses = []

        print(f"\n=> Running validation...", flush=True)

        with torch.no_grad():
            for i, batch in enumerate(self.val_loader):
                images = batch['image'].cuda(self.local_rank, non_blocking=True)

                esm2_emb = None
                teacher_emb = None
                if self.use_esm2_conditioning and 'esm2_embedding' in batch:
                    esm2_emb = batch['esm2_embedding'].cuda(self.local_rank, non_blocking=True)
                if self.use_subcell_distillation and 'teacher_embedding' in batch:
                    teacher_emb = batch['teacher_embedding'].cuda(self.local_rank, non_blocking=True)

                with torch.cuda.amp.autocast(True):
                    if self.use_subcell_distillation and teacher_emb is not None:
                        recon_loss, distill_loss = model(
                            images,
                            esm2_emb=esm2_emb,
                            teacher_emb=teacher_emb,
                            return_image=False,
                            return_distill_loss=True
                        )
                    else:
                        recon_loss = model(
                            images,
                            esm2_emb=esm2_emb,
                            return_image=False
                        )
                        distill_loss = torch.tensor(0.0, device=images.device)

                val_recon_losses.append(recon_loss.item())
                val_distill_losses.append(distill_loss.item())

                if i % 10 == 0 and self.rank == 0:
                    if self.use_subcell_distillation:
                        print(f"   Val Iter {i}/{len(self.val_loader)} | "
                              f"Recon: {recon_loss.item():.4f} | "
                              f"Distill: {distill_loss.item():.4f}", flush=True)
                    else:
                        print(f"   Val Iter {i}/{len(self.val_loader)} | "
                              f"Recon: {recon_loss.item():.4f}", flush=True)

        if len(val_recon_losses) == 0:
            if self.rank == 0:
                print(f"=> WARNING: No validation data available")
            return None

        # Compute average losses
        avg_recon_loss = sum(val_recon_losses) / len(val_recon_losses)
        avg_distill_loss = sum(val_distill_losses) / len(val_distill_losses)

        if self.use_subcell_distillation:
            avg_total_loss = avg_recon_loss + self.distill_weight * avg_distill_loss
        else:
            avg_total_loss = avg_recon_loss

        # Synchronize across GPUs if distributed
        if self.is_distributed:
            import torch.distributed as dist
            losses_tensor = torch.tensor([avg_recon_loss, avg_distill_loss], device=self.local_rank)
            dist.all_reduce(losses_tensor, op=dist.ReduceOp.AVG)
            avg_recon_loss, avg_distill_loss = losses_tensor.tolist()
            if self.use_subcell_distillation:
                avg_total_loss = avg_recon_loss + self.distill_weight * avg_distill_loss
            else:
                avg_total_loss = avg_recon_loss

        if self.rank == 0:
            if self.use_subcell_distillation:
                print(f"=> Validation | Recon: {avg_recon_loss:.4f} | "
                      f"Distill: {avg_distill_loss:.4f} | "
                      f"Total: {avg_total_loss:.4f}\n", flush=True)
            else:
                print(f"=> Validation | Recon: {avg_recon_loss:.4f}\n", flush=True)

            log_dict = {
                "val_recon_loss": avg_recon_loss,
                "val_loss": avg_total_loss,
                "epoch": epoch,
            }
            if self.use_subcell_distillation:
                log_dict["val_distill_loss"] = avg_distill_loss

            wandb.log(log_dict)

        return avg_total_loss

    def resume(self):
        """
        Load pretrained weights.

        Uses strict=False to allow missing keys (esm2_proj, teacher_proj).
        Does NOT load optimizer state or set start_epoch since this may be
        fine-tuning from a different model type.
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

        # Load model state with strict=False
        state_dict = checkpoint['state_dict']

        if self.is_distributed:
            msg = self.wrapped_model.module.load_state_dict(state_dict, strict=False)
        else:
            msg = self.model.load_state_dict(state_dict, strict=False)

        print(f"=> loaded pretrained weights (strict=False)")
        print(f"   Missing keys: {msg.missing_keys}")
        print(f"   Unexpected keys: {msg.unexpected_keys[:5] if len(msg.unexpected_keys) > 5 else msg.unexpected_keys}")

        print(f"=> pretrained checkpoint loaded (epoch {checkpoint.get('epoch', 'N/A')} in original training)")
        print(f"=> fine-tuning will start from epoch 0")

        return checkpoint
