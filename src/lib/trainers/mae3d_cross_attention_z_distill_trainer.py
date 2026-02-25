"""
Trainer for MAE3D with Channel Cross-Attention and Z-Aware Attention Distillation.

Extends MAE3DChannelCrossAttentionDistillTrainer with:
- Alpha blending between attention-based and global-pool distillation
- Alpha ramp-up schedule
- Separate logging of distill_loss_attn, distill_loss_global, distill_alpha
"""

import torch
import wandb

from .mae3d_cross_attention_distill_trainer import MAE3DChannelCrossAttentionDistillTrainer
from lib.models.mae3d_cross_attention_z_distill import MAE3DChannelCrossAttentionZDistill


class MAE3DChannelCrossAttentionZDistillTrainer(MAE3DChannelCrossAttentionDistillTrainer):
    """
    Trainer for MAE3D with Z-Aware Attention Distillation.

    Key differences from MAE3DChannelCrossAttentionDistillTrainer:
    - Uses MAE3DChannelCrossAttentionZDistill model
    - Manages distill_attn_alpha with ramp-up
    - Logs attention and global distillation losses separately
    """

    def __init__(self, args):
        super().__init__(args)
        self.model_name = 'MAE3DChannelCrossAttentionZDistill'

        # Z-aware attention distillation configuration
        self.distill_attn_alpha = getattr(args, 'distill_attn_alpha', 0.8)
        self.distill_attn_alpha_rampup_epochs = getattr(args, 'distill_attn_alpha_rampup_epochs',
                                                         self.distill_rampup_epochs)

        print(f"\n=> Z-Aware Attention Distillation Configuration:")
        print(f"   Attention alpha (target): {self.distill_attn_alpha}")
        print(f"   Alpha ramp-up epochs: {self.distill_attn_alpha_rampup_epochs}")

    def build_model(self):
        """
        Build MAE3D model with z-aware attention distillation.
        """
        if self.model is not None:
            raise ValueError("Model has been created. Do not create twice")

        args = self.args
        print(f"=> creating model {self.model_name}")
        print(f"   Cross-attention enabled: True")
        print(f"   Z-Aware Attention Distillation enabled: True")
        print(f"   Channels: {args.in_chans}")
        print(f"   Encoder depth: {args.encoder_depth}")
        print(f"   Decoder depth: {args.decoder_depth}")
        print(f"   Encoder embed dim: {args.encoder_embed_dim}")
        print(f"   Decoder embed dim: {args.decoder_embed_dim}")
        print(f"   Teacher embed dim: {getattr(args, 'teacher_embed_dim', 1536)}")
        print(f"   Distill loss type: {getattr(args, 'distill_loss_type', 'cosine')}")
        print(f"   Num query tokens: {getattr(args, 'distill_num_query_tokens', 4)}")
        print(f"   Num attn heads: {getattr(args, 'distill_num_attn_heads', 6)}")

        # Create model with z-aware attention distillation
        self.model = MAE3DChannelCrossAttentionZDistill(args=args)

        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        attn_head_params = sum(p.numel() for p in self.model.attn_distill_head.parameters())
        print(f"   Total parameters: {total_params:,}")
        print(f"   Trainable parameters: {trainable_params:,}")
        print(f"   Attention distill head parameters: {attn_head_params:,}")

        # Wrap with DDP
        self.wrap_model()

    def get_attn_alpha(self, epoch, step_in_epoch, total_steps_in_epoch):
        """
        Compute attention alpha with ramp-up.

        During ramp-up, alpha increases linearly from 0 to target.
        This allows the global pool path to dominate early (more stable),
        then gradually shifts to the attention path.

        Args:
            epoch: Current epoch (0-indexed)
            step_in_epoch: Current step within epoch
            total_steps_in_epoch: Total steps in one epoch

        Returns:
            Current attention alpha
        """
        if self.distill_attn_alpha_rampup_epochs <= 0:
            return self.distill_attn_alpha

        if epoch >= self.distill_attn_alpha_rampup_epochs:
            return self.distill_attn_alpha

        # Linear ramp-up
        total_rampup_steps = self.distill_attn_alpha_rampup_epochs * total_steps_in_epoch
        current_step = epoch * total_steps_in_epoch + step_in_epoch
        progress = min(current_step / total_rampup_steps, 1.0)

        return self.distill_attn_alpha * progress

    def epoch_train(self, epoch):
        """
        Training logic for one epoch with z-aware attention distillation.
        """
        args = self.args
        model = self.wrapped_model
        optimizer = self.optimizer
        scaler = self.scaler

        # Handle decoder freezing during ramp-up
        if self.freeze_decoder_during_rampup:
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
            teacher_emb = batch['teacher_embedding'].cuda(self.local_rank, non_blocking=True)

            # Per-step LR adjustment during warmup
            if self.global_step < warmup_steps:
                progress = min(self.global_step / warmup_steps, 1.0)
                current_lr = warmup_start_lr + (self.lr - warmup_start_lr) * progress
                for param_group in optimizer.param_groups:
                    param_group['lr'] = current_lr

            # Get current distillation weight with ramp-up
            current_distill_weight = self.get_distill_weight(epoch, i, total_steps_in_epoch)

            # Get current attention alpha with ramp-up
            current_alpha = self.get_attn_alpha(epoch, i, total_steps_in_epoch)

            # Determine if we should visualize
            should_visualize = (self.global_step % args.vis_freq == 0) and (self.rank == 0) and (self.fixed_vis_sample is not None)

            # Zero gradients at the start of accumulation
            if self.accum_iter == 0:
                optimizer.zero_grad()

            # Forward pass with z-aware attention distillation
            with torch.cuda.amp.autocast(True):
                recon_loss, distill_loss, attn_distill_loss, global_distill_loss = model(
                    images,
                    teacher_emb=teacher_emb,
                    return_image=False,
                    return_distill_loss=True,
                    distill_attn_alpha=current_alpha,
                )

                # Combined loss
                total_loss = recon_loss + current_distill_weight * distill_loss

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
                        outputs = model(
                            self.fixed_vis_sample['image'],
                            teacher_emb=self.fixed_vis_sample['teacher_embedding'],
                            return_image=True,
                            return_distill_loss=True,
                            distill_attn_alpha=current_alpha,
                        )
                        # outputs: recon_loss, distill_loss, attn_distill_loss, global_distill_loss, original, recon, masked
                        original_patches = outputs[4]
                        recon_patches = outputs[5]
                        masked_patches = outputs[6]
                vis_outputs = (original_patches, recon_patches, masked_patches)

            # Log metrics (only on rank 0)
            if self.rank == 0:
                log_dict = {
                    "loss": total_loss.item() * self.gradient_accumulation_steps,
                    "recon_loss": recon_loss.item(),
                    "distill_loss": distill_loss.item(),
                    "distill_loss_attn": attn_distill_loss.item(),
                    "distill_loss_global": global_distill_loss.item(),
                    "distill_weight": current_distill_weight,
                    "distill_alpha": current_alpha,
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
                          f"Loss: {total_loss.item() * self.gradient_accumulation_steps:.4f} | "
                          f"Recon: {recon_loss.item():.4f} | "
                          f"Distill: {distill_loss.item():.4f} "
                          f"(attn: {attn_distill_loss.item():.4f}, "
                          f"global: {global_distill_loss.item():.4f}) | "
                          f"Alpha: {current_alpha:.3f} | "
                          f"Weight: {current_distill_weight:.3f}", flush=True)

            self.global_step += 1

    def validate_epoch(self, epoch):
        """
        Validation logic for one epoch with z-aware attention distillation.
        """
        if self.val_loader is None:
            return None

        args = self.args
        model = self.wrapped_model

        model.eval()
        val_recon_losses = []
        val_distill_losses = []
        val_attn_distill_losses = []
        val_global_distill_losses = []

        print(f"\n=> Running validation...", flush=True)

        with torch.no_grad():
            for i, batch in enumerate(self.val_loader):
                images = batch['image'].cuda(self.local_rank, non_blocking=True)
                teacher_emb = batch['teacher_embedding'].cuda(self.local_rank, non_blocking=True)

                with torch.cuda.amp.autocast(True):
                    recon_loss, distill_loss, attn_distill_loss, global_distill_loss = model(
                        images,
                        teacher_emb=teacher_emb,
                        return_image=False,
                        return_distill_loss=True,
                        distill_attn_alpha=self.distill_attn_alpha,
                    )

                val_recon_losses.append(recon_loss.item())
                val_distill_losses.append(distill_loss.item())
                val_attn_distill_losses.append(attn_distill_loss.item())
                val_global_distill_losses.append(global_distill_loss.item())

                if i % 10 == 0 and self.rank == 0:
                    print(f"   Val Iter {i}/{len(self.val_loader)} | "
                          f"Recon: {recon_loss.item():.4f} | "
                          f"Distill: {distill_loss.item():.4f} "
                          f"(attn: {attn_distill_loss.item():.4f}, "
                          f"global: {global_distill_loss.item():.4f})", flush=True)

        if len(val_recon_losses) == 0:
            if self.rank == 0:
                print(f"=> WARNING: No validation data available")
            return None

        # Compute average losses
        avg_recon_loss = sum(val_recon_losses) / len(val_recon_losses)
        avg_distill_loss = sum(val_distill_losses) / len(val_distill_losses)
        avg_attn_distill_loss = sum(val_attn_distill_losses) / len(val_attn_distill_losses)
        avg_global_distill_loss = sum(val_global_distill_losses) / len(val_global_distill_losses)
        avg_total_loss = avg_recon_loss + self.distill_weight * avg_distill_loss

        # Synchronize across GPUs if distributed
        if self.is_distributed:
            import torch.distributed as dist
            losses_tensor = torch.tensor(
                [avg_recon_loss, avg_distill_loss, avg_attn_distill_loss, avg_global_distill_loss],
                device=self.local_rank
            )
            dist.all_reduce(losses_tensor, op=dist.ReduceOp.AVG)
            avg_recon_loss, avg_distill_loss, avg_attn_distill_loss, avg_global_distill_loss = losses_tensor.tolist()
            avg_total_loss = avg_recon_loss + self.distill_weight * avg_distill_loss

        if self.rank == 0:
            print(f"=> Validation | Recon: {avg_recon_loss:.4f} | "
                  f"Distill: {avg_distill_loss:.4f} "
                  f"(attn: {avg_attn_distill_loss:.4f}, "
                  f"global: {avg_global_distill_loss:.4f}) | "
                  f"Total: {avg_total_loss:.4f}\n", flush=True)

            wandb.log({
                "val_recon_loss": avg_recon_loss,
                "val_distill_loss": avg_distill_loss,
                "val_distill_loss_attn": avg_attn_distill_loss,
                "val_distill_loss_global": avg_global_distill_loss,
                "val_loss": avg_total_loss,
                "epoch": epoch,
            })

        return avg_total_loss
