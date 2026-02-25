"""
Trainer for MAE3D with Channel Cross-Attention and FFT Loss.

FFT Loss Schedule:
- Warmup epochs: No FFT loss (only MSE)
- FFT ramp-up epochs: Gradually increase FFT loss weight from 0 to target
- Remaining epochs: Full FFT loss weight

Example with warmup_epochs=1 and fft_rampup_epochs=1:
- Epoch 0: FFT weight = 0 (warmup, MSE only)
- Epoch 1: FFT weight ramps 0 -> fft_loss_weight (linear per step)
- Epoch 2+: FFT weight = fft_loss_weight (full)
"""

import os
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import wandb

from .mae3d_cross_attention_trainer import MAE3DChannelCrossAttentionTrainer
from lib.models.mae3d_cross_attention_fft import MAE3DChannelCrossAttentionFFT


class MAE3DChannelCrossAttentionFFTTrainer(MAE3DChannelCrossAttentionTrainer):
    """
    Trainer for MAE3D with Channel Cross-Attention and FFT Loss.

    Handles FFT loss weight scheduling:
    - No FFT loss during warmup epochs
    - Linear ramp-up of FFT loss weight over fft_rampup_epochs
    - Full FFT loss weight for remaining epochs
    """

    def __init__(self, args):
        super().__init__(args)
        self.model_name = 'MAE3DChannelCrossAttentionFFT'

        # FFT loss schedule parameters (decoupled from LR warmup)
        self.fft_loss_weight = getattr(args, 'fft_loss_weight', 0.1)
        self.fft_warmup_epochs = getattr(args, 'fft_warmup_epochs', 0)
        self.fft_rampup_epochs = getattr(args, 'fft_rampup_epochs', 1)

        print(f"=> FFT Loss Schedule:")
        print(f"   FFT warmup epochs (no FFT): {self.fft_warmup_epochs}")
        print(f"   FFT ramp-up epochs: {self.fft_rampup_epochs}")
        print(f"   Target FFT weight: {self.fft_loss_weight}")

    def build_model(self):
        """
        Build MAE3D model with channel cross-attention and FFT loss.
        """
        if self.model is not None:
            raise ValueError("Model has been created. Do not create twice")

        args = self.args
        print(f"=> creating model {self.model_name}")
        print(f"   Cross-attention enabled: True")
        print(f"   FFT loss enabled: True")
        print(f"   Channels: {args.in_chans}")
        print(f"   Encoder depth: {args.encoder_depth}")
        print(f"   Decoder depth: {args.decoder_depth}")
        print(f"   Encoder embed dim: {args.encoder_embed_dim}")
        print(f"   Decoder embed dim: {args.decoder_embed_dim}")

        # Create model with FFT loss
        self.model = MAE3DChannelCrossAttentionFFT(args=args)

        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"   Total parameters: {total_params:,}")
        print(f"   Trainable parameters: {trainable_params:,}")

        # Wrap with DDP
        self.wrap_model()

    def _get_fft_weight_for_step(self, epoch: int, step_in_epoch: int, steps_per_epoch: int) -> float:
        """
        Calculate FFT weight based on current training progress.

        Schedule:
        - epoch < fft_warmup_epochs: weight = 0
        - fft_warmup_epochs <= epoch < fft_warmup_epochs + fft_rampup_epochs: linear ramp 0 -> target
        - epoch >= fft_warmup_epochs + fft_rampup_epochs: weight = target

        Args:
            epoch: Current epoch (0-indexed)
            step_in_epoch: Current step within epoch
            steps_per_epoch: Total steps per epoch

        Returns:
            Current FFT loss weight
        """
        warmup_epochs = self.fft_warmup_epochs
        rampup_epochs = self.fft_rampup_epochs
        target_weight = self.fft_loss_weight

        # During FFT warmup: no FFT loss
        if epoch < warmup_epochs:
            return 0.0

        # After FFT warmup + rampup: full weight
        if epoch >= warmup_epochs + rampup_epochs:
            return target_weight

        # During rampup: linear interpolation
        rampup_start_epoch = warmup_epochs
        epochs_into_rampup = epoch - rampup_start_epoch

        # Calculate total rampup steps and current rampup step
        total_rampup_steps = rampup_epochs * steps_per_epoch
        current_rampup_step = epochs_into_rampup * steps_per_epoch + step_in_epoch

        # Linear ramp from 0 to target
        progress = min(current_rampup_step / total_rampup_steps, 1.0)
        return target_weight * progress

    def epoch_train(self, epoch):
        """
        Training logic for one epoch with FFT loss scheduling.
        """
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
                loss, loss_mse, loss_fft = model(images, return_image=False)
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
                        _, _, _, original_patches, recon_patches, masked_patches = model(
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
        """
        Validation logic with separate MSE and FFT loss reporting.
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
                    loss, loss_mse, loss_fft = model(images, return_image=False)

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
