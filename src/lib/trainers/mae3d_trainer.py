import os
import time
import torch
import torch.distributed as dist
from torch.utils.data import DistributedSampler
import wandb
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .base_trainer import BaseTrainer
from lib.models.mae3d import MAE3D
from lib.networks.mae_vit import MAEViTEncoder, MAEViTDecoder
from data.opencell.dataset import OpenCellDataset
from data.opencell.transforms import get_opencell_train_transforms

# Import pynvml for GPU monitoring
try:
    import pynvml
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False
    print("Warning: pynvml not available. GPU metrics will not be logged.")


class MAE3DTrainer(BaseTrainer):
    """
    3D Masked Autoencoder Trainer for OpenCell dataset.
    """

    def __init__(self, args):
        super().__init__(args)
        self.model_name = 'MAE3D'
        self.scaler = torch.cuda.amp.GradScaler()
        self.global_step = 0
        self.fixed_vis_sample = None  # Store a fixed sample for consistent visualization

        # Gradient accumulation
        self.gradient_accumulation_steps = getattr(args, 'gradient_accumulation_steps', 1)
        self.accum_iter = 0  # Track accumulation iterations

        # Initialize GPU monitoring (only on rank 0)
        self.gpu_handles = []
        if self.rank == 0 and NVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                # Get handles for all GPUs in the job
                for i in range(self.world_size):
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    self.gpu_handles.append(handle)
                print(f"GPU monitoring initialized for {len(self.gpu_handles)} GPUs")
            except Exception as e:
                print(f"Warning: Could not initialize GPU monitoring: {e}")
                self.gpu_handles = []

    def build_model(self):
        """
        Build MAE3D model with encoder and decoder.
        """
        if self.model is not None:
            raise ValueError("Model has been created. Do not create twice")

        args = self.args
        print(f"=> creating model {self.model_name}")

        # Get encoder and decoder from args
        encoder_cls = MAEViTEncoder
        decoder_cls = MAEViTDecoder

        if hasattr(args, 'enc_arch'):
            # If encoder architecture is specified in config, use it
            # (allows for future extensibility)
            print(f"   Encoder: {args.enc_arch}")
            print(f"   Decoder: {args.dec_arch}")
        else:
            print(f"   Encoder: MAEViTEncoder")
            print(f"   Decoder: MAEViTDecoder")

        # Create model
        self.model = MAE3D(
            encoder=encoder_cls,
            decoder=decoder_cls,
            args=args
        )

        # Wrap with DDP
        self.wrap_model()

    def build_optimizer(self):
        """
        Build AdamW optimizer with parameter grouping.
        """
        assert self.model is not None and self.wrapped_model is not None, \
            "Model is not created and wrapped yet. Please create model first."

        print("=> creating optimizer")
        args = self.args

        # Group parameters (separate weight decay for conv/linear weights)
        optim_params = self.group_params(self.wrapped_model)

        # Scale learning rate by world_size for distributed training
        self.lr = args.lr * self.world_size

        # Create AdamW optimizer
        self.optimizer = torch.optim.AdamW(
            optim_params,
            lr=self.lr,
            betas=(args.beta1, args.beta2),
            weight_decay=args.weight_decay
        )

        print(f"   Optimizer: AdamW")
        print(f"   Learning rate: {self.lr:.6f} (base: {args.lr:.6f}, scaled by {self.world_size})")
        print(f"   Betas: ({args.beta1}, {args.beta2})")
        print(f"   Weight decay: {args.weight_decay}")
        print(f"   Gradient accumulation steps: {self.gradient_accumulation_steps}")

    def build_dataloader(self):
        """
        Build OpenCell train and validation dataloaders with distributed support.
        """
        if self.train_loader is not None:
            raise ValueError("Dataloader has been created. Do not create twice.")

        print("=> creating dataloaders")
        args = self.args

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
            z_slice_end=getattr(args, 'z_slice_end', None)
        )

        # Display cache statistics if caching is enabled (only rank 0)
        if args.cache_rate > 0 and self.rank == 0:
            cache_stats = train_dataset.get_cache_stats()
            print(f"\n  Train Dataset Cache Statistics:")
            print(f"    Total images: {cache_stats['total_images']}")
            print(f"    Cached images: {cache_stats['cached_images']}")
            print(f"    Cache rate: {cache_stats['cache_rate']:.2%}")
            print(f"    Cache hit rate: {cache_stats['cache_hit_rate']:.2%}\n")

        # Use 0 workers in distributed mode to avoid CUDA context conflicts
        # (same strategy as validation to ensure stability across epochs)
        train_workers = 0 if self.is_distributed else args.workers

        # Create train dataloader with DistributedSampler if using multiple GPUs
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

        # Create validation dataset (no augmentation for validation)
        val_csv_path = os.path.join(args.csv_path, "val.csv")
        if os.path.exists(val_csv_path):
            from data.opencell.transforms import get_opencell_val_transforms
            val_transform = get_opencell_val_transforms()

            val_dataset = OpenCellDataset(
                csv_path=val_csv_path,
                split='val',
                transform=val_transform,
                cache_rate=0.0,  # No caching for validation to save memory
                num_workers=args.workers,
                z_slice_start=getattr(args, 'z_slice_start', None),
                z_slice_end=getattr(args, 'z_slice_end', None)
            )

            # Create validation dataloader (no distributed sampler, no shuffling)
            # Always use 0 workers for validation to avoid CUDA initialization errors
            # (CUDA context from training conflicts with worker processes)
            val_workers = 0

            if self.is_distributed:
                val_sampler = DistributedSampler(
                    val_dataset,
                    num_replicas=self.world_size,
                    rank=self.rank,
                    shuffle=False  # Don't shuffle validation
                )
                self.val_loader = torch.utils.data.DataLoader(
                    val_dataset,
                    batch_size=args.batch_size,
                    sampler=val_sampler,
                    num_workers=val_workers,  # Use 0 workers in distributed mode
                    pin_memory=True
                )
            else:
                self.val_loader = val_dataset.get_dataloader(
                    batch_size=args.batch_size,
                    shuffle=False,
                    num_workers=val_workers
                )

            print(f"   Val samples: {len(val_dataset)}")
            print(f"   Val workers: {val_workers} (config: {args.workers})")
            print(f"   Validation enabled: True\n")
        else:
            print(f"   Validation file not found: {val_csv_path}")
            print(f"   Validation enabled: False\n")

        # Store a fixed sample for visualization (only on rank 0)
        if self.rank == 0:
            self._store_fixed_vis_sample()

    def _store_fixed_vis_sample(self):
        """
        Store a fixed sample from the dataset for consistent visualization across training.
        This allows tracking reconstruction quality on the same image over time.
        """
        print("\n=> Storing fixed visualization sample...")

        # Get the first batch from the dataloader
        for batch in self.train_loader:
            # Take only the first image from the batch
            self.fixed_vis_sample = batch['image'][0:1].cuda(self.local_rank, non_blocking=True)
            print(f"   Fixed sample shape: {self.fixed_vis_sample.shape}")
            break  # Only need one sample

    def get_gpu_metrics(self):
        """
        Get GPU metrics (utilization, memory, power, temperature) for all GPUs.
        Returns a dict with metrics for each GPU.
        """
        if not self.gpu_handles:
            return {}

        metrics = {}
        try:
            for i, handle in enumerate(self.gpu_handles):
                # Get utilization
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                metrics[f'gpu_{i}_utilization'] = util.gpu
                metrics[f'gpu_{i}_memory_utilization'] = util.memory

                # Get memory info
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                metrics[f'gpu_{i}_memory_used_mb'] = mem_info.used / 1024**2
                metrics[f'gpu_{i}_memory_total_mb'] = mem_info.total / 1024**2

                # Get power
                try:
                    power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert mW to W
                    metrics[f'gpu_{i}_power_watts'] = power
                except:
                    pass

                # Get temperature
                try:
                    temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                    metrics[f'gpu_{i}_temperature_c'] = temp
                except:
                    pass

        except Exception as e:
            print(f"Warning: Could not get GPU metrics: {e}")

        return metrics

    def epoch_train(self, epoch):
        """
        Training logic for one epoch.
        Note: Per-step LR warmup is handled here during warmup epochs.
        Per-epoch LR adjustment is handled by BaseTrainer.run() after warmup.
        """
        args = self.args
        model = self.wrapped_model
        optimizer = self.optimizer
        scaler = self.scaler

        model.train()

        # Calculate warmup settings
        warmup_steps = args.warmup_epochs * len(self.train_loader)
        warmup_start_lr = self.lr * 0.01  # Start from 1% of base LR

        for i, batch in enumerate(self.train_loader):
            images = batch['image'].cuda(self.local_rank, non_blocking=True)

            # Per-step LR adjustment during warmup
            if self.global_step < warmup_steps:
                # Linear warmup from 1% to 100% of base LR
                progress = min(self.global_step / warmup_steps, 1.0)  # Cap at 1.0
                current_lr = warmup_start_lr + (self.lr - warmup_start_lr) * progress
                for param_group in optimizer.param_groups:
                    param_group['lr'] = current_lr

            # Determine if we should visualize this step
            should_visualize = (self.global_step % args.vis_freq == 0) and (self.rank == 0) and (self.fixed_vis_sample is not None)

            # Zero gradients at the start of accumulation
            if self.accum_iter == 0:
                optimizer.zero_grad()

            # Forward pass with mixed precision (for training)
            with torch.cuda.amp.autocast(True):
                loss = model(images, return_image=False)
                # Scale loss for gradient accumulation
                loss = loss / self.gradient_accumulation_steps

            # Backward pass
            scaler.scale(loss).backward()

            # Increment accumulation counter
            self.accum_iter += 1

            # Only update weights after accumulating enough gradients
            if self.accum_iter >= self.gradient_accumulation_steps:
                # Gradient clipping (must unscale first when using GradScaler)
                if hasattr(args, 'grad_clip') and args.grad_clip is not None and args.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

                scaler.step(optimizer)
                scaler.update()

                # Reset accumulation counter
                self.accum_iter = 0

            # Generate visualization on fixed sample (only on rank 0)
            vis_outputs = None
            if should_visualize:
                with torch.no_grad():
                    with torch.cuda.amp.autocast(True):
                        _, original_patches, recon_patches, masked_patches = model(self.fixed_vis_sample, return_image=True)
                vis_outputs = (original_patches, recon_patches, masked_patches)

            # Log metrics (only on rank 0)
            if self.rank == 0:
                log_dict = {
                    "loss": loss.item() * self.gradient_accumulation_steps,  # Unscale for logging
                    "epoch": epoch,
                    "step": self.global_step,
                    "lr": optimizer.param_groups[0]['lr']
                }

                # Add GPU metrics for all GPUs
                gpu_metrics = self.get_gpu_metrics()
                log_dict.update(gpu_metrics)

                # Add visualization if needed
                if should_visualize and vis_outputs is not None:
                    try:
                        print(f"  Generating visualization at step {self.global_step} (using fixed sample)...", flush=True)
                        original_patches, recon_patches, masked_patches = vis_outputs
                        # Get visualization z-slices from config or use defaults based on input size
                        vis_z_slices = getattr(args, 'vis_z_slices', None)
                        if vis_z_slices is None:
                            # Default: use 3 evenly spaced slices
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
                          f"Loss: {loss.item() * self.gradient_accumulation_steps:.4f}", flush=True)

            self.global_step += 1

    def validate_epoch(self, epoch):
        """
        Validation logic for one epoch.
        Computes average reconstruction loss on validation set.
        """
        if self.val_loader is None:
            return None

        args = self.args
        model = self.wrapped_model

        model.eval()
        val_losses = []

        print(f"\n=> Running validation...", flush=True)

        with torch.no_grad():
            for i, batch in enumerate(self.val_loader):
                images = batch['image'].cuda(self.local_rank, non_blocking=True)

                # Forward pass with mixed precision
                with torch.cuda.amp.autocast(True):
                    loss = model(images, return_image=False)

                val_losses.append(loss.item())

                if i % 10 == 0 and self.rank == 0:
                    print(f"   Val Iter {i}/{len(self.val_loader)} | Loss: {loss.item():.4f}", flush=True)

        # Check if we have any validation samples
        if len(val_losses) == 0:
            if self.rank == 0:
                print(f"=> WARNING: No validation data available (val_losses is empty)")
            return None

        # Compute average loss
        avg_val_loss = sum(val_losses) / len(val_losses)

        # Synchronize validation loss across all GPUs if distributed
        if self.is_distributed:
            import torch.distributed as dist
            val_loss_tensor = torch.tensor([avg_val_loss], device=self.local_rank)
            dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.AVG)
            avg_val_loss = val_loss_tensor.item()

        if self.rank == 0:
            print(f"=> Validation Loss: {avg_val_loss:.4f}\n", flush=True)

        return avg_val_loss

    def run(self):
        """
        Main training loop with validation support.
        Overrides BaseTrainer.run() to add validation at the end of each epoch.
        """
        args = self.args

        for epoch in range(args.start_epoch, args.epochs):
            # Set epoch for distributed sampler
            if self.is_distributed and hasattr(self.train_loader, 'sampler'):
                if hasattr(self.train_loader.sampler, 'set_epoch'):
                    self.train_loader.sampler.set_epoch(epoch)

            # Adjust learning rate
            current_lr = self.adjust_learning_rate(epoch)

            if self.rank == 0:
                print(f"\n{'='*60}", flush=True)
                print(f"Epoch {epoch}/{args.epochs} | LR: {current_lr:.6f}", flush=True)
                print(f"{'='*60}", flush=True)

            # Train for one epoch
            self.epoch_train(epoch)

            # Validate if validation loader exists
            val_loss = None
            if self.val_loader is not None:
                val_loss = self.validate_epoch(epoch)

                # Log validation loss to wandb (only rank 0)
                if self.rank == 0 and val_loss is not None:
                    wandb.log({
                        "val_loss": val_loss,
                        "epoch": epoch,
                    })

            # Save checkpoint
            if (epoch + 1) % args.save_freq == 0:
                checkpoint_kwargs = {}
                if val_loss is not None:
                    checkpoint_kwargs['val_loss'] = val_loss
                self.save_checkpoint(epoch, **checkpoint_kwargs)

        if self.rank == 0:
            print("\nTraining completed!")

    def visualize_mae_reconstruction(self, original_patches, masked_patches, recon_patches,
                                     z_slices=[20, 50, 80]):
        """
        Create a 6x4 grid visualization showing MAE reconstruction:
        - Rows: 2 channels × 3 z-slices = 6 rows
        - Columns: Original | Encoder Input (masked) | Reconstructed | Colorbar
        - Each row shares the same colormap range for fair comparison
        """
        from matplotlib.gridspec import GridSpec

        args = self.args

        # Unpatchify all images (take first sample in batch)
        original = self.unpatchify_image(original_patches[:1])
        masked = self.unpatchify_image(masked_patches[:1])
        recon = self.unpatchify_image(recon_patches[:1])

        # Convert to numpy
        # Shape after unpatchify: [C, Z, Y, X] where input_size=[Z, Y, X]=[100, 176, 176]
        original = original[0].cpu().numpy()  # [C, Z, Y, X] = [2, 100, 176, 176]
        masked = masked[0].cpu().numpy()
        recon = recon[0].cpu().numpy()

        # Create figure with GridSpec for equal column spacing + colorbar column
        n_slices = len(z_slices)
        n_channels = args.in_chans
        n_rows = n_channels * n_slices

        fig = plt.figure(figsize=(16, 2.5 * n_rows), facecolor='white')
        # 3 equal image columns + 1 narrow colorbar column
        gs = GridSpec(n_rows, 4, figure=fig, width_ratios=[1, 1, 1, 0.05], wspace=0.08, hspace=0.15)

        # Overall title with metadata
        fig.suptitle(f'MAE 3D Reconstruction | Shape: {tuple(args.input_size)} (Z,Y,X) | Mask Ratio: {args.mask_ratio:.1%}',
                     fontsize=22, fontweight='bold', y=0.995)

        # Column titles
        col_titles = ['Original', 'Encoder Input (masked)', 'Reconstructed']
        channel_names = ['Nucleus', 'Protein'] if n_channels == 2 else [f'Ch{i}' for i in range(n_channels)]

        # Channel colormaps: nucleus → black-to-blue (DAPI style), protein → grayscale
        import matplotlib.colors as mcolors
        nucleus_cmap = mcolors.LinearSegmentedColormap.from_list('nucleus', ['black', '#4488ff'])
        channel_cmaps = [nucleus_cmap, 'gray'] if n_channels == 2 else ['gray'] * n_channels

        # Iterate over channels and slices
        for ch in range(n_channels):
            for slice_idx, z_slice in enumerate(z_slices):
                row_idx = ch * n_slices + slice_idx

                # Extract Y-X plane at Z position z_slice for all three columns
                # img shape: [C, Z, Y, X] = [2, 100, 176, 176]
                orig_slice = original[ch, z_slice, :, :]
                masked_slice = masked[ch, z_slice, :, :]
                recon_slice = recon[ch, z_slice, :, :]

                # Compute shared intensity range across the entire row for fair comparison
                all_values = np.concatenate([orig_slice.flatten(), masked_slice.flatten(), recon_slice.flatten()])
                vmin, vmax = np.percentile(all_values, [1, 99])

                im = None
                for col_idx, (img_slice, title) in enumerate(zip([orig_slice, masked_slice, recon_slice], col_titles)):
                    ax = fig.add_subplot(gs[row_idx, col_idx])

                    im = ax.imshow(img_slice, cmap=channel_cmaps[ch], vmin=vmin, vmax=vmax,
                                  interpolation='bilinear', aspect='equal')

                    # Add column title on first row
                    if row_idx == 0:
                        ax.set_title(title, fontsize=18, fontweight='bold', pad=10)

                    # Add row label on first column
                    if col_idx == 0:
                        label = f'{channel_names[ch]} | Z={z_slice}'
                        ax.set_ylabel(label, fontsize=14, fontweight='bold',
                                     rotation=0, ha='right', va='center', labelpad=55)

                        # Add slice info as text on image (top-left corner)
                        z_dim = args.input_size[0]
                        ax.text(0.02, 0.98, f'Z={z_slice}/{z_dim-1}',
                               transform=ax.transAxes, fontsize=12,
                               verticalalignment='top', horizontalalignment='left',
                               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

                    # Clean up ticks
                    ax.set_xticks([])
                    ax.set_yticks([])

                    # Add thin border
                    for spine in ax.spines.values():
                        spine.set_edgecolor('gray')
                        spine.set_linewidth(0.5)

                # Add colorbar in the 4th column for this row
                cbar_ax = fig.add_subplot(gs[row_idx, 3])
                cbar = plt.colorbar(im, cax=cbar_ax)
                cbar.ax.tick_params(labelsize=12)

        # Convert to wandb image
        wandb_image = wandb.Image(fig)
        plt.close(fig)

        return wandb_image

    def unpatchify_image(self, patches):
        """
        Reverse of patchify_image: converts patches back to image.

        Args:
            patches: [B, gh*gw*gd, ph*pw*pd*C]

        Returns:
            [B, C, Z, Y, X] where:
                - Z = input_size[0] = roi_z = 100 (depth/number of slices)
                - Y = input_size[1] = roi_y = 176 (height)
                - X = input_size[2] = roi_x = 176 (width)

        Note: Variable names H,W,D are used for compatibility with patchify_image,
              but they correspond to Z,Y,X dimensions respectively.
        """
        args = self.args
        B = patches.shape[0]
        # args.input_size = [roi_z, roi_y, roi_x] = [100, 176, 176]
        H, W, D = args.input_size  # H=Z=100, W=Y=176, D=X=176
        ph, pw, pd = args.patch_size
        gh, gw, gd = H // ph, W // pw, D // pd
        in_chans = args.in_chans

        # Reshape patches to [B, gh, gw, gd, ph, pw, pd, C]
        x = patches.reshape(B, gh, gw, gd, ph, pw, pd, in_chans)
        # Permute to [B, C, gh, ph, gw, pw, gd, pd]
        x = x.permute(0, 7, 1, 4, 2, 5, 3, 6)
        # Reshape to [B, C, H, W, D] = [B, C, Z, Y, X]
        x = x.reshape(B, in_chans, H, W, D)

        return x

    def save_checkpoint(self, epoch, **kwargs):
        """
        Override to include scaler state in checkpoint.
        """
        if self.rank != 0:
            return

        # Get unwrapped model state dict
        model_state = self.model.state_dict() if not self.is_distributed else self.wrapped_model.module.state_dict()

        checkpoint = {
            'epoch': epoch + 1,
            'arch': self.args.arch if hasattr(self.args, 'arch') else 'mae3d',
            'state_dict': model_state,
            'optimizer': self.optimizer.state_dict(),
            'scaler': self.scaler.state_dict(),
            'global_step': self.global_step,
        }

        # Add any additional kwargs
        checkpoint.update(kwargs)

        from pathlib import Path
        checkpoint_path = Path(self.args.ckpt_dir) / f"checkpoint_{epoch:04d}.pth.tar"
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")

    def resume(self):
        """
        Override to include scaler state when resuming.
        """
        checkpoint = super().resume()

        if checkpoint is not None and 'scaler' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler'])
            print("=> loaded scaler state")

        if checkpoint is not None and 'global_step' in checkpoint:
            self.global_step = checkpoint['global_step']
            print(f"=> resuming from global step {self.global_step}")
