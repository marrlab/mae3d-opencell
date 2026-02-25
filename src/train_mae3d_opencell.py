"""
MAE3D Training Script for OpenCell Dataset (Direct Script Approach)

This is the original direct/functional approach to training MAE3D.
All logic is contained in this single file for simplicity.

For a more modular, object-oriented approach using the Trainer pattern,
see: train_with_trainer.py and lib/trainers/

Both scripts produce identical results - choose based on your preference:
- This script: Simple, straightforward, all code in one place
- Trainer pattern: Modular, extensible, easier to add new tasks

See src/TRAINER_README.md for detailed comparison.

Usage:
    # Single GPU
    python src/train_mae3d_opencell.py

    # Multi-GPU with torchrun
    torchrun --nproc_per_node=4 src/train_mae3d_opencell.py
"""

import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler
import wandb
from pathlib import Path
from omegaconf import OmegaConf
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for server environments
import matplotlib.pyplot as plt

from lib.models.mae3d import MAE3D
from lib.networks.mae_vit import MAEViTEncoder, MAEViTDecoder
from utils.utils import set_seed, get_conf
from utils.logger import redirect_stdout_to_file, save_config_to_file
from data.opencell.dataset import OpenCellDataset
from data.opencell.transforms import get_opencell_train_transforms


def unpatchify_image(patches, patch_size, image_size, in_chans):
    """
    Reverse of patchify_image: converts patches back to image.
    patches: [B, gh*gw*gd, ph*pw*pd*C]
    Returns: [B, C, H, W, D]
    """
    B = patches.shape[0]
    H, W, D = image_size
    ph, pw, pd = patch_size
    gh, gw, gd = H // ph, W // pw, D // pd

    # Reshape patches to [B, gh, gw, gd, ph, pw, pd, C]
    x = patches.reshape(B, gh, gw, gd, ph, pw, pd, in_chans)
    # Permute to [B, C, gh, ph, gw, pw, gd, pd]
    x = x.permute(0, 7, 1, 4, 2, 5, 3, 6)
    # Reshape to [B, C, H, W, D]
    x = x.reshape(B, in_chans, H, W, D)

    return x


def visualize_mae_reconstruction(original_patches, masked_patches, recon_patches,
                                 patch_size, image_size, in_chans, mask_ratio,
                                 z_slices=[20, 50, 80]):
    """
    Create a beautiful 6x4 grid visualization showing MAE reconstruction:
    - Rows: 2 channels × 3 z-slices = 6 rows
    - Columns: Original | Encoder Input (masked) | Reconstructed | Colorbar
    - Shows image shape and mask ratio as suptitle
    - Each row shares the same colormap range for fair comparison
    """
    from matplotlib.gridspec import GridSpec

    # Unpatchify all images (take first sample in batch)
    original = unpatchify_image(original_patches[:1], patch_size, image_size, in_chans)
    masked = unpatchify_image(masked_patches[:1], patch_size, image_size, in_chans)
    recon = unpatchify_image(recon_patches[:1], patch_size, image_size, in_chans)

    # Convert to numpy
    original = original[0].cpu().numpy()  # [C, H, W, D]
    masked = masked[0].cpu().numpy()
    recon = recon[0].cpu().numpy()

    # Create figure with GridSpec for equal column spacing + colorbar column
    n_slices = len(z_slices)
    n_channels = 2
    n_rows = n_channels * n_slices

    fig = plt.figure(figsize=(16, 4 * n_slices), facecolor='white')
    # 3 equal image columns + 1 narrow colorbar column
    gs = GridSpec(n_rows, 4, figure=fig, width_ratios=[1, 1, 1, 0.05], wspace=0.08, hspace=0.15)

    # Overall title with metadata
    fig.suptitle(f'MAE 3D Reconstruction | Shape: {tuple(image_size)} | Mask Ratio: {mask_ratio:.1%}',
                 fontsize=22, fontweight='bold', y=0.995)

    # Column titles
    col_titles = ['Original', 'Encoder Input (masked)', 'Reconstructed']

    # Store image mappables for colorbars
    row_images = []

    # Iterate over channels and slices
    for ch in range(n_channels):
        for slice_idx, z_slice in enumerate(z_slices):
            row_idx = ch * n_slices + slice_idx

            # Extract z-slices for this row (all three columns)
            orig_slice = original[ch, :, :, z_slice]
            masked_slice = masked[ch, :, :, z_slice]
            recon_slice = recon[ch, :, :, z_slice]

            # Compute shared intensity range across the entire row for fair comparison
            all_values = np.concatenate([orig_slice.flatten(), masked_slice.flatten(), recon_slice.flatten()])
            vmin, vmax = np.percentile(all_values, [1, 99])

            im = None
            for col_idx, (img_slice, title) in enumerate(zip([orig_slice, masked_slice, recon_slice], col_titles)):
                ax = fig.add_subplot(gs[row_idx, col_idx])

                # Plot with viridis colormap (prettier than gray)
                im = ax.imshow(img_slice, cmap='viridis', vmin=vmin, vmax=vmax,
                              interpolation='bilinear')

                # Add column title on first row
                if row_idx == 0:
                    ax.set_title(title, fontsize=18, fontweight='bold', pad=10)

                # Add row label on first column
                if col_idx == 0:
                    label = f'Ch{ch} | Z={z_slice}'
                    ax.set_ylabel(label, fontsize=14, fontweight='bold',
                                 rotation=0, ha='right', va='center', labelpad=45)

                # Add slice info as text on image (top-left corner)
                if col_idx == 0:
                    ax.text(0.02, 0.98, f'Z={z_slice}/{image_size[2]-1}',
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


def main():
    # Initialize distributed training (works with torchrun)
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)
        is_distributed = True
        print(f'Distributed training: rank {rank}/{world_size}, local_rank {local_rank}')
    else:
        rank = 0
        world_size = 1
        local_rank = 0
        is_distributed = False
        print('Single GPU training')

    # Load config
    code_base_path = '/lustre/groups/aih/amirhossein.kardoost/codes/single-cell-foundation-model/single-cell-foundation-model/'
    if rank == 0:
        print(f'code base path: {code_base_path}')
    args = get_conf(os.path.join(code_base_path,
                                 'configs/opencell/opencell_3d.yaml'))
    set_seed(args.seed + rank)  # Different seed per rank

    # Setup logging (redirect stdout to file)
    tee_logger = None
    if rank == 0:
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)

        # Redirect stdout to file
        tee_logger = redirect_stdout_to_file(args.output_dir, log_filename='training.log', rank=rank)

        # Save config to output directory
        save_config_to_file(args, args.output_dir, filename='config.yaml', rank=rank)

    # Create dataset
    csv_path = os.path.join(args.csv_path, "train.csv")
    train_transform = get_opencell_train_transforms(
        flip_prob=args.RandFlipd_prob,
        rotate_prob=args.RandRotate90d_prob
    )

    train_dataset = OpenCellDataset(
        csv_path=csv_path,
        split='train',
        transform=train_transform,
        cache_rate=args.cache_rate,
        num_workers=args.workers
    )

    # Display cache statistics if caching is enabled (only rank 0)
    if args.cache_rate > 0 and rank == 0:
        cache_stats = train_dataset.get_cache_stats()
        print(f"\nDataset Cache Statistics:")
        print(f"  Total images: {cache_stats['total_images']}")
        print(f"  Cached images: {cache_stats['cached_images']}")
        print(f"  Cache rate: {cache_stats['cache_rate']:.2%}")
        print(f"  Cache hit rate: {cache_stats['cache_hit_rate']:.2%}\n")

    # Create dataloader with DistributedSampler if using multiple GPUs
    if is_distributed:
        sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            sampler=sampler,
            num_workers=args.workers,
            pin_memory=True
        )
    else:
        train_loader = train_dataset.get_dataloader(
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers
        )

    # Create model
    model = MAE3D(encoder=MAEViTEncoder,
                  decoder=MAEViTDecoder,
                  args=args)
    model = model.cuda(local_rank)

    # Wrap with DDP if distributed
    if is_distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
        if rank == 0:
            print(f"Model wrapped with DistributedDataParallel on {world_size} GPUs")

    # Create optimizer (after DDP wrapping)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr * world_size,  # Scale learning rate with number of GPUs
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay
    )

    # Setup wandb (only on rank 0)
    if rank == 0:
        # Configure wandb to save locally in output_dir
        wandb_dir = os.path.join(args.output_dir, 'wandb')
        os.makedirs(wandb_dir, exist_ok=True)

        run = wandb.init(
            project=f"{args.proj_name}_{args.dataset}",
            name=args.run_name,
            config=OmegaConf.to_container(args, resolve=True),
            dir=wandb_dir,  # Save wandb files locally
            save_code=True   # Save code snapshot
        )
        print(f'\nWandB directory: {wandb_dir}')
    else:
        run = None

    # Training loop
    scaler = torch.cuda.amp.GradScaler()
    global_step = 0

    # Gradient accumulation
    gradient_accumulation_steps = getattr(args, 'gradient_accumulation_steps', 1)
    accum_iter = 0

    if rank == 0:
        print(f"\nGradient accumulation steps: {gradient_accumulation_steps}")

    for epoch in range(args.epochs):
        # Set epoch for distributed sampler
        if is_distributed:
            sampler.set_epoch(epoch)

        model.train()

        for i, batch in enumerate(train_loader):
            images = batch['image'].cuda(local_rank)

            # Visualize based on vis_freq from config (only on rank 0)
            should_visualize = (global_step % args.vis_freq == 0) and (rank == 0)

            # Zero gradients at the start of accumulation
            if accum_iter == 0:
                optimizer.zero_grad()

            # Forward
            with torch.cuda.amp.autocast(True):
                if should_visualize:
                    loss, original_patches, recon_patches, masked_patches = model(images, return_image=True)
                else:
                    loss = model(images, return_image=False)
                # Scale loss for gradient accumulation
                loss = loss / gradient_accumulation_steps

            # Backward
            scaler.scale(loss).backward()

            # Increment accumulation counter
            accum_iter += 1

            # Only update weights after accumulating enough gradients
            if accum_iter >= gradient_accumulation_steps:
                scaler.step(optimizer)
                scaler.update()
                # Reset accumulation counter
                accum_iter = 0

            # Log metrics (only on rank 0)
            if rank == 0:
                log_dict = {"loss": loss.item() * gradient_accumulation_steps, "epoch": epoch, "step": global_step}  # Unscale for logging

                # Add visualization based on vis_freq
                if should_visualize:
                    try:
                        print(f"  Generating visualization at step {global_step}...")
                        vis_image = visualize_mae_reconstruction(
                            original_patches=original_patches,
                            masked_patches=masked_patches,
                            recon_patches=recon_patches,
                            patch_size=args.patch_size,
                            image_size=args.input_size,
                            in_chans=args.in_chans,
                            mask_ratio=args.mask_ratio,
                            z_slices=[20, 50, 80]
                        )
                        log_dict["reconstruction"] = vis_image
                        print(f"  Visualization logged successfully!")
                    except Exception as e:
                        print(f"  WARNING: Visualization failed: {e}")
                        import traceback
                        traceback.print_exc()

                wandb.log(log_dict)

                # Print
                if i % args.print_freq == 0:
                    print(f"Epoch {epoch}/{args.epochs} | "
                          f"Iter {i}/{len(train_loader)} | "
                          f"Step {global_step} | "
                          f"Loss: {loss.item() * gradient_accumulation_steps:.4f}")

            global_step += 1

        # Save checkpoint (only on rank 0)
        if (epoch + 1) % args.save_freq == 0 and rank == 0:
            # Save the unwrapped model if using DDP
            model_state = model.module.state_dict() if is_distributed else model.state_dict()
            checkpoint_path = Path(args.ckpt_dir) / f"checkpoint_{epoch:04d}.pth.tar"
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                'epoch': epoch + 1,
                'state_dict': model_state,
                'optimizer': optimizer.state_dict(),
                'scaler': scaler.state_dict(),
            }, checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")

    # Cleanup
    if rank == 0:
        if run is not None:
            run.finish()

        # Close logger
        if tee_logger is not None:
            tee_logger.close()
            # Restore stdout
            sys.stdout = tee_logger.terminal
            sys.stderr = tee_logger.terminal

    if is_distributed:
        dist.destroy_process_group()

if __name__ == '__main__':
    main()
