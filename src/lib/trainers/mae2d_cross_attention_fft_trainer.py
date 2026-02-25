"""
Trainer for MAE2D with Channel Cross-Attention and FFT Loss.

Extends MAE3DChannelCrossAttentionFFTTrainer with:
- 2D model (MAE2DChannelCrossAttentionFFT)
- Max-projection dataloader (use_max_projection=True)
- 2D reconstruction visualisation (one row per channel, no z-slices)
  with blue (nucleus) and gray (protein) colormaps
"""

import os
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import wandb

from torch.utils.data import DistributedSampler

from .mae3d_cross_attention_fft_trainer import MAE3DChannelCrossAttentionFFTTrainer
from lib.models.mae2d_cross_attention_fft import MAE2DChannelCrossAttentionFFT
from data.opencell.dataset import OpenCellDataset
from data.opencell.transforms import get_opencell_2d_train_transforms, get_opencell_2d_val_transforms


class MAE2DChannelCrossAttentionFFTTrainer(MAE3DChannelCrossAttentionFFTTrainer):
    """
    Trainer for MAE2D with Channel Cross-Attention and FFT Loss.

    Inherits all FFT scheduling and training logic from the 3D FFT trainer;
    overrides model creation, dataloader (max-projection 2D), and visualisation.
    """

    def __init__(self, args):
        super().__init__(args)
        self.model_name = 'MAE2DChannelCrossAttentionFFT'

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------

    def build_model(self):
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
        print(f"   Input size (2D): {args.input_size}")
        print(f"   Patch size: {args.patch_size}")

        self.model = MAE2DChannelCrossAttentionFFT(args=args)

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"   Total parameters: {total_params:,}")
        print(f"   Trainable parameters: {trainable_params:,}")

        self.wrap_model()

    # ------------------------------------------------------------------
    # Dataloader  (max-projection 2D, mirrors MAE2DTrainer.build_dataloader)
    # ------------------------------------------------------------------

    def build_dataloader(self):
        if self.train_loader is not None:
            raise ValueError("Dataloader has been created. Do not create twice.")

        print("=> creating dataloaders (2D max-projection)")
        args = self.args

        train_csv_path = os.path.join(args.csv_path, "train.csv")
        train_transform = get_opencell_2d_train_transforms(
            flip_prob=args.RandFlipd_prob,
            rotate_prob=args.RandRotate90d_prob
        )

        train_dataset = OpenCellDataset(
            csv_path=train_csv_path,
            split='train',
            transform=train_transform,
            cache_rate=args.cache_rate,
            num_workers=args.workers,
            use_max_projection=True
        )

        train_workers = 0 if self.is_distributed else args.workers

        if self.is_distributed:
            sampler = DistributedSampler(
                train_dataset, num_replicas=self.world_size,
                rank=self.rank, shuffle=True
            )
            self.train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=args.batch_size,
                sampler=sampler, num_workers=train_workers, pin_memory=True
            )
        else:
            self.train_loader = train_dataset.get_dataloader(
                batch_size=args.batch_size, shuffle=True, num_workers=train_workers
            )

        print(f"   Dataset: OpenCell (2D max-projection)")
        print(f"   Train samples: {len(train_dataset)}")
        print(f"   Batch size: {args.batch_size}")
        print(f"   Iterations per epoch: {len(self.train_loader)}")

        # Validation
        val_csv_path = os.path.join(args.csv_path, "val.csv")
        if os.path.exists(val_csv_path):
            val_transform = get_opencell_2d_val_transforms()
            val_dataset = OpenCellDataset(
                csv_path=val_csv_path,
                split='val',
                transform=val_transform,
                cache_rate=0.0,
                num_workers=args.workers,
                use_max_projection=True
            )

            if self.is_distributed:
                val_sampler = DistributedSampler(
                    val_dataset, num_replicas=self.world_size,
                    rank=self.rank, shuffle=False
                )
                self.val_loader = torch.utils.data.DataLoader(
                    val_dataset, batch_size=args.batch_size,
                    sampler=val_sampler, num_workers=0, pin_memory=True
                )
            else:
                self.val_loader = val_dataset.get_dataloader(
                    batch_size=args.batch_size, shuffle=False, num_workers=0
                )

            print(f"   Val samples: {len(val_dataset)}")
            print(f"   Validation enabled: True\n")
        else:
            print(f"   Validation file not found: {val_csv_path}")
            print(f"   Validation enabled: False\n")

        # Fixed visualisation sample
        if self.rank == 0:
            for batch in self.train_loader:
                self.fixed_vis_sample = batch['image'][0:1].cuda(self.local_rank, non_blocking=True)
                print(f"=> Fixed vis sample shape: {self.fixed_vis_sample.shape}")
                break

    # ------------------------------------------------------------------
    # 2D unpatchify  [B, num_patches, patch_area*C] → [B, C, H, W]
    # ------------------------------------------------------------------

    def unpatchify_image(self, patches):
        args = self.args
        B = patches.shape[0]
        H, W = args.input_size
        pH, pW = args.patch_size
        gH, gW = H // pH, W // pW
        C = args.in_chans
        patch_area = pH * pW

        # [B, gH*gW, C*patch_area] → [B, gH*gW, C, patch_area]
        patches = patches.reshape(B, gH * gW, C, patch_area)
        # [B, gH, gW, C, pH, pW]
        x = patches.reshape(B, gH, gW, C, pH, pW)
        # [B, C, gH, pH, gW, pW]
        x = x.permute(0, 3, 1, 4, 2, 5)
        # [B, C, H, W]
        x = x.reshape(B, C, H, W)
        return x

    # ------------------------------------------------------------------
    # 2D visualisation
    # ------------------------------------------------------------------

    def visualize_mae_reconstruction(self, original_patches, masked_patches,
                                     recon_patches, **kwargs):
        """
        2D reconstruction visualisation: one row per channel.

        Nucleus → black-to-blue colormap
        Protein → grayscale
        """
        from matplotlib.gridspec import GridSpec

        args = self.args

        original = self.unpatchify_image(original_patches[:1])[0].cpu().numpy()  # [C, H, W]
        masked   = self.unpatchify_image(masked_patches[:1])[0].cpu().numpy()
        recon    = self.unpatchify_image(recon_patches[:1])[0].cpu().numpy()

        n_channels = args.in_chans
        nucleus_cmap = mcolors.LinearSegmentedColormap.from_list('nucleus', ['black', '#4488ff'])
        channel_cmaps = [nucleus_cmap, 'gray'] if n_channels == 2 else ['gray'] * n_channels
        channel_names = ['Nucleus', 'Protein'] if n_channels == 2 else [f'Ch{i}' for i in range(n_channels)]

        fig = plt.figure(figsize=(16, 5 * n_channels), facecolor='white')
        gs = GridSpec(n_channels, 4, figure=fig,
                      width_ratios=[1, 1, 1, 0.05], wspace=0.08, hspace=0.15)

        fig.suptitle(
            f'MAE 2D Cross-Attention Reconstruction | '
            f'Shape: {tuple(args.input_size)} (H,W) | Mask: {args.mask_ratio:.1%}',
            fontsize=22, fontweight='bold', y=0.98
        )

        col_titles = ['Original', 'Encoder Input (masked)', 'Reconstructed']

        for ch in range(n_channels):
            orig_sl   = original[ch]
            masked_sl = masked[ch]
            recon_sl  = recon[ch]

            all_vals = np.concatenate([orig_sl.flatten(), masked_sl.flatten(), recon_sl.flatten()])
            vmin, vmax = np.percentile(all_vals, [1, 99])

            im = None
            for col_idx, (img, title) in enumerate(
                    zip([orig_sl, masked_sl, recon_sl], col_titles)):
                ax = fig.add_subplot(gs[ch, col_idx])

                im = ax.imshow(img, cmap=channel_cmaps[ch],
                               vmin=vmin, vmax=vmax,
                               interpolation='bilinear', aspect='equal')

                if ch == 0:
                    ax.set_title(title, fontsize=18, fontweight='bold', pad=10)
                if col_idx == 0:
                    ax.set_ylabel(channel_names[ch], fontsize=14, fontweight='bold',
                                  rotation=0, ha='right', va='center', labelpad=55)

                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_edgecolor('gray')
                    spine.set_linewidth(0.5)

            cbar_ax = fig.add_subplot(gs[ch, 3])
            cbar = plt.colorbar(im, cax=cbar_ax)
            cbar.ax.tick_params(labelsize=12)

        wandb_image = wandb.Image(fig)
        plt.close(fig)
        return wandb_image
