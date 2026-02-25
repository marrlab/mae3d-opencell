"""
Trainer for MAE3D with Channel Cross-Attention.

Extends the standard MAE3DTrainer with support for the channel cross-attention model.
"""

import os
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import wandb

from .mae3d_trainer import MAE3DTrainer
from lib.models.mae3d_cross_attention import MAE3DChannelCrossAttention


class MAE3DChannelCrossAttentionTrainer(MAE3DTrainer):
    """
    Trainer for MAE3D with Channel Cross-Attention.

    Key differences from MAE3DTrainer:
    - Uses MAE3DChannelCrossAttention model instead of MAE3D
    - No encoder/decoder classes needed (model handles this internally)
    """

    def __init__(self, args):
        super().__init__(args)
        self.model_name = 'MAE3DChannelCrossAttention'

    def build_model(self):
        """
        Build MAE3D model with channel cross-attention.
        """
        if self.model is not None:
            raise ValueError("Model has been created. Do not create twice")

        args = self.args
        print(f"=> creating model {self.model_name}")
        print(f"   Cross-attention enabled: True")
        print(f"   Channels: {args.in_chans}")
        print(f"   Encoder depth: {args.encoder_depth}")
        print(f"   Decoder depth: {args.decoder_depth}")
        print(f"   Encoder embed dim: {args.encoder_embed_dim}")
        print(f"   Decoder embed dim: {args.decoder_embed_dim}")

        # Create model (no separate encoder/decoder needed)
        self.model = MAE3DChannelCrossAttention(args=args)

        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"   Total parameters: {total_params:,}")
        print(f"   Trainable parameters: {trainable_params:,}")

        # Wrap with DDP
        self.wrap_model()

    def visualize_mae_reconstruction(self, original_patches, masked_patches, recon_patches,
                                     z_slices=[20, 50, 80]):
        """
        Create visualization for channel cross-attention model.

        The patches are concatenated per channel (channel-last), so we need to
        handle them correctly.

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
        # Shape after unpatchify: [C, Z, Y, X]
        original = original[0].cpu().numpy()
        masked = masked[0].cpu().numpy()
        recon = recon[0].cpu().numpy()

        # Create figure with GridSpec for equal column spacing + colorbar column
        n_slices = len(z_slices)
        n_channels = args.in_chans
        n_rows = n_channels * n_slices

        fig = plt.figure(figsize=(16, 2.5 * n_rows), facecolor='white')
        # 3 equal image columns + 1 narrow colorbar column
        gs = GridSpec(n_rows, 4, figure=fig, width_ratios=[1, 1, 1, 0.05], wspace=0.08, hspace=0.15)

        # Overall title
        fig.suptitle(f'MAE 3D Cross-Attention Reconstruction | Shape: {tuple(args.input_size)} | Mask: {args.mask_ratio:.1%}',
                     fontsize=22, fontweight='bold', y=0.995)

        col_titles = ['Original', 'Encoder Input (masked)', 'Reconstructed']
        channel_names = ['Nucleus', 'Protein'] if n_channels == 2 else [f'Ch{i}' for i in range(n_channels)]

        # Channel colormaps: nucleus → black-to-blue (DAPI style), protein → grayscale
        import matplotlib.colors as mcolors
        nucleus_cmap = mcolors.LinearSegmentedColormap.from_list('nucleus', ['black', '#4488ff'])
        channel_cmaps = [nucleus_cmap, 'gray'] if n_channels == 2 else ['gray'] * n_channels

        for ch in range(n_channels):
            for slice_idx, z_slice in enumerate(z_slices):
                row_idx = ch * n_slices + slice_idx

                # Extract z-slices for this row (all three columns)
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

                    if row_idx == 0:
                        ax.set_title(title, fontsize=18, fontweight='bold', pad=10)

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

                    ax.set_xticks([])
                    ax.set_yticks([])

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
        Unpatchify for cross-attention model.

        The cross-attention model outputs patches with channels concatenated:
        [B, num_patches, patch_volume * C]

        Args:
            patches: [B, num_patches, patch_volume * C]

        Returns:
            [B, C, Z, Y, X]
        """
        args = self.args
        B = patches.shape[0]
        H, W, D = args.input_size  # Z, Y, X
        ph, pw, pd = args.patch_size
        gh, gw, gd = H // ph, W // pw, D // pd
        in_chans = args.in_chans
        patch_volume = ph * pw * pd

        # Split channels from the patch dimension
        # patches: [B, num_patches, patch_volume * C]
        # -> [B, num_patches, C, patch_volume]
        patches = patches.reshape(B, gh * gw * gd, in_chans, patch_volume)

        # Rearrange to spatial
        # [B, gh, gw, gd, C, patch_volume]
        x = patches.reshape(B, gh, gw, gd, in_chans, ph, pw, pd)
        # [B, C, gh, ph, gw, pw, gd, pd]
        x = x.permute(0, 4, 1, 5, 2, 6, 3, 7)
        # [B, C, H, W, D]
        x = x.reshape(B, in_chans, H, W, D)

        return x
