"""
MAE3D with Channel Cross-Attention and FFT Loss.

Extends MAE3DChannelCrossAttention to include FFT loss on full reconstructions.
The FFT loss encourages the model to learn proper frequency distributions,
helping with global structure (low freq) and sharp edges (high freq).

FFT Loss Schedule:
- During warmup epochs: No FFT loss (only MSE)
- FFT ramp-up epoch: Gradually increase FFT loss weight from 0 to target
- Remaining epochs: Full FFT loss weight
"""

import torch
import torch.nn as nn
import numpy as np
from timm.models.layers.helpers import to_3tuple

from lib.models.mae3d_cross_attention import (
    MAE3DChannelCrossAttention,
    patchify_image_channelwise,
    unpatchify_image_channelwise,
    batched_shuffle_indices
)
from lib.losses.fft3d_loss import FFT3DLoss


__all__ = ["MAE3DChannelCrossAttentionFFT"]


class MAE3DChannelCrossAttentionFFT(MAE3DChannelCrossAttention):
    """
    3D Masked Autoencoder with Channel Cross-Attention and FFT Loss.

    Extends the base cross-attention model with:
    - FFT loss on full reconstructed images (not just masked patches)
    - Configurable FFT loss weight with schedule support
    - Per-channel FFT loss computation

    The FFT loss weight is controlled externally by the trainer based on epoch.
    """

    def __init__(self, args):
        super().__init__(args)

        # FFT loss settings
        self.use_fft_loss = getattr(args, 'use_fft_loss', True)
        self.fft_loss_weight = getattr(args, 'fft_loss_weight', 0.1)
        self.fft_use_log = getattr(args, 'fft_use_log', True)

        # Initialize FFT loss criterion
        if self.use_fft_loss:
            self.fft_criterion = FFT3DLoss(
                use_log=self.fft_use_log,
                norm_type='ortho'
            )
            print(f"   FFT Loss enabled: weight={self.fft_loss_weight}, use_log={self.fft_use_log}")

        # Current FFT weight (modified by trainer during training)
        self._current_fft_weight = 0.0

    def set_fft_weight(self, weight: float):
        """Set current FFT loss weight (called by trainer based on schedule)."""
        self._current_fft_weight = weight

    def get_fft_weight(self) -> float:
        """Get current FFT loss weight."""
        return self._current_fft_weight

    def _reconstruct_full_image_per_channel(
        self,
        out_ch: torch.Tensor,
        x_ch: torch.Tensor,
        unshuffle_indices: torch.Tensor
    ) -> torch.Tensor:
        """
        Reconstruct full image for a single channel from decoder output.

        Args:
            out_ch: Decoder output [B, 1+num_patches, patch_volume] (includes CLS)
            x_ch: Original patches [B, num_patches, patch_volume] for denormalization
            unshuffle_indices: Indices to unshuffle patches back to original order

        Returns:
            Reconstructed volume [B, Z, Y, X]
        """
        B = out_ch.shape[0]

        # Unshuffle reconstruction (skip CLS token)
        recon = out_ch[:, 1:, :].gather(
            dim=1,
            index=unshuffle_indices[:, :, None].expand(-1, -1, self.patch_volume)
        )

        # Denormalize using original patch statistics
        recon = recon * (x_ch.var(dim=-1, unbiased=True, keepdim=True).sqrt() + 1e-6)
        recon = recon + x_ch.mean(dim=-1, keepdim=True)

        # Unpatchify to get full volume [B, Z, Y, X]
        recon_volume = unpatchify_image_channelwise(
            [recon],
            self.patch_size,
            self.grid_size
        )[:, 0]  # Remove channel dim since single channel

        return recon_volume

    def _get_original_image_per_channel(self, x_ch: torch.Tensor) -> torch.Tensor:
        """
        Get original image for a single channel from patches.

        Args:
            x_ch: Original patches [B, num_patches, patch_volume]

        Returns:
            Original volume [B, Z, Y, X]
        """
        orig_volume = unpatchify_image_channelwise(
            [x_ch],
            self.patch_size,
            self.grid_size
        )[:, 0]  # Remove channel dim

        return orig_volume

    def forward(self, x, return_image=False):
        """
        Forward pass with channel-wise processing, cross-attention, and FFT loss.

        Args:
            x: Input tensor [B, C, H, W, D]
            return_image: If True, also return reconstruction for visualization

        Returns:
            loss: Combined MSE + FFT reconstruction loss
            loss_mse: MSE loss only (for logging)
            loss_fft: FFT loss only (for logging), or 0 if FFT disabled
            (optional) original_patches, recon_patches, masked_patches for visualization
        """
        args = self.args
        B = x.size(0)
        C = x.size(1)
        assert C == args.in_chans, f"Expected {args.in_chans} channels, got {C}"

        # ============ Patchify per channel ============
        x_channels = patchify_image_channelwise(x, self.patch_size)

        # ============ Masking (same mask for all channels) ============
        length = self.num_patches
        sel_length = int(length * (1 - args.mask_ratio))
        msk_length = length - sel_length

        shuffle_indices = batched_shuffle_indices(B, length, device=x.device)
        unshuffle_indices = shuffle_indices.argsort(dim=1)

        sel_x_channels = []
        msk_x_channels = []
        for x_ch in x_channels:
            shuffled = x_ch.gather(
                dim=1,
                index=shuffle_indices[:, :, None].expand(-1, -1, self.patch_volume)
            )
            sel_x_channels.append(shuffled[:, :sel_length, :])
            msk_x_channels.append(shuffled[:, -msk_length:, :])

        sel_indices = shuffle_indices[:, :sel_length]

        # ============ Encoder ============
        embedded_channels = []
        for i, (sel_x, embed, cls_token) in enumerate(
            zip(sel_x_channels, self.patch_embeds, self.cls_tokens)):

            x_emb = embed(sel_x)
            cls = cls_token.expand(B, -1, -1)
            x_emb = torch.cat([cls, x_emb], dim=1)

            sel_pos_embed = self.encoder_pos_embed.expand(B, -1, -1).gather(
                dim=1,
                index=sel_indices[:, :, None].expand(-1, -1, args.encoder_embed_dim)
            )
            cls_pe = torch.zeros(B, 1, args.encoder_embed_dim, device=x.device)
            pos_embed_full = torch.cat([cls_pe, sel_pos_embed], dim=1)

            x_emb = self.pos_drop(x_emb + pos_embed_full)
            embedded_channels.append(x_emb)

        x_ch0, x_ch1 = embedded_channels[0], embedded_channels[1]
        for block in self.encoder_blocks:
            x_ch0, x_ch1 = block(x_ch0, x_ch1)

        x_ch0 = self.encoder_to_decoder[0](self.encoder_norms[0](x_ch0))
        x_ch1 = self.encoder_to_decoder[1](self.encoder_norms[1](x_ch1))

        # ============ Decoder ============
        dec_channels = []
        for x_enc, mask_token in zip([x_ch0, x_ch1], self.mask_tokens):
            all_x = torch.cat([
                x_enc,
                mask_token.expand(B, msk_length, -1)
            ], dim=1)
            dec_channels.append(all_x)

        shuffled_dec_pos = self.decoder_pos_embed.expand(B, -1, -1).gather(
            dim=1,
            index=shuffle_indices[:, :, None].expand(-1, -1, args.decoder_embed_dim)
        )
        for i in range(len(dec_channels)):
            dec_channels[i][:, 1:, :] = dec_channels[i][:, 1:, :] + shuffled_dec_pos

        x_ch0, x_ch1 = dec_channels[0], dec_channels[1]
        for block in self.decoder_blocks:
            x_ch0, x_ch1 = block(x_ch0, x_ch1)

        out_ch0 = self.decoder_heads[0](self.decoder_norms[0](x_ch0))
        out_ch1 = self.decoder_heads[1](self.decoder_norms[1](x_ch1))

        # ============ MSE Loss (on masked patches only) ============
        loss_ch0_mse = self.criterion(
            out_ch0[:, -msk_length:, :],
            self.patch_norm(msk_x_channels[0].detach())
        )
        loss_ch1_mse = self.criterion(
            out_ch1[:, -msk_length:, :],
            self.patch_norm(msk_x_channels[1].detach())
        )
        loss_mse = (loss_ch0_mse + loss_ch1_mse) / 2

        # ============ FFT Loss (on full reconstruction) ============
        loss_fft = torch.tensor(0.0, device=x.device)

        if self.use_fft_loss and self._current_fft_weight > 0:
            # Reconstruct full images for FFT loss
            recon_ch0 = self._reconstruct_full_image_per_channel(
                out_ch0, x_channels[0], unshuffle_indices
            )
            recon_ch1 = self._reconstruct_full_image_per_channel(
                out_ch1, x_channels[1], unshuffle_indices
            )

            # Get original images
            orig_ch0 = self._get_original_image_per_channel(x_channels[0])
            orig_ch1 = self._get_original_image_per_channel(x_channels[1])

            # Compute FFT loss per channel
            fft_loss_ch0 = self.fft_criterion(recon_ch0, orig_ch0.detach())
            fft_loss_ch1 = self.fft_criterion(recon_ch1, orig_ch1.detach())
            loss_fft = (fft_loss_ch0 + fft_loss_ch1) / 2

        # ============ Combined Loss ============
        loss = loss_mse + self._current_fft_weight * loss_fft

        if return_image:
            # Reconstruct full patches for visualization
            recon_channels = []
            masked_channels = []
            original_channels = []

            for out_ch, sel_x, x_ch in zip(
                [out_ch0, out_ch1], sel_x_channels, x_channels):

                recon = out_ch[:, 1:, :].gather(
                    dim=1,
                    index=unshuffle_indices[:, :, None].expand(-1, -1, self.patch_volume)
                )
                recon = recon * (x_ch.var(dim=-1, unbiased=True, keepdim=True).sqrt() + 1e-6)
                recon = recon + x_ch.mean(dim=-1, keepdim=True)
                recon_channels.append(recon)

                shuffled_visible = x_ch.gather(
                    dim=1,
                    index=shuffle_indices[:, :, None].expand(-1, -1, self.patch_volume)
                )
                masked = torch.cat([
                    shuffled_visible[:, :sel_length, :],
                    torch.zeros(B, msk_length, self.patch_volume, device=x.device)
                ], dim=1).gather(
                    dim=1,
                    index=unshuffle_indices[:, :, None].expand(-1, -1, self.patch_volume)
                )
                masked_channels.append(masked)
                original_channels.append(x_ch)

            original = torch.cat(original_channels, dim=-1)
            recon = torch.cat(recon_channels, dim=-1)
            masked = torch.cat(masked_channels, dim=-1)

            return loss, loss_mse, loss_fft, original.detach(), recon.detach(), masked.detach()

        return loss, loss_mse, loss_fft
