"""
MAE3D with Channel Cross-Attention, FFT Loss, and External Embedding Concatenation in Decoder.

Extends MAE3DChannelCrossAttentionFFT to inject external embeddings (ESM2, SubCell)
as additional tokens in the decoder. This allows the decoder to attend to multi-modal
context during reconstruction.

Architecture:
    Image -> Encoder -> [B, 1+sel_length, enc_dim] per channel
                         |
                         v (project to decoder dim)
                 [B, 1+sel_length, dec_dim] + mask_tokens -> [B, 1+num_patches, dec_dim]
                         |
    External embeddings -> project -> [B, K, dec_dim]
                         |
                         v (concatenate)
                 [B, 1+num_patches+K, dec_dim] per channel
                         |
                         v (decoder with position-wise cross-attention)
                 Reconstruct -> Loss on masked patches only
"""

import torch
import torch.nn as nn
import numpy as np
from timm.models.layers.helpers import to_3tuple

from lib.models.mae3d_cross_attention_fft import MAE3DChannelCrossAttentionFFT
from lib.models.mae3d_cross_attention import (
    patchify_image_channelwise,
    unpatchify_image_channelwise,
    batched_shuffle_indices
)


__all__ = ["MAE3DChannelCrossAttentionConcat"]


class MAE3DChannelCrossAttentionConcat(MAE3DChannelCrossAttentionFFT):
    """
    3D Masked Autoencoder with Channel Cross-Attention, FFT Loss, and
    External Embedding Concatenation in Decoder.

    External embeddings (e.g., ESM2 protein embeddings, SubCell embeddings) are
    projected and appended as additional tokens to each channel's decoder input.
    Both channels receive identical external tokens to maintain compatibility
    with position-wise cross-attention.

    Decoder sequence layout:
        [CLS | visible_tokens | mask_tokens | external_tokens]
         0     1..sel_length   sel+1..num_patches  num_patches+1..num_patches+K
    """

    def __init__(self, args):
        super().__init__(args)

        dec_embed_dim = args.decoder_embed_dim

        # ESM2 embedding concatenation
        self.concat_esm2 = getattr(args, 'concat_esm2', False)
        self.esm2_embed_dim = getattr(args, 'esm2_embed_dim', 1280)
        self.num_esm2_tokens = getattr(args, 'num_esm2_tokens', 1)

        if self.concat_esm2:
            self.esm2_proj = nn.Linear(self.esm2_embed_dim, self.num_esm2_tokens * dec_embed_dim)
            self.esm2_pos_embed = nn.Parameter(
                torch.zeros(1, self.num_esm2_tokens, dec_embed_dim)
            )
            nn.init.normal_(self.esm2_pos_embed, std=0.02)
            print(f"   Concat ESM2: dim={self.esm2_embed_dim} -> {self.num_esm2_tokens} tokens of {dec_embed_dim}")

        # SubCell embedding concatenation
        self.concat_subcell = getattr(args, 'concat_subcell', False)
        self.subcell_embed_dim = getattr(args, 'subcell_embed_dim', 1536)
        self.num_subcell_tokens = getattr(args, 'num_subcell_tokens', 1)

        if self.concat_subcell:
            self.subcell_proj = nn.Linear(self.subcell_embed_dim, self.num_subcell_tokens * dec_embed_dim)
            self.subcell_pos_embed = nn.Parameter(
                torch.zeros(1, self.num_subcell_tokens, dec_embed_dim)
            )
            nn.init.normal_(self.subcell_pos_embed, std=0.02)
            print(f"   Concat SubCell: dim={self.subcell_embed_dim} -> {self.num_subcell_tokens} tokens of {dec_embed_dim}")

        # Total external tokens
        self.num_external_tokens = 0
        if self.concat_esm2:
            self.num_external_tokens += self.num_esm2_tokens
        if self.concat_subcell:
            self.num_external_tokens += self.num_subcell_tokens

        print(f"   Total external decoder tokens: {self.num_external_tokens}")

    def _project_external_embeddings(self, esm2_emb=None, subcell_emb=None):
        """
        Project and concatenate external embeddings into decoder tokens.

        Args:
            esm2_emb: ESM2 embeddings [B, esm2_embed_dim] or None
            subcell_emb: SubCell embeddings [B, subcell_embed_dim] or None

        Returns:
            External tokens [B, K_total, dec_embed_dim] or None if no embeddings
        """
        dec_embed_dim = self.args.decoder_embed_dim
        tokens = []

        if self.concat_esm2 and esm2_emb is not None:
            B = esm2_emb.shape[0]
            proj = self.esm2_proj(esm2_emb)  # [B, num_esm2_tokens * dec_embed_dim]
            proj = proj.reshape(B, self.num_esm2_tokens, dec_embed_dim)
            proj = proj + self.esm2_pos_embed
            tokens.append(proj)

        if self.concat_subcell and subcell_emb is not None:
            B = subcell_emb.shape[0]
            proj = self.subcell_proj(subcell_emb)  # [B, num_subcell_tokens * dec_embed_dim]
            proj = proj.reshape(B, self.num_subcell_tokens, dec_embed_dim)
            proj = proj + self.subcell_pos_embed
            tokens.append(proj)

        if len(tokens) == 0:
            return None

        return torch.cat(tokens, dim=1)  # [B, K_total, dec_embed_dim]

    def _reconstruct_full_image_per_channel(
        self,
        out_ch: torch.Tensor,
        x_ch: torch.Tensor,
        unshuffle_indices: torch.Tensor
    ) -> torch.Tensor:
        """
        Reconstruct full image for a single channel from decoder output.
        Overrides base to exclude external tokens from reconstruction.

        Args:
            out_ch: Decoder output [B, 1+num_patches+K, patch_volume] (includes CLS and external)
            x_ch: Original patches [B, num_patches, patch_volume] for denormalization
            unshuffle_indices: Indices to unshuffle patches back to original order

        Returns:
            Reconstructed volume [B, Z, Y, X]
        """
        B = out_ch.shape[0]

        # Slice to image tokens only (skip CLS, exclude external tokens)
        image_tokens = out_ch[:, 1:1 + self.num_patches, :]

        # Unshuffle reconstruction
        recon = image_tokens.gather(
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

    def forward(self, x, esm2_emb=None, subcell_emb=None, return_image=False):
        """
        Forward pass with channel-wise processing, cross-attention, FFT loss,
        and external embedding concatenation in decoder.

        Args:
            x: Input tensor [B, C, H, W, D]
            esm2_emb: ESM2 protein embeddings [B, esm2_embed_dim] or None
            subcell_emb: SubCell embeddings [B, subcell_embed_dim] or None
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
        # Combine visible tokens with mask tokens per channel
        dec_channels = []
        for x_enc, mask_token in zip([x_ch0, x_ch1], self.mask_tokens):
            all_x = torch.cat([
                x_enc,
                mask_token.expand(B, msk_length, -1)
            ], dim=1)
            dec_channels.append(all_x)

        # Add shuffled position embeddings to image tokens (skip CLS position)
        shuffled_dec_pos = self.decoder_pos_embed.expand(B, -1, -1).gather(
            dim=1,
            index=shuffle_indices[:, :, None].expand(-1, -1, args.decoder_embed_dim)
        )
        for i in range(len(dec_channels)):
            dec_channels[i][:, 1:, :] = dec_channels[i][:, 1:, :] + shuffled_dec_pos

        # ============ Append external tokens ============
        external_tokens = self._project_external_embeddings(esm2_emb, subcell_emb)
        if external_tokens is not None:
            for i in range(len(dec_channels)):
                dec_channels[i] = torch.cat([dec_channels[i], external_tokens], dim=1)

        # ============ Decoder blocks ============
        x_ch0, x_ch1 = dec_channels[0], dec_channels[1]
        for block in self.decoder_blocks:
            x_ch0, x_ch1 = block(x_ch0, x_ch1)

        # Slice to image tokens only before prediction head
        # Sequence: [CLS | visible | mask | external]
        # Image tokens are at positions 0..num_patches (inclusive of CLS)
        x_ch0_img = x_ch0[:, :1 + self.num_patches, :]
        x_ch1_img = x_ch1[:, :1 + self.num_patches, :]

        out_ch0 = self.decoder_heads[0](self.decoder_norms[0](x_ch0_img))
        out_ch1 = self.decoder_heads[1](self.decoder_norms[1](x_ch1_img))

        # ============ MSE Loss (on masked patches only) ============
        # Masked patches are at positions 1+sel_length through 1+num_patches (exclusive)
        loss_ch0_mse = self.criterion(
            out_ch0[:, 1 + sel_length:1 + self.num_patches, :],
            self.patch_norm(msk_x_channels[0].detach())
        )
        loss_ch1_mse = self.criterion(
            out_ch1[:, 1 + sel_length:1 + self.num_patches, :],
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

                # Slice to image tokens only (skip CLS, exclude external)
                image_tokens = out_ch[:, 1:1 + self.num_patches, :]

                recon = image_tokens.gather(
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
