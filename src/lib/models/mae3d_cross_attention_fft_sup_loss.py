"""
MAE3D with Channel Cross-Attention, FFT Loss, and Supervised Classification Loss.

Extends MAE3DChannelCrossAttentionFFT to add a protein classification head on top
of encoder features. This encourages the encoder to learn protein-discriminative
representations alongside reconstruction.

The classification head pools visible (non-masked) encoder tokens from both channels
via global average pooling, concatenates them (768-dim), and maps to num_classes logits.

Important: Validation proteins have 0% overlap with training proteins, so the
supervised loss is only applied during training (when labels are provided).
"""

import torch
import torch.nn as nn

from lib.models.mae3d_cross_attention_fft import MAE3DChannelCrossAttentionFFT
from lib.models.mae3d_cross_attention import (
    patchify_image_channelwise,
    batched_shuffle_indices,
)


__all__ = ["MAE3DChannelCrossAttentionFFTSupLoss"]


class MAE3DChannelCrossAttentionFFTSupLoss(MAE3DChannelCrossAttentionFFT):
    """
    3D Masked Autoencoder with Channel Cross-Attention, FFT Loss,
    and Supervised Protein Classification Loss.

    Adds a linear classification head on pooled encoder features to predict
    which protein is in the image (1048 classes for train set).
    """

    def __init__(self, args):
        super().__init__(args)

        # Supervised classification head
        self.num_classes = getattr(args, 'num_classes', 1048)
        self.sup_loss_weight = getattr(args, 'sup_loss_weight', 0.1)
        self.sup_pool_mode = getattr(args, 'sup_pool_mode', 'concat')
        label_smoothing = getattr(args, 'sup_loss_label_smoothing', 0.1)

        if self.sup_pool_mode == 'concat':
            cls_input_dim = args.encoder_embed_dim * 2  # both channels
        else:
            cls_input_dim = args.encoder_embed_dim

        self.cls_head = nn.Linear(cls_input_dim, self.num_classes)
        self.sup_criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

        print(f"   Supervised classification head: {cls_input_dim} -> {self.num_classes}")
        print(f"   Sup loss weight: {self.sup_loss_weight}, pool: {self.sup_pool_mode}")
        print(f"   Label smoothing: {label_smoothing}")

    def forward(self, x, return_image=False, protein_labels=None):
        """
        Forward pass with channel-wise processing, cross-attention, FFT loss,
        and optional supervised classification loss.

        Args:
            x: Input tensor [B, C, H, W, D]
            return_image: If True, also return reconstruction for visualization
            protein_labels: Optional [B] integer tensor of protein class indices.
                           If provided, computes supervised classification loss.

        Returns:
            loss: Combined MSE + FFT + sup reconstruction loss
            loss_mse: MSE loss only (for logging)
            loss_fft: FFT loss only (for logging)
            loss_sup: Supervised loss only (for logging), 0 if no labels
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

        # ============ Supervised Classification (before decoder) ============
        loss_sup = torch.tensor(0.0, device=x.device)
        if protein_labels is not None:
            # Pool visible encoder tokens (skip CLS at index 0)
            feat_ch0 = x_ch0[:, 1:, :].mean(dim=1)  # [B, 384]
            feat_ch1 = x_ch1[:, 1:, :].mean(dim=1)  # [B, 384]

            if self.sup_pool_mode == 'concat':
                feat = torch.cat([feat_ch0, feat_ch1], dim=-1)  # [B, 768]
            else:
                feat = (feat_ch0 + feat_ch1) / 2  # [B, 384]

            logits = self.cls_head(feat)  # [B, num_classes]
            loss_sup = self.sup_criterion(logits, protein_labels)

        # ============ Encoder to Decoder projection ============
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
            recon_ch0 = self._reconstruct_full_image_per_channel(
                out_ch0, x_channels[0], unshuffle_indices
            )
            recon_ch1 = self._reconstruct_full_image_per_channel(
                out_ch1, x_channels[1], unshuffle_indices
            )

            orig_ch0 = self._get_original_image_per_channel(x_channels[0])
            orig_ch1 = self._get_original_image_per_channel(x_channels[1])

            fft_loss_ch0 = self.fft_criterion(recon_ch0, orig_ch0.detach())
            fft_loss_ch1 = self.fft_criterion(recon_ch1, orig_ch1.detach())
            loss_fft = (fft_loss_ch0 + fft_loss_ch1) / 2

        # ============ Combined Loss ============
        loss = loss_mse + self._current_fft_weight * loss_fft + self.sup_loss_weight * loss_sup

        if return_image:
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

            return loss, loss_mse, loss_fft, loss_sup, original.detach(), recon.detach(), masked.detach()

        return loss, loss_mse, loss_fft, loss_sup
