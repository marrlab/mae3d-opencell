"""
MAE3D with Channel Cross-Attention and Distillation Support.

Extends MAE3DChannelCrossAttention to support knowledge distillation from
precomputed teacher embeddings (e.g., SubCell).

Key additions:
- Projection layer for teacher embeddings (e.g., 1536 → 384)
- Method to extract student embeddings from CLS tokens
- Distillation loss computation (MSE or Cosine similarity)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.models.mae3d_cross_attention import MAE3DChannelCrossAttention


__all__ = ["MAE3DChannelCrossAttentionDistill"]


class MAE3DChannelCrossAttentionDistill(MAE3DChannelCrossAttention):
    """
    3D Masked Autoencoder with Channel Cross-Attention and Distillation.

    Adds distillation capability to learn from precomputed teacher embeddings
    (e.g., SubCell) while performing MAE reconstruction.

    Key additions:
    - teacher_proj: Projects teacher embeddings to student embedding dimension
    - get_student_embedding(): Extracts embedding using global pooling + concat (matching localization)
    - distillation_loss(): Computes MSE or Cosine similarity loss
    """

    def __init__(self, args):
        super().__init__(args)

        # Distillation configuration
        self.teacher_embed_dim = getattr(args, 'teacher_embed_dim', 1536)
        self.distill_loss_type = getattr(args, 'distill_loss_type', 'cosine')  # 'mse' or 'cosine'
        self.use_global_pool = getattr(args, 'distill_use_global_pool', True)
        self.pool_mode = getattr(args, 'distill_pool_mode', 'concat')
        self.distill_use_standardization = getattr(args, 'distill_use_standardization', False)
        self.distill_nonmasked = getattr(args, 'distill_nonmasked', False)

        # Calculate student embedding dimension based on pool_mode
        if self.pool_mode == 'concat':
            self.student_embed_dim = args.encoder_embed_dim * args.in_chans  # 384 * 2 = 768
        else:
            self.student_embed_dim = args.encoder_embed_dim  # 384

        # Projection layer: teacher_dim → student_embed_dim
        self.teacher_proj = nn.Linear(self.teacher_embed_dim, self.student_embed_dim)

        # Initialize projection layer
        nn.init.xavier_uniform_(self.teacher_proj.weight)
        nn.init.zeros_(self.teacher_proj.bias)

        print(f"   Distillation enabled:")
        print(f"     Teacher embed dim: {self.teacher_embed_dim}")
        print(f"     Student embed dim: {self.student_embed_dim}")
        print(f"     Use global pool: {self.use_global_pool}")
        print(f"     Pool mode: {self.pool_mode}")
        print(f"     Loss type: {self.distill_loss_type}")
        print(f"     Batch standardization: {self.distill_use_standardization}")
        print(f"     Non-masked distillation: {self.distill_nonmasked}")

    def forward_encoder(self, x):
        """
        Forward pass through encoder only.

        Returns encoded tokens for both channels including CLS tokens.
        Used for extracting embeddings for distillation.

        Args:
            x: Input tensor [B, C, H, W, D]

        Returns:
            x_ch0: Encoded tokens for channel 0 [B, 1+sel_length, embed_dim]
            x_ch1: Encoded tokens for channel 1 [B, 1+sel_length, embed_dim]
            shuffle_indices: Indices used for masking
            sel_length: Number of selected (visible) patches
        """
        from lib.models.mae3d_cross_attention import patchify_image_channelwise, batched_shuffle_indices

        args = self.args
        B = x.size(0)

        # Patchify per channel
        x_channels = patchify_image_channelwise(x, self.patch_size)

        # Masking (same mask for all channels)
        length = self.num_patches
        sel_length = int(length * (1 - args.mask_ratio))

        shuffle_indices = batched_shuffle_indices(B, length, device=x.device)

        # Apply masking to each channel
        sel_x_channels = []
        for x_ch in x_channels:
            shuffled = x_ch.gather(
                dim=1,
                index=shuffle_indices[:, :, None].expand(-1, -1, self.patch_volume)
            )
            sel_x_channels.append(shuffled[:, :sel_length, :])

        sel_indices = shuffle_indices[:, :sel_length]

        # Embed patches per channel
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

        # Process through encoder blocks with cross-attention
        x_ch0, x_ch1 = embedded_channels[0], embedded_channels[1]
        for block in self.encoder_blocks:
            x_ch0, x_ch1 = block(x_ch0, x_ch1)

        # Normalize
        x_ch0 = self.encoder_norms[0](x_ch0)
        x_ch1 = self.encoder_norms[1](x_ch1)

        return x_ch0, x_ch1, shuffle_indices, sel_length

    def get_student_embedding(self, x_ch0, x_ch1):
        """
        Extract student embedding from encoder outputs.

        Uses the same approach as ViT3DCrossAttentionClassifier:
        - If use_global_pool=True: global average pooling over spatial tokens (excluding CLS)
        - If use_global_pool=False: use CLS tokens
        - Combines features based on pool_mode: 'concat', 'mean', or 'sum'

        Args:
            x_ch0: Encoded tokens for channel 0 [B, 1+sel_length, embed_dim]
            x_ch1: Encoded tokens for channel 1 [B, 1+sel_length, embed_dim]

        Returns:
            student_emb: Combined embedding [B, student_embed_dim]
        """
        if self.use_global_pool:
            # Global average pooling over spatial tokens (skip CLS token at position 0)
            feat_ch0 = x_ch0[:, 1:, :].mean(dim=1)  # [B, embed_dim]
            feat_ch1 = x_ch1[:, 1:, :].mean(dim=1)  # [B, embed_dim]
        else:
            # Use CLS tokens
            feat_ch0 = x_ch0[:, 0, :]  # [B, embed_dim]
            feat_ch1 = x_ch1[:, 0, :]  # [B, embed_dim]

        # Combine channel features based on pool_mode
        if self.pool_mode == 'concat':
            student_emb = torch.cat([feat_ch0, feat_ch1], dim=-1)  # [B, embed_dim * 2]
        elif self.pool_mode == 'mean':
            student_emb = (feat_ch0 + feat_ch1) / 2  # [B, embed_dim]
        elif self.pool_mode == 'sum':
            student_emb = feat_ch0 + feat_ch1  # [B, embed_dim]
        else:
            raise ValueError(f"Unknown pool_mode: {self.pool_mode}")

        return student_emb

    def project_teacher_embedding(self, teacher_emb):
        """
        Project teacher embedding to student embedding dimension.

        Args:
            teacher_emb: Teacher embedding [B, teacher_embed_dim]

        Returns:
            projected: Projected embedding [B, student_embed_dim]
        """
        return self.teacher_proj(teacher_emb)

    @staticmethod
    def batch_standardize(emb):
        """Standardize embeddings to zero mean and unit variance per feature dim across the batch."""
        mean = emb.mean(dim=0, keepdim=True)
        std = emb.std(dim=0, unbiased=False, keepdim=True) + 1e-4
        return (emb - mean) / std

    def distillation_loss(self, student_emb, teacher_emb):
        """
        Compute distillation loss between student and teacher embeddings.

        Args:
            student_emb: Student embedding [B, embed_dim]
            teacher_emb: Teacher embedding (already projected) [B, embed_dim]

        Returns:
            loss: Scalar loss value
        """
        if self.distill_use_standardization:
            student_emb = self.batch_standardize(student_emb)
            teacher_emb = self.batch_standardize(teacher_emb)

        if self.distill_loss_type == 'mse':
            # Euclidean / MSE loss
            loss = F.mse_loss(student_emb, teacher_emb)
        elif self.distill_loss_type == 'cosine':
            # Cosine similarity loss (1 - cos_sim)
            # Clamp to prevent extreme values after standardization
            student_emb = student_emb.clamp(-10, 10)
            teacher_emb = teacher_emb.clamp(-10, 10)
            # Normalize embeddings
            student_norm = F.normalize(student_emb, p=2, dim=-1)
            teacher_norm = F.normalize(teacher_emb, p=2, dim=-1)
            # Cosine similarity: dot product of normalized vectors
            cos_sim = (student_norm * teacher_norm).sum(dim=-1)  # [B]
            # Loss: 1 - cos_sim (minimize to maximize similarity)
            loss = (1 - cos_sim).mean()
        else:
            raise ValueError(f"Unknown distillation loss type: {self.distill_loss_type}")

        return loss

    def forward(self, x, teacher_emb=None, return_image=False, return_distill_loss=False):
        """
        Forward pass with optional distillation.

        Args:
            x: Input tensor [B, C, H, W, D]
            teacher_emb: Optional teacher embeddings [B, teacher_embed_dim]
            return_image: If True, return reconstruction for visualization
            return_distill_loss: If True, return distillation loss separately

        Returns:
            If return_distill_loss and teacher_emb provided:
                recon_loss, distill_loss, (optional visualization outputs)
            Otherwise:
                Standard MAE outputs
        """
        from lib.models.mae3d_cross_attention import patchify_image_channelwise, batched_shuffle_indices

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

        # Apply masking to each channel
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

        # Normalize encoder outputs (for distillation)
        x_ch0_norm = self.encoder_norms[0](x_ch0)
        x_ch1_norm = self.encoder_norms[1](x_ch1)

        # Get student embedding for distillation (before projection to decoder)
        distill_loss = None
        if teacher_emb is not None and return_distill_loss:
            if self.distill_nonmasked:
                distill_loss = self._compute_nonmasked_distill(x, teacher_emb)
            else:
                student_emb = self.get_student_embedding(x_ch0_norm, x_ch1_norm)
                projected_teacher = self.project_teacher_embedding(teacher_emb)
                distill_loss = self.distillation_loss(student_emb, projected_teacher)

        # Project to decoder dimension
        x_ch0 = self.encoder_to_decoder[0](x_ch0_norm)
        x_ch1 = self.encoder_to_decoder[1](x_ch1_norm)

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

        # ============ Reconstruction Loss ============
        loss_ch0 = self.criterion(
            out_ch0[:, -msk_length:, :],
            self.patch_norm(msk_x_channels[0].detach())
        )
        loss_ch1 = self.criterion(
            out_ch1[:, -msk_length:, :],
            self.patch_norm(msk_x_channels[1].detach())
        )
        recon_loss = (loss_ch0 + loss_ch1) / 2

        # ============ Return ============
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

            if return_distill_loss and distill_loss is not None:
                return recon_loss, distill_loss, original.detach(), recon.detach(), masked.detach()
            return recon_loss, original.detach(), recon.detach(), masked.detach()

        if return_distill_loss and distill_loss is not None:
            return recon_loss, distill_loss

        return recon_loss

    def _compute_nonmasked_distill(self, x, teacher_emb):
        """
        Compute distillation loss using a non-masked encoder pass (all patches visible).

        Runs a second forward pass through the encoder with no masking, then computes
        distillation loss against the teacher embedding.

        Args:
            x: Input tensor [B, C, D, H, W]
            teacher_emb: Teacher embedding [B, teacher_embed_dim]

        Returns:
            distill_loss: Scalar loss value
        """
        from lib.models.mae3d_cross_attention import patchify_image_channelwise

        args = self.args
        B = x.size(0)

        # Patchify per channel (no masking — all patches)
        x_channels = patchify_image_channelwise(x, self.patch_size)

        # Embed all patches per channel
        embedded_channels = []
        for i, (x_ch, embed, cls_token) in enumerate(
            zip(x_channels, self.patch_embeds, self.cls_tokens)):

            x_emb = embed(x_ch)  # [B, num_patches, embed_dim]
            cls = cls_token.expand(B, -1, -1)
            x_emb = torch.cat([cls, x_emb], dim=1)  # [B, 1+num_patches, embed_dim]

            # Full positional embedding (no gathering needed — all patches present)
            cls_pe = torch.zeros(B, 1, args.encoder_embed_dim, device=x.device)
            pos_embed_full = torch.cat([cls_pe, self.encoder_pos_embed.expand(B, -1, -1)], dim=1)

            x_emb = self.pos_drop(x_emb + pos_embed_full)
            embedded_channels.append(x_emb)

        # Process through encoder blocks with cross-attention
        x_ch0, x_ch1 = embedded_channels[0], embedded_channels[1]
        for block in self.encoder_blocks:
            x_ch0, x_ch1 = block(x_ch0, x_ch1)

        # Normalize
        x_ch0 = self.encoder_norms[0](x_ch0)
        x_ch1 = self.encoder_norms[1](x_ch1)

        # Extract student embedding and compute loss
        student_emb = self.get_student_embedding(x_ch0, x_ch1)
        projected_teacher = self.project_teacher_embedding(teacher_emb)
        return self.distillation_loss(student_emb, projected_teacher)
