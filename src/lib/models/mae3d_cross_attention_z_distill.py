"""
MAE3D with Channel Cross-Attention and Z-Aware Attention Distillation.

Extends MAE3DChannelCrossAttentionDistill with a perceiver-style attention
distillation head that is aware of z-depth structure.

Key additions:
- AttentionDistillationHead: Perceiver cross-attention with z-level and channel embeddings
- Combined loss: alpha * attention_distill + (1-alpha) * global_pool_distill
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.models.mae3d_cross_attention_distill import MAE3DChannelCrossAttentionDistill
from lib.networks.cross_attention import CrossAttention


__all__ = ["MAE3DChannelCrossAttentionZDistill"]


class AttentionDistillationHead(nn.Module):
    """
    Perceiver-style cross-attention distillation head with z-depth awareness.

    Architecture:
        1. Add z-level embedding (10 learned) + channel embedding (2 learned) to tokens
        2. Concat both channels -> all_tokens [B, 2L, D]
        3. Learnable query tokens cross-attend to all_tokens
        4. FFN on query outputs
        5. Flatten + MLP projection to teacher dimension

    Args:
        embed_dim: Encoder embedding dimension (384)
        teacher_dim: Teacher embedding dimension (1536)
        num_z_levels: Number of z-levels in the grid (10)
        num_channels: Number of image channels (2)
        num_query_tokens: Number of learnable query tokens (4)
        num_attn_heads: Number of attention heads for cross-attention (6)
        dropout: Dropout rate
    """

    def __init__(self, embed_dim, teacher_dim, num_z_levels=10,
                 num_channels=2, num_query_tokens=4, num_attn_heads=6,
                 dropout=0.0):
        super().__init__()

        self.embed_dim = embed_dim
        self.teacher_dim = teacher_dim
        self.num_z_levels = num_z_levels
        self.num_channels = num_channels
        self.num_query_tokens = num_query_tokens

        # Z-level embedding: maps z-index (0..num_z_levels-1) to embed_dim
        self.z_embed = nn.Embedding(num_z_levels, embed_dim)

        # Channel embedding: maps channel index (0 or 1) to embed_dim
        self.channel_embed = nn.Embedding(num_channels, embed_dim)

        # Learnable query tokens
        self.query_tokens = nn.Parameter(torch.zeros(1, num_query_tokens, embed_dim))
        nn.init.trunc_normal_(self.query_tokens, std=0.02)

        # Cross-attention: queries attend to all encoder tokens
        self.ln_q = nn.LayerNorm(embed_dim)
        self.ln_kv = nn.LayerNorm(embed_dim)
        self.cross_attn = CrossAttention(
            dim=embed_dim,
            num_heads=num_attn_heads,
            qkv_bias=True,
            attn_drop=dropout,
            proj_drop=dropout
        )

        # FFN after cross-attention
        self.ln_ffn = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout),
        )

        # Projection MLP: flatten queries -> teacher_dim
        # num_query_tokens * embed_dim -> teacher_dim
        flatten_dim = num_query_tokens * embed_dim  # 4 * 384 = 1536
        self.proj_mlp = nn.Sequential(
            nn.Linear(flatten_dim, teacher_dim),
            nn.GELU(),
            nn.LayerNorm(teacher_dim),
            nn.Linear(teacher_dim, teacher_dim),
        )

        self._init_weights()

    def _init_weights(self):
        # Initialize embeddings
        nn.init.trunc_normal_(self.z_embed.weight, std=0.02)
        nn.init.trunc_normal_(self.channel_embed.weight, std=0.02)

        # Initialize projection MLP
        for m in self.proj_mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x_ch0, x_ch1, sel_indices, grid_size):
        """
        Forward pass of the attention distillation head.

        Args:
            x_ch0: Encoded tokens for channel 0 [B, 1+L, embed_dim] (with CLS)
            x_ch1: Encoded tokens for channel 1 [B, 1+L, embed_dim] (with CLS)
            sel_indices: Selected (visible) patch indices [B, L]
            grid_size: Tuple (gZ, gY, gX) for z-level computation

        Returns:
            student_emb: Student embedding [B, teacher_dim]
        """
        B = x_ch0.shape[0]
        gZ, gY, gX = grid_size

        # 1. Drop CLS tokens
        tokens_ch0 = x_ch0[:, 1:, :]  # [B, L, D]
        tokens_ch1 = x_ch1[:, 1:, :]  # [B, L, D]

        # 2. Compute z-level indices for visible patches
        # Z-major ordering: z_idx = flat_index // (gY * gX)
        z_indices = sel_indices // (gY * gX)  # [B, L], values in [0, gZ-1]

        # Add z-level embedding
        z_emb = self.z_embed(z_indices)  # [B, L, D]
        tokens_ch0 = tokens_ch0 + z_emb
        tokens_ch1 = tokens_ch1 + z_emb

        # Add channel embedding
        ch0_emb = self.channel_embed(torch.zeros(1, dtype=torch.long, device=x_ch0.device))  # [1, D]
        ch1_emb = self.channel_embed(torch.ones(1, dtype=torch.long, device=x_ch0.device))   # [1, D]
        tokens_ch0 = tokens_ch0 + ch0_emb.unsqueeze(0)  # broadcast [1, 1, D]
        tokens_ch1 = tokens_ch1 + ch1_emb.unsqueeze(0)

        # 3. Concat channels -> all_tokens [B, 2L, D]
        all_tokens = torch.cat([tokens_ch0, tokens_ch1], dim=1)

        # 4. Perceiver cross-attention: queries attend to all_tokens
        queries = self.query_tokens.expand(B, -1, -1)  # [B, K, D]
        queries = queries + self.cross_attn(
            self.ln_q(queries),
            self.ln_kv(all_tokens)
        )  # [B, K, D]

        # 5. FFN
        queries = queries + self.ffn(self.ln_ffn(queries))  # [B, K, D]

        # 6. Flatten and project to teacher dimension
        queries_flat = queries.reshape(B, -1)  # [B, K*D]
        student_emb = self.proj_mlp(queries_flat)  # [B, teacher_dim]

        return student_emb


class MAE3DChannelCrossAttentionZDistill(MAE3DChannelCrossAttentionDistill):
    """
    MAE3D with Channel Cross-Attention and Z-Aware Attention Distillation.

    Extends the standard distillation model with a perceiver-style attention
    distillation head that is aware of z-depth structure. Uses a combined loss:
    alpha * attention_distill_loss + (1-alpha) * global_pool_distill_loss.
    """

    def __init__(self, args):
        super().__init__(args)

        # Z-aware distillation configuration
        num_query_tokens = getattr(args, 'distill_num_query_tokens', 4)
        num_attn_heads = getattr(args, 'distill_num_attn_heads', 6)
        num_z_levels = self.grid_size[0]  # gZ from grid_size

        # Create attention distillation head
        self.attn_distill_head = AttentionDistillationHead(
            embed_dim=args.encoder_embed_dim,
            teacher_dim=self.teacher_embed_dim,
            num_z_levels=num_z_levels,
            num_channels=args.in_chans,
            num_query_tokens=num_query_tokens,
            num_attn_heads=num_attn_heads,
        )

        print(f"   Z-Aware Attention Distillation:")
        print(f"     Num query tokens: {num_query_tokens}")
        print(f"     Num attention heads: {num_attn_heads}")
        print(f"     Num z-levels: {num_z_levels}")
        print(f"     Teacher dim: {self.teacher_embed_dim}")

    def forward(self, x, teacher_emb=None, return_image=False, return_distill_loss=False,
                distill_attn_alpha=0.8):
        """
        Forward pass with z-aware attention distillation.

        When teacher_emb is provided and return_distill_loss=True, computes both:
        - attn_distill_loss: from the attention distillation head
        - global_distill_loss: from the standard mean-pool approach (inherited)
        - combined: alpha * attn + (1-alpha) * global

        Args:
            x: Input tensor [B, C, Z, Y, X]
            teacher_emb: Optional teacher embeddings [B, teacher_embed_dim]
            return_image: If True, return reconstruction for visualization
            return_distill_loss: If True, return distillation losses separately
            distill_attn_alpha: Blending factor for attention vs global distill loss

        Returns:
            If return_distill_loss and teacher_emb provided:
                recon_loss, distill_loss, attn_distill_loss, global_distill_loss, (optional vis)
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

        # Normalize encoder outputs
        x_ch0_norm = self.encoder_norms[0](x_ch0)
        x_ch1_norm = self.encoder_norms[1](x_ch1)

        # ============ Distillation ============
        distill_loss = None
        attn_distill_loss = None
        global_distill_loss = None

        if teacher_emb is not None and return_distill_loss:
            if self.distill_nonmasked:
                distill_loss, attn_distill_loss, global_distill_loss = \
                    self._compute_nonmasked_distill(x, teacher_emb, distill_attn_alpha)
            else:
                # Attention-based distillation (z-aware)
                attn_student_emb = self.attn_distill_head(
                    x_ch0_norm, x_ch1_norm, sel_indices, tuple(self.grid_size)
                )
                attn_distill_loss = self._attn_distill_loss(attn_student_emb, teacher_emb)

                # Global pool distillation (inherited approach)
                global_student_emb = self.get_student_embedding(x_ch0_norm, x_ch1_norm)
                projected_teacher = self.project_teacher_embedding(teacher_emb)
                global_distill_loss = self.distillation_loss(global_student_emb, projected_teacher)

                # Combined loss
                distill_loss = distill_attn_alpha * attn_distill_loss + (1 - distill_attn_alpha) * global_distill_loss

        # ============ Decoder ============
        # Project to decoder dimension
        x_ch0_dec = self.encoder_to_decoder[0](x_ch0_norm)
        x_ch1_dec = self.encoder_to_decoder[1](x_ch1_norm)

        dec_channels = []
        for x_enc, mask_token in zip([x_ch0_dec, x_ch1_dec], self.mask_tokens):
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
                return recon_loss, distill_loss, attn_distill_loss, global_distill_loss, original.detach(), recon.detach(), masked.detach()
            return recon_loss, original.detach(), recon.detach(), masked.detach()

        if return_distill_loss and distill_loss is not None:
            return recon_loss, distill_loss, attn_distill_loss, global_distill_loss

        return recon_loss

    def _attn_distill_loss(self, attn_student_emb, teacher_emb):
        """Compute cosine distillation loss for the attention path, with optional standardization."""
        if self.distill_use_standardization:
            attn_student_emb = self.batch_standardize(attn_student_emb)
            teacher_emb = self.batch_standardize(teacher_emb)

        # Clamp to prevent extreme values after standardization
        attn_student_emb = attn_student_emb.clamp(-10, 10)
        teacher_emb = teacher_emb.clamp(-10, 10)

        attn_student_norm = F.normalize(attn_student_emb, p=2, dim=-1)
        teacher_norm = F.normalize(teacher_emb, p=2, dim=-1)
        cos_sim_attn = (attn_student_norm * teacher_norm).sum(dim=-1)
        return (1 - cos_sim_attn).mean()

    def _compute_nonmasked_distill(self, x, teacher_emb, distill_attn_alpha):
        """
        Compute distillation losses using a non-masked encoder pass (all patches visible).

        Returns:
            distill_loss, attn_distill_loss, global_distill_loss
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

            x_emb = embed(x_ch)
            cls = cls_token.expand(B, -1, -1)
            x_emb = torch.cat([cls, x_emb], dim=1)

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

        # All patch indices (no masking)
        all_indices = torch.arange(self.num_patches, device=x.device).unsqueeze(0).expand(B, -1)

        # Attention-based distillation (z-aware)
        attn_student_emb = self.attn_distill_head(
            x_ch0, x_ch1, all_indices, tuple(self.grid_size)
        )
        attn_distill_loss = self._attn_distill_loss(attn_student_emb, teacher_emb)

        # Global pool distillation
        global_student_emb = self.get_student_embedding(x_ch0, x_ch1)
        projected_teacher = self.project_teacher_embedding(teacher_emb)
        global_distill_loss = self.distillation_loss(global_student_emb, projected_teacher)

        # Combined loss
        distill_loss = distill_attn_alpha * attn_distill_loss + (1 - distill_attn_alpha) * global_distill_loss

        return distill_loss, attn_distill_loss, global_distill_loss
