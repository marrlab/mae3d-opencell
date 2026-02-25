"""
MAE3D with Channel Cross-Attention and ESM2 Protein Conditioning.

This model extends MAE3DChannelCrossAttention to support:
1. ESM2 protein embedding conditioning (via cross-attention in encoder)
2. SubCell distillation (optional)

Both modalities are independently toggleable via config flags:
- use_esm2_conditioning: Enable ESM2 cross-attention in encoder
- use_subcell_distillation: Enable SubCell embedding distillation

When both are disabled, the model is identical to MAE3DChannelCrossAttention.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from functools import partial

from timm.models.layers.helpers import to_3tuple

from lib.models.mae3d_cross_attention import (
    MAE3DChannelCrossAttention,
    PatchEmbedChannelwise,
    patchify_image_channelwise,
    batched_shuffle_indices,
    build_3d_sincos_position_embedding
)
from lib.networks.cross_attention import DualChannelTransformerBlock
from lib.networks.protein_modality import DualChannelTransformerBlockESM2


__all__ = ["MAE3DChannelCrossAttentionESM2"]


class MAE3DChannelCrossAttentionESM2(nn.Module):
    """
    3D Masked Autoencoder with Channel Cross-Attention, ESM2 Conditioning, and Distillation.

    This model combines:
    - Channel cross-attention between nucleus and protein image tokens
    - Optional ESM2 protein embedding conditioning in encoder blocks
    - Optional SubCell distillation support

    Key features:
    - ESM2 conditioning: Image tokens attend to ESM2 embedding in encoder (one-way)
    - SubCell distillation: Align encoder representation with SubCell embeddings
    - Backward compatible: Identical to base MAE3DChannelCrossAttention when both features disabled

    Config flags:
    - use_esm2_conditioning: Enable ESM2 cross-attention (default: False)
    - use_subcell_distillation: Enable SubCell distillation (default: False)
    """

    def __init__(self, args):
        super().__init__()
        self.args = args

        # Extract feature flags
        self.use_esm2_conditioning = getattr(args, 'use_esm2_conditioning', False)
        self.use_subcell_distillation = getattr(args, 'use_subcell_distillation', False)

        # Basic setup
        input_size = to_3tuple(args.input_size)
        patch_size = to_3tuple(args.patch_size)
        self.input_size = input_size
        self.patch_size = patch_size
        self.in_chans = args.in_chans

        # Patch dimensions
        self.patch_volume = np.prod(patch_size)
        self.out_chans = self.patch_volume * args.in_chans

        # Grid size
        grid_size = []
        for in_size, pa_size in zip(input_size, patch_size):
            assert in_size % pa_size == 0, "Input size must be divisible by patch size"
            grid_size.append(in_size // pa_size)
        self.grid_size = grid_size
        self.num_patches = np.prod(grid_size)

        # Build position embeddings (shared across channels)
        with torch.no_grad():
            self.encoder_pos_embed = build_3d_sincos_position_embedding(
                grid_size, args.encoder_embed_dim, num_tokens=0
            )
            self.decoder_pos_embed = build_3d_sincos_position_embedding(
                grid_size, args.decoder_embed_dim, num_tokens=0
            )

        # Per-channel patch embeddings
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.patch_embeds = nn.ModuleList([
            PatchEmbedChannelwise(patch_size, args.encoder_embed_dim, norm_layer)
            for _ in range(args.in_chans)
        ])

        # CLS tokens for each channel
        self.cls_tokens = nn.ParameterList([
            nn.Parameter(torch.zeros(1, 1, args.encoder_embed_dim))
            for _ in range(args.in_chans)
        ])

        # Dropout
        self.pos_drop = nn.Dropout(p=0.)

        # Cross-attention type
        self.cross_attention_type = getattr(args, 'cross_attention_type', 'position_wise')
        print(f"   Cross-attention type: {self.cross_attention_type}")

        # ============ Encoder blocks ============
        # Use ESM2-conditioned blocks if enabled
        dpr = [x.item() for x in torch.linspace(0, 0., args.encoder_depth)]
        if self.use_esm2_conditioning:
            self.encoder_blocks = nn.ModuleList([
                DualChannelTransformerBlockESM2(
                    dim=args.encoder_embed_dim,
                    num_heads=args.encoder_num_heads,
                    mlp_ratio=4.,
                    qkv_bias=True,
                    drop=0.,
                    attn_drop=0.,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    cross_attention_type=self.cross_attention_type,
                    use_esm2=True
                )
                for i in range(args.encoder_depth)
            ])
        else:
            self.encoder_blocks = nn.ModuleList([
                DualChannelTransformerBlock(
                    dim=args.encoder_embed_dim,
                    num_heads=args.encoder_num_heads,
                    mlp_ratio=4.,
                    qkv_bias=True,
                    drop=0.,
                    attn_drop=0.,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    cross_attention_type=self.cross_attention_type
                )
                for i in range(args.encoder_depth)
            ])

        # Encoder output normalization
        self.encoder_norms = nn.ModuleList([
            norm_layer(args.encoder_embed_dim)
            for _ in range(args.in_chans)
        ])

        # Encoder to decoder projection (per channel)
        self.encoder_to_decoder = nn.ModuleList([
            nn.Linear(args.encoder_embed_dim, args.decoder_embed_dim)
            for _ in range(args.in_chans)
        ])

        # Mask tokens (per channel)
        self.mask_tokens = nn.ParameterList([
            nn.Parameter(torch.zeros(1, 1, args.decoder_embed_dim))
            for _ in range(args.in_chans)
        ])

        # ============ Decoder blocks ============
        # Decoder uses standard blocks (no ESM2 conditioning)
        dpr_dec = [x.item() for x in torch.linspace(0, 0., args.decoder_depth)]
        self.decoder_blocks = nn.ModuleList([
            DualChannelTransformerBlock(
                dim=args.decoder_embed_dim,
                num_heads=args.decoder_num_heads,
                mlp_ratio=4.,
                qkv_bias=True,
                drop=0.,
                attn_drop=0.,
                drop_path=dpr_dec[i],
                norm_layer=norm_layer,
                cross_attention_type=self.cross_attention_type
            )
            for i in range(args.decoder_depth)
        ])

        # Decoder output normalization and projection
        self.decoder_norms = nn.ModuleList([
            norm_layer(args.decoder_embed_dim)
            for _ in range(args.in_chans)
        ])

        self.decoder_heads = nn.ModuleList([
            nn.Linear(args.decoder_embed_dim, self.patch_volume)
            for _ in range(args.in_chans)
        ])

        # Patch normalization for loss computation
        self.patch_norm = nn.LayerNorm(
            normalized_shape=(self.patch_volume,),
            eps=1e-6,
            elementwise_affine=False
        )

        self.criterion = nn.MSELoss()

        # ============ ESM2 conditioning components ============
        if self.use_esm2_conditioning:
            self.esm2_embed_dim = getattr(args, 'esm2_embed_dim', 1280)
            self.esm2_proj = nn.Linear(self.esm2_embed_dim, args.encoder_embed_dim)
            print(f"   ESM2 conditioning enabled:")
            print(f"     ESM2 embed dim: {self.esm2_embed_dim}")
            print(f"     Projection to: {args.encoder_embed_dim}")

        # ============ Distillation components ============
        if self.use_subcell_distillation:
            self.teacher_embed_dim = getattr(args, 'teacher_embed_dim', 1536)
            self.distill_loss_type = getattr(args, 'distill_loss_type', 'cosine')
            self.use_global_pool = getattr(args, 'distill_use_global_pool', True)
            self.pool_mode = getattr(args, 'distill_pool_mode', 'concat')

            # Calculate student embedding dimension
            if self.pool_mode == 'concat':
                self.student_embed_dim = args.encoder_embed_dim * args.in_chans
            else:
                self.student_embed_dim = args.encoder_embed_dim

            self.teacher_proj = nn.Linear(self.teacher_embed_dim, self.student_embed_dim)
            print(f"   SubCell distillation enabled:")
            print(f"     Teacher embed dim: {self.teacher_embed_dim}")
            print(f"     Student embed dim: {self.student_embed_dim}")
            print(f"     Use global pool: {self.use_global_pool}")
            print(f"     Pool mode: {self.pool_mode}")
            print(f"     Loss type: {self.distill_loss_type}")

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights for all components."""
        for cls_token in self.cls_tokens:
            nn.init.normal_(cls_token, std=.02)
        for mask_token in self.mask_tokens:
            nn.init.normal_(mask_token, std=.02)
        for proj in self.encoder_to_decoder:
            nn.init.xavier_uniform_(proj.weight)
        for embed in self.patch_embeds:
            nn.init.xavier_uniform_(embed.proj.weight)

        if self.use_esm2_conditioning:
            nn.init.xavier_uniform_(self.esm2_proj.weight)
            nn.init.zeros_(self.esm2_proj.bias)

        if self.use_subcell_distillation:
            nn.init.xavier_uniform_(self.teacher_proj.weight)
            nn.init.zeros_(self.teacher_proj.bias)

    def get_student_embedding(self, x_ch0, x_ch1):
        """
        Extract student embedding from encoder outputs for distillation.

        Args:
            x_ch0: Encoded tokens for channel 0 [B, 1+sel_length, embed_dim]
            x_ch1: Encoded tokens for channel 1 [B, 1+sel_length, embed_dim]

        Returns:
            student_emb: Combined embedding [B, student_embed_dim]
        """
        if self.use_global_pool:
            # Global average pooling over spatial tokens (skip CLS at position 0)
            feat_ch0 = x_ch0[:, 1:, :].mean(dim=1)
            feat_ch1 = x_ch1[:, 1:, :].mean(dim=1)
        else:
            # Use CLS tokens
            feat_ch0 = x_ch0[:, 0, :]
            feat_ch1 = x_ch1[:, 0, :]

        # Combine channel features
        if self.pool_mode == 'concat':
            student_emb = torch.cat([feat_ch0, feat_ch1], dim=-1)
        elif self.pool_mode == 'mean':
            student_emb = (feat_ch0 + feat_ch1) / 2
        elif self.pool_mode == 'sum':
            student_emb = feat_ch0 + feat_ch1
        else:
            raise ValueError(f"Unknown pool_mode: {self.pool_mode}")

        return student_emb

    def distillation_loss(self, student_emb, teacher_emb):
        """
        Compute distillation loss between student and projected teacher embeddings.

        Args:
            student_emb: Student embedding [B, embed_dim]
            teacher_emb: Teacher embedding (already projected) [B, embed_dim]

        Returns:
            loss: Scalar loss value
        """
        if self.distill_loss_type == 'mse':
            loss = F.mse_loss(student_emb, teacher_emb)
        elif self.distill_loss_type == 'cosine':
            student_norm = F.normalize(student_emb, p=2, dim=-1)
            teacher_norm = F.normalize(teacher_emb, p=2, dim=-1)
            cos_sim = (student_norm * teacher_norm).sum(dim=-1)
            loss = (1 - cos_sim).mean()
        else:
            raise ValueError(f"Unknown distillation loss type: {self.distill_loss_type}")

        return loss

    def forward_encoder_no_mask(self, x):
        """
        Forward pass through encoder without masking (for embedding extraction).
        ESM2 conditioning is NOT used at inference time.

        Args:
            x: Input tensor [B, C, H, W, D]

        Returns:
            x_ch0: Encoded channel 0 [B, 1+num_patches, embed_dim]
            x_ch1: Encoded channel 1 [B, 1+num_patches, embed_dim]
        """
        args = self.args
        B = x.size(0)

        # Patchify per channel
        x_channels = patchify_image_channelwise(x, self.patch_size)

        # Embed all patches (no masking)
        embedded_channels = []
        for i, (x_ch, embed, cls_token) in enumerate(
            zip(x_channels, self.patch_embeds, self.cls_tokens)):

            x_emb = embed(x_ch)
            cls = cls_token.expand(B, -1, -1)
            x_emb = torch.cat([cls, x_emb], dim=1)

            # Full position embedding
            cls_pe = torch.zeros(B, 1, args.encoder_embed_dim, device=x.device)
            pos_embed_full = torch.cat([cls_pe, self.encoder_pos_embed.expand(B, -1, -1)], dim=1)

            x_emb = self.pos_drop(x_emb + pos_embed_full)
            embedded_channels.append(x_emb)

        # Process through encoder blocks (no ESM2 conditioning at inference)
        x_ch0, x_ch1 = embedded_channels[0], embedded_channels[1]
        for block in self.encoder_blocks:
            if self.use_esm2_conditioning:
                x_ch0, x_ch1 = block(x_ch0, x_ch1, None)
            else:
                x_ch0, x_ch1 = block(x_ch0, x_ch1)

        # Normalize
        x_ch0 = self.encoder_norms[0](x_ch0)
        x_ch1 = self.encoder_norms[1](x_ch1)

        return x_ch0, x_ch1

    def forward(self, x, esm2_emb=None, teacher_emb=None, return_image=False, return_distill_loss=False):
        """
        Forward pass with optional ESM2 conditioning and distillation.

        Args:
            x: Input tensor [B, C, H, W, D]
            esm2_emb: Optional ESM2 embeddings [B, esm2_embed_dim]
            teacher_emb: Optional teacher embeddings for distillation [B, teacher_embed_dim]
            return_image: If True, return reconstruction for visualization
            return_distill_loss: If True, return distillation loss separately

        Returns:
            If return_distill_loss and teacher_emb provided:
                recon_loss, distill_loss, (optional visualization outputs)
            Otherwise:
                Standard MAE outputs (loss or loss + visualizations)
        """
        args = self.args
        B = x.size(0)
        C = x.size(1)
        assert C == args.in_chans, f"Expected {args.in_chans} channels, got {C}"

        # ============ Project ESM2 embedding if provided ============
        esm2_ctx = None
        if self.use_esm2_conditioning and esm2_emb is not None:
            # Project ESM2 embedding: [B, esm2_dim] -> [B, 1, embed_dim]
            esm2_ctx = self.esm2_proj(esm2_emb).unsqueeze(1)

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

        # Process through encoder blocks with cross-attention and optional ESM2
        x_ch0, x_ch1 = embedded_channels[0], embedded_channels[1]
        for block in self.encoder_blocks:
            if self.use_esm2_conditioning:
                x_ch0, x_ch1 = block(x_ch0, x_ch1, esm2_ctx)
            else:
                x_ch0, x_ch1 = block(x_ch0, x_ch1)

        # Normalize encoder outputs
        x_ch0_norm = self.encoder_norms[0](x_ch0)
        x_ch1_norm = self.encoder_norms[1](x_ch1)

        # ============ Distillation Loss (optional) ============
        distill_loss = None
        if self.use_subcell_distillation and teacher_emb is not None and return_distill_loss:
            student_emb = self.get_student_embedding(x_ch0_norm, x_ch1_norm)
            projected_teacher = self.teacher_proj(teacher_emb)
            distill_loss = self.distillation_loss(student_emb, projected_teacher)

        # Project to decoder dimension
        x_ch0 = self.encoder_to_decoder[0](x_ch0_norm)
        x_ch1 = self.encoder_to_decoder[1](x_ch1_norm)

        # ============ Decoder ============
        # Note: Decoder uses standard blocks (no ESM2 conditioning)
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
