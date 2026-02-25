"""
MAE3D with Channel Cross-Attention and CLIP-based ESM2 Integration.

This model extends the base MAE3D cross-attention architecture with:
1. ESM2 token concatenated in the decoder (not encoder)
2. InfoNCE (CLIP-style) contrastive loss between image and ESM2 embeddings

The encoder uses standard DualChannelTransformerBlocks (no ESM2 conditioning),
making it compatible with FFT pretrained checkpoints.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
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


__all__ = ["MAE3DChannelCrossAttentionCLIP"]


class MAE3DChannelCrossAttentionCLIP(nn.Module):
    """
    3D Masked Autoencoder with Channel Cross-Attention and CLIP-based ESM2 Integration.

    Architecture:
    - Encoder: Standard dual-channel cross-attention (no ESM2)
    - Decoder: ESM2 embedding concatenated as extra token per channel
    - CLIP head: InfoNCE contrastive loss between image and ESM2 embeddings
    """

    def __init__(self, args):
        super().__init__()
        self.args = args

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

        # ============ Encoder blocks (standard, no ESM2) ============
        dpr = [x.item() for x in torch.linspace(0, 0., args.encoder_depth)]
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

        # ============ ESM2 decoder components ============
        self.esm2_embed_dim = getattr(args, 'esm2_embed_dim', 1280)
        # Project ESM2 to decoder dim for concatenation
        self.esm2_decoder_proj = nn.Linear(self.esm2_embed_dim, args.decoder_embed_dim)
        print(f"   ESM2 decoder projection: {self.esm2_embed_dim} -> {args.decoder_embed_dim}")

        # ============ CLIP / InfoNCE components ============
        self.clip_embed_dim = getattr(args, 'clip_embed_dim', 256)
        # Image embedding: concat global-pooled channels -> [B, encoder_dim * 2]
        image_embed_dim = args.encoder_embed_dim * args.in_chans  # 384 * 2 = 768
        self.image_proj = nn.Linear(image_embed_dim, self.clip_embed_dim)
        self.esm2_proj = nn.Linear(self.esm2_embed_dim, self.clip_embed_dim)

        # Learnable temperature (init to ln(1/0.07) ≈ 2.6593)
        clip_temperature_init = getattr(args, 'clip_temperature_init', 0.07)
        self.log_temperature = nn.Parameter(
            torch.tensor(math.log(1.0 / clip_temperature_init))
        )
        print(f"   CLIP embed dim: {self.clip_embed_dim}")
        print(f"   CLIP temperature init: {clip_temperature_init}")

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

        # ESM2 decoder projection
        nn.init.xavier_uniform_(self.esm2_decoder_proj.weight)
        nn.init.zeros_(self.esm2_decoder_proj.bias)

        # CLIP projection heads
        nn.init.xavier_uniform_(self.image_proj.weight)
        nn.init.zeros_(self.image_proj.bias)
        nn.init.xavier_uniform_(self.esm2_proj.weight)
        nn.init.zeros_(self.esm2_proj.bias)

    def get_image_embedding(self, x_ch0, x_ch1):
        """
        Extract image embedding from encoder outputs via global pooling + concat.

        Args:
            x_ch0: Encoded channel 0 [B, 1+N, embed_dim] (with CLS)
            x_ch1: Encoded channel 1 [B, 1+N, embed_dim] (with CLS)

        Returns:
            image_emb: [B, encoder_embed_dim * 2]
        """
        # Global average pooling over spatial tokens (skip CLS at position 0)
        feat_ch0 = x_ch0[:, 1:, :].mean(dim=1)  # [B, embed_dim]
        feat_ch1 = x_ch1[:, 1:, :].mean(dim=1)  # [B, embed_dim]
        return torch.cat([feat_ch0, feat_ch1], dim=-1)  # [B, embed_dim * 2]

    def info_nce_loss(self, image_emb, esm2_emb):
        """
        Compute symmetric InfoNCE (CLIP-style) contrastive loss.

        Args:
            image_emb: Image embeddings [B, clip_embed_dim]
            esm2_emb: ESM2 embeddings [B, clip_embed_dim]

        Returns:
            loss: Scalar InfoNCE loss
        """
        B = image_emb.size(0)

        # L2 normalize
        image_emb = F.normalize(image_emb, dim=-1)
        esm2_emb = F.normalize(esm2_emb, dim=-1)

        # Scaled cosine similarity
        temperature = self.log_temperature.exp()
        logits = image_emb @ esm2_emb.T * temperature  # [B, B]

        # Symmetric cross-entropy
        labels = torch.arange(B, device=logits.device)
        loss_i2e = F.cross_entropy(logits, labels)
        loss_e2i = F.cross_entropy(logits.T, labels)
        return (loss_i2e + loss_e2i) / 2

    def forward_encoder_no_mask(self, x):
        """
        Forward pass through encoder without masking (for embedding extraction).
        No ESM2 used in encoder.

        Args:
            x: Input tensor [B, C, D, H, W]

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

        # Process through encoder blocks
        x_ch0, x_ch1 = embedded_channels[0], embedded_channels[1]
        for block in self.encoder_blocks:
            x_ch0, x_ch1 = block(x_ch0, x_ch1)

        # Normalize
        x_ch0 = self.encoder_norms[0](x_ch0)
        x_ch1 = self.encoder_norms[1](x_ch1)

        return x_ch0, x_ch1

    def forward(self, x, esm2_emb=None, return_image=False, return_clip_loss=False):
        """
        Forward pass with ESM2 in decoder and optional CLIP loss.

        Args:
            x: Input tensor [B, C, D, H, W]
            esm2_emb: ESM2 embeddings [B, esm2_embed_dim]
            return_image: If True, return reconstruction for visualization
            return_clip_loss: If True, return CLIP loss separately

        Returns:
            If return_clip_loss and esm2_emb provided:
                recon_loss, clip_loss
            Otherwise:
                recon_loss (or recon_loss + visualization outputs)
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

        # ============ Encoder (standard, no ESM2) ============
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

        # Process through encoder blocks
        x_ch0, x_ch1 = embedded_channels[0], embedded_channels[1]
        for block in self.encoder_blocks:
            x_ch0, x_ch1 = block(x_ch0, x_ch1)

        # Normalize encoder outputs
        x_ch0_norm = self.encoder_norms[0](x_ch0)
        x_ch1_norm = self.encoder_norms[1](x_ch1)

        # ============ CLIP Loss (optional) ============
        clip_loss = None
        if return_clip_loss and esm2_emb is not None:
            image_emb = self.get_image_embedding(x_ch0_norm, x_ch1_norm)
            image_proj = self.image_proj(image_emb)
            esm2_proj = self.esm2_proj(esm2_emb)
            clip_loss = self.info_nce_loss(image_proj, esm2_proj)

        # Project to decoder dimension
        x_ch0_dec = self.encoder_to_decoder[0](x_ch0_norm)
        x_ch1_dec = self.encoder_to_decoder[1](x_ch1_norm)

        # ============ Decoder (with ESM2 token) ============
        # Project ESM2 to decoder dim: [B, esm2_dim] -> [B, 1, decoder_dim]
        esm2_dec_token = None
        if esm2_emb is not None:
            esm2_dec_token = self.esm2_decoder_proj(esm2_emb).unsqueeze(1)  # [B, 1, decoder_dim]

        dec_channels = []
        for x_enc, mask_token in zip([x_ch0_dec, x_ch1_dec], self.mask_tokens):
            # Sequence: [CLS, visible_tokens, ESM2_token, mask_tokens]
            if esm2_dec_token is not None:
                all_x = torch.cat([
                    x_enc,                                    # [B, 1+sel_length, decoder_dim]
                    esm2_dec_token,                           # [B, 1, decoder_dim]
                    mask_token.expand(B, msk_length, -1)      # [B, msk_length, decoder_dim]
                ], dim=1)
            else:
                all_x = torch.cat([
                    x_enc,
                    mask_token.expand(B, msk_length, -1)
                ], dim=1)
            dec_channels.append(all_x)

        # Add positional embeddings to decoder tokens
        shuffled_dec_pos = self.decoder_pos_embed.expand(B, -1, -1).gather(
            dim=1,
            index=shuffle_indices[:, :, None].expand(-1, -1, args.decoder_embed_dim)
        )

        if esm2_dec_token is not None:
            # Position embeddings: CLS=0, visible=selected_pos, ESM2=0, masked=masked_pos
            esm2_zero_pe = torch.zeros(B, 1, args.decoder_embed_dim, device=x.device)
            for i in range(len(dec_channels)):
                # Skip CLS (pos 0), apply pos embed to visible (pos 1:1+sel_length),
                # zero for ESM2 (pos 1+sel_length), pos embed to masked (rest)
                dec_channels[i][:, 1:1+sel_length, :] = dec_channels[i][:, 1:1+sel_length, :] + shuffled_dec_pos[:, :sel_length, :]
                # ESM2 token gets zero position embedding (already zeros)
                dec_channels[i][:, 2+sel_length:, :] = dec_channels[i][:, 2+sel_length:, :] + shuffled_dec_pos[:, sel_length:, :]
        else:
            for i in range(len(dec_channels)):
                dec_channels[i][:, 1:, :] = dec_channels[i][:, 1:, :] + shuffled_dec_pos

        x_ch0_d, x_ch1_d = dec_channels[0], dec_channels[1]
        for block in self.decoder_blocks:
            x_ch0_d, x_ch1_d = block(x_ch0_d, x_ch1_d)

        # ============ Decoder output ============
        # Extract only the patch tokens for reconstruction (skip CLS and ESM2 token)
        if esm2_dec_token is not None:
            # Remove ESM2 token before head: positions are [CLS, visible..., ESM2, masked...]
            # We need [CLS, visible..., masked...] for the heads
            out_ch0 = torch.cat([x_ch0_d[:, :1+sel_length, :], x_ch0_d[:, 2+sel_length:, :]], dim=1)
            out_ch1 = torch.cat([x_ch1_d[:, :1+sel_length, :], x_ch1_d[:, 2+sel_length:, :]], dim=1)
        else:
            out_ch0 = x_ch0_d
            out_ch1 = x_ch1_d

        out_ch0 = self.decoder_heads[0](self.decoder_norms[0](out_ch0))
        out_ch1 = self.decoder_heads[1](self.decoder_norms[1](out_ch1))

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

            if return_clip_loss and clip_loss is not None:
                return recon_loss, clip_loss, original.detach(), recon.detach(), masked.detach()
            return recon_loss, original.detach(), recon.detach(), masked.detach()

        if return_clip_loss and clip_loss is not None:
            return recon_loss, clip_loss

        return recon_loss
