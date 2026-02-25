"""
MAE3D with Channel Cross-Attention.

This model treats each channel (nucleus, protein) as a separate token stream
and uses cross-attention to allow information exchange between channels.

Key differences from standard MAE3D:
1. Channel-wise patchification: Each channel gets its own tokens
2. Dual-stream encoder: Parallel processing with cross-attention
3. Channel-aware masking: Same spatial mask applied to both channels
4. Dual-stream decoder: Separate reconstruction per channel
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from functools import partial

from timm.models.layers.helpers import to_3tuple

from lib.networks.cross_attention import (
    ChannelCrossAttentionEncoder,
    ChannelCrossAttentionDecoder,
    DualChannelTransformerBlock
)


__all__ = ["MAE3DChannelCrossAttention"]


def build_3d_sincos_position_embedding(grid_size, embed_dim, num_tokens=0, temperature=10000.):
    """Build 3D sinusoidal position embedding."""
    grid_size = to_3tuple(grid_size)
    h, w, d = grid_size
    grid_h = torch.arange(h, dtype=torch.float32)
    grid_w = torch.arange(w, dtype=torch.float32)
    grid_d = torch.arange(d, dtype=torch.float32)

    grid_h, grid_w, grid_d = torch.meshgrid(grid_h, grid_w, grid_d, indexing='ij')
    assert embed_dim % 6 == 0, 'Embed dimension must be divisible by 6 for 3D sin-cos position embedding'
    pos_dim = embed_dim // 6
    omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
    omega = 1. / (temperature ** omega)

    out_h = torch.einsum('m,d->md', [grid_h.flatten(), omega])
    out_w = torch.einsum('m,d->md', [grid_w.flatten(), omega])
    out_d = torch.einsum('m,d->md', [grid_d.flatten(), omega])

    pos_emb = torch.cat([
        torch.sin(out_h), torch.cos(out_h),
        torch.sin(out_w), torch.cos(out_w),
        torch.sin(out_d), torch.cos(out_d)
    ], dim=1)[None, :, :]

    if num_tokens == 1:
        pe_token = torch.zeros([1, 1, embed_dim], dtype=torch.float32)
        pos_embed = nn.Parameter(torch.cat([pe_token, pos_emb], dim=1))
    else:
        pos_embed = nn.Parameter(pos_emb)
    pos_embed.requires_grad = False
    return pos_embed


def patchify_image_channelwise(x, patch_size):
    """
    Patchify 3D image into separate token streams per channel.

    Args:
        x: Input tensor [B, C, H, W, D]
        patch_size: Tuple of (pH, pW, pD)

    Returns:
        List of C tensors, each of shape [B, num_patches, patch_volume]
        where patch_volume = pH * pW * pD
    """
    B, C, H, W, D = x.shape
    patch_size = to_3tuple(patch_size)
    pH, pW, pD = patch_size
    gH, gW, gD = H // pH, W // pW, D // pD

    # Reshape to extract patches per channel
    # [B, C, H, W, D] -> [B, C, gH, pH, gW, pW, gD, pD]
    x = x.reshape(B, C, gH, pH, gW, pW, gD, pD)
    # [B, C, gH, gW, gD, pH, pW, pD]
    x = x.permute(0, 1, 2, 4, 6, 3, 5, 7)
    # [B, C, num_patches, patch_volume]
    x = x.reshape(B, C, gH * gW * gD, pH * pW * pD)

    # Split into list per channel
    channels = [x[:, c, :, :] for c in range(C)]
    return channels


def unpatchify_image_channelwise(patches_list, patch_size, grid_size):
    """
    Reverse of patchify_image_channelwise.

    Args:
        patches_list: List of C tensors, each [B, num_patches, patch_volume]
        patch_size: Tuple of (pH, pW, pD)
        grid_size: Tuple of (gH, gW, gD)

    Returns:
        Reconstructed tensor [B, C, H, W, D]
    """
    B = patches_list[0].shape[0]
    C = len(patches_list)
    patch_size = to_3tuple(patch_size)
    pH, pW, pD = patch_size
    gH, gW, gD = grid_size

    channels = []
    for patches in patches_list:
        # [B, num_patches, patch_volume] -> [B, gH, gW, gD, pH, pW, pD]
        x = patches.reshape(B, gH, gW, gD, pH, pW, pD)
        # [B, gH, pH, gW, pW, gD, pD]
        x = x.permute(0, 1, 4, 2, 5, 3, 6)
        # [B, H, W, D]
        x = x.reshape(B, gH * pH, gW * pW, gD * pD)
        channels.append(x)

    # Stack channels: [B, C, H, W, D]
    return torch.stack(channels, dim=1)


def batched_shuffle_indices(batch_size, length, device):
    """Generate random permutations for masking."""
    rand = torch.rand(batch_size, length).to(device)
    batch_perm = rand.argsort(dim=1)
    return batch_perm


class PatchEmbedChannelwise(nn.Module):
    """
    Patch embedding for single-channel input.

    Projects a flattened patch to embedding dimension.
    """

    def __init__(self, patch_size, embed_dim, norm_layer=None):
        super().__init__()
        self.patch_size = to_3tuple(patch_size)
        self.patch_volume = np.prod(self.patch_size)
        self.embed_dim = embed_dim
        self.num_patches = 1  # For compatibility

        self.proj = nn.Linear(self.patch_volume, embed_dim)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        """
        Args:
            x: [B, L, patch_volume] - flattened patches

        Returns:
            [B, L, embed_dim]
        """
        x = self.proj(x)
        x = self.norm(x)
        return x


class MAE3DChannelCrossAttention(nn.Module):
    """
    3D Masked Autoencoder with Channel Cross-Attention.

    This model processes each channel (nucleus, protein) as a separate token stream
    and uses cross-attention to allow channels to exchange information while
    maintaining channel-specific representations.

    Key features:
    - Channel-wise tokens: Each channel has its own spatial tokens
    - Cross-attention: Channels attend to each other at each layer
    - Shared masking: Same spatial positions are masked for all channels
    - Per-channel reconstruction: Separate outputs for each channel
    """

    def __init__(self, args):
        super().__init__()
        self.args = args

        input_size = to_3tuple(args.input_size)
        patch_size = to_3tuple(args.patch_size)
        self.input_size = input_size
        self.patch_size = patch_size
        self.in_chans = args.in_chans

        # Patch volume per channel (single channel patches)
        self.patch_volume = np.prod(patch_size)
        # Output channels = patch_volume * num_channels (for compatibility)
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

        # Cross-attention type: 'position_wise' (default) or 'full'
        # Position-wise: each position attends only to same position in other channel (O(N))
        # Full: each position attends to all positions in other channel (O(N²))
        self.cross_attention_type = getattr(args, 'cross_attention_type', 'position_wise')
        print(f"   Cross-attention type: {self.cross_attention_type}")

        # Encoder: Dual-channel transformer blocks with position-wise cross-attention
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

        # Decoder: Dual-channel transformer blocks with position-wise cross-attention
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

        # Initialize
        self._init_weights()

    def _init_weights(self):
        for cls_token in self.cls_tokens:
            nn.init.normal_(cls_token, std=.02)
        for mask_token in self.mask_tokens:
            nn.init.normal_(mask_token, std=.02)
        for proj in self.encoder_to_decoder:
            nn.init.xavier_uniform_(proj.weight)
        for embed in self.patch_embeds:
            nn.init.xavier_uniform_(embed.proj.weight)

    def forward_encoder_no_mask(self, x):
        """
        Forward pass through encoder without masking (for embedding extraction).

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

            # Full position embedding (no gathering needed - all patches)
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

    def forward(self, x, return_image=False):
        """
        Forward pass with channel-wise processing and cross-attention.

        Args:
            x: Input tensor [B, C, H, W, D]
            return_image: If True, also return reconstruction for visualization

        Returns:
            loss: MSE reconstruction loss
            (optional) original_patches, recon_patches, masked_patches for visualization
        """
        args = self.args
        B = x.size(0)
        C = x.size(1)
        assert C == args.in_chans, f"Expected {args.in_chans} channels, got {C}"

        # ============ Patchify per channel ============
        # List of [B, num_patches, patch_volume] per channel
        x_channels = patchify_image_channelwise(x, self.patch_size)

        # ============ Masking (same mask for all channels) ============
        length = self.num_patches
        sel_length = int(length * (1 - args.mask_ratio))
        msk_length = length - sel_length

        # Generate shuffle indices (shared across channels)
        shuffle_indices = batched_shuffle_indices(B, length, device=x.device)
        unshuffle_indices = shuffle_indices.argsort(dim=1)

        # Apply masking to each channel
        sel_x_channels = []
        msk_x_channels = []
        for x_ch in x_channels:
            # Shuffle
            shuffled = x_ch.gather(
                dim=1,
                index=shuffle_indices[:, :, None].expand(-1, -1, self.patch_volume)
            )
            sel_x_channels.append(shuffled[:, :sel_length, :])
            msk_x_channels.append(shuffled[:, -msk_length:, :])

        sel_indices = shuffle_indices[:, :sel_length]

        # ============ Encoder ============
        # Embed patches per channel
        embedded_channels = []
        for i, (sel_x, embed, cls_token) in enumerate(
            zip(sel_x_channels, self.patch_embeds, self.cls_tokens)):

            # Project patches
            x_emb = embed(sel_x)  # [B, sel_length, embed_dim]

            # Add CLS token
            cls = cls_token.expand(B, -1, -1)
            x_emb = torch.cat([cls, x_emb], dim=1)

            # Add position embedding
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

        # Normalize and project to decoder dimension
        x_ch0 = self.encoder_to_decoder[0](self.encoder_norms[0](x_ch0))
        x_ch1 = self.encoder_to_decoder[1](self.encoder_norms[1](x_ch1))

        # ============ Decoder ============
        # Combine visible tokens with mask tokens per channel
        dec_channels = []
        for x_enc, mask_token in zip([x_ch0, x_ch1], self.mask_tokens):
            # x_enc: [B, 1+sel_length, dec_embed_dim]
            all_x = torch.cat([
                x_enc,
                mask_token.expand(B, msk_length, -1)
            ], dim=1)
            dec_channels.append(all_x)

        # Add shuffled position embeddings (skip CLS position)
        shuffled_dec_pos = self.decoder_pos_embed.expand(B, -1, -1).gather(
            dim=1,
            index=shuffle_indices[:, :, None].expand(-1, -1, args.decoder_embed_dim)
        )
        for i in range(len(dec_channels)):
            dec_channels[i][:, 1:, :] = dec_channels[i][:, 1:, :] + shuffled_dec_pos

        # Process through decoder blocks with cross-attention
        x_ch0, x_ch1 = dec_channels[0], dec_channels[1]
        for block in self.decoder_blocks:
            x_ch0, x_ch1 = block(x_ch0, x_ch1)

        # Project to output dimension (per channel)
        out_ch0 = self.decoder_heads[0](self.decoder_norms[0](x_ch0))
        out_ch1 = self.decoder_heads[1](self.decoder_norms[1](x_ch1))

        # ============ Loss ============
        # Compute loss on masked patches only (skip CLS token)
        loss_ch0 = self.criterion(
            out_ch0[:, -msk_length:, :],
            self.patch_norm(msk_x_channels[0].detach())
        )
        loss_ch1 = self.criterion(
            out_ch1[:, -msk_length:, :],
            self.patch_norm(msk_x_channels[1].detach())
        )
        loss = (loss_ch0 + loss_ch1) / 2

        if return_image:
            # Reconstruct full patches for visualization
            # Unshuffle and combine visible + reconstructed
            recon_channels = []
            masked_channels = []
            original_channels = []

            for out_ch, sel_x, x_ch in zip(
                [out_ch0, out_ch1], sel_x_channels, x_channels):

                # Unshuffle reconstruction (skip CLS)
                recon = out_ch[:, 1:, :].gather(
                    dim=1,
                    index=unshuffle_indices[:, :, None].expand(-1, -1, self.patch_volume)
                )
                # Denormalize
                recon = recon * (x_ch.var(dim=-1, unbiased=True, keepdim=True).sqrt() + 1e-6)
                recon = recon + x_ch.mean(dim=-1, keepdim=True)
                recon_channels.append(recon)

                # Create masked version (zeros for masked patches)
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

            # Concatenate channels back to original format for compatibility
            # [B, num_patches, patch_volume * C]
            original = torch.cat(original_channels, dim=-1)
            recon = torch.cat(recon_channels, dim=-1)
            masked = torch.cat(masked_channels, dim=-1)

            return loss, original.detach(), recon.detach(), masked.detach()

        return loss
