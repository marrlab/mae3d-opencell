"""
MAE2D with Channel Cross-Attention and FFT Loss.

2D adaptation of MAE3DChannelCrossAttentionFFT for max-projected OpenCell images.

Each channel (nucleus, protein) is processed as a separate token stream with
position-wise cross-attention between channels at every transformer layer.
An FFT loss on full per-channel reconstructions encourages proper frequency learning.

Input:  [B, 2, H, W]   (max-projected, H=W=176)
Patch:  [8, 8]         → 22×22 = 484 patches per channel
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from functools import partial

from timm.models.layers.helpers import to_2tuple

from lib.models.mae2d import build_2d_sincos_position_embedding
from lib.networks.cross_attention import DualChannelTransformerBlock
from lib.losses.fft2d_loss import FFT2DLoss


__all__ = ["MAE2DChannelCrossAttentionFFT"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def patchify_channelwise_2d(x, patch_size):
    """
    Split 2D image into per-channel patch sequences.

    Args:
        x:          [B, C, H, W]
        patch_size: (pH, pW)

    Returns:
        List of C tensors, each [B, num_patches, patch_area]
        where patch_area = pH * pW
    """
    B, C, H, W = x.shape
    pH, pW = patch_size
    gH, gW = H // pH, W // pW

    # [B, C, gH, pH, gW, pW]
    x = x.reshape(B, C, gH, pH, gW, pW)
    # [B, C, gH, gW, pH, pW]
    x = x.permute(0, 1, 2, 4, 3, 5)
    # [B, C, num_patches, patch_area]
    x = x.reshape(B, C, gH * gW, pH * pW)

    return [x[:, c, :, :] for c in range(C)]


def unpatchify_channelwise_2d(patches_list, patch_size, grid_size):
    """
    Reverse of patchify_channelwise_2d.

    Args:
        patches_list: List of C tensors, each [B, num_patches, patch_area]
        patch_size:   (pH, pW)
        grid_size:    (gH, gW)

    Returns:
        [B, C, H, W]
    """
    B = patches_list[0].shape[0]
    pH, pW = patch_size
    gH, gW = grid_size

    channels = []
    for patches in patches_list:
        # [B, gH*gW, pH*pW] -> [B, gH, gW, pH, pW]
        x = patches.reshape(B, gH, gW, pH, pW)
        # [B, gH, pH, gW, pW]
        x = x.permute(0, 1, 3, 2, 4)
        # [B, H, W]
        x = x.reshape(B, gH * pH, gW * pW)
        channels.append(x)

    return torch.stack(channels, dim=1)  # [B, C, H, W]


def batched_shuffle_indices(batch_size, length, device):
    rand = torch.rand(batch_size, length, device=device)
    return rand.argsort(dim=1)


class PatchEmbedChannelwise2D(nn.Module):
    """Linear projection of flattened 2D patches (single-channel input)."""

    def __init__(self, patch_area, embed_dim, norm_layer=None):
        super().__init__()
        self.proj = nn.Linear(patch_area, embed_dim)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        """
        Args:
            x: [B, L, patch_area]
        Returns:
            [B, L, embed_dim]
        """
        return self.norm(self.proj(x))


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class MAE2DChannelCrossAttentionFFT(nn.Module):
    """
    2D Masked Autoencoder with Channel Cross-Attention and FFT Loss.

    Processes nucleus and protein channels as separate token streams with
    position-wise cross-attention at each encoder/decoder layer.
    FFT loss is computed on full per-channel reconstructions.

    Architecture mirrors MAE3DChannelCrossAttentionFFT but operates on
    2D max-projected images instead of 3D volumes.
    """

    def __init__(self, args):
        super().__init__()
        self.args = args

        patch_size = to_2tuple(args.patch_size)
        input_size = to_2tuple(args.input_size)
        self.patch_size = patch_size
        self.input_size = input_size
        self.in_chans = args.in_chans

        self.patch_area = patch_size[0] * patch_size[1]
        self.out_chans = self.patch_area * args.in_chans  # for compatibility

        grid_size = []
        for in_sz, pa_sz in zip(input_size, patch_size):
            assert in_sz % pa_sz == 0, "Input size must be divisible by patch size"
            grid_size.append(in_sz // pa_sz)
        self.grid_size = grid_size
        self.num_patches = grid_size[0] * grid_size[1]

        # ------------------------------------------------------------------
        # Position embeddings (shared across channels)
        # ------------------------------------------------------------------
        with torch.no_grad():
            self.encoder_pos_embed = build_2d_sincos_position_embedding(
                grid_size, args.encoder_embed_dim, num_tokens=0
            )
            self.decoder_pos_embed = build_2d_sincos_position_embedding(
                grid_size, args.decoder_embed_dim, num_tokens=0
            )

        # ------------------------------------------------------------------
        # Per-channel patch embeddings, CLS tokens
        # ------------------------------------------------------------------
        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        self.patch_embeds = nn.ModuleList([
            PatchEmbedChannelwise2D(self.patch_area, args.encoder_embed_dim, norm_layer)
            for _ in range(args.in_chans)
        ])

        self.cls_tokens = nn.ParameterList([
            nn.Parameter(torch.zeros(1, 1, args.encoder_embed_dim))
            for _ in range(args.in_chans)
        ])

        self.pos_drop = nn.Dropout(p=0.)

        cross_attention_type = getattr(args, 'cross_attention_type', 'position_wise')
        print(f"   Cross-attention type: {cross_attention_type}")

        # ------------------------------------------------------------------
        # Encoder: dual-stream transformer blocks with cross-attention
        # ------------------------------------------------------------------
        dpr = [0.] * args.encoder_depth
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
                cross_attention_type=cross_attention_type
            )
            for i in range(args.encoder_depth)
        ])

        self.encoder_norms = nn.ModuleList([
            norm_layer(args.encoder_embed_dim) for _ in range(args.in_chans)
        ])

        self.encoder_to_decoder = nn.ModuleList([
            nn.Linear(args.encoder_embed_dim, args.decoder_embed_dim)
            for _ in range(args.in_chans)
        ])

        self.mask_tokens = nn.ParameterList([
            nn.Parameter(torch.zeros(1, 1, args.decoder_embed_dim))
            for _ in range(args.in_chans)
        ])

        # ------------------------------------------------------------------
        # Decoder: dual-stream transformer blocks with cross-attention
        # ------------------------------------------------------------------
        dpr_dec = [0.] * args.decoder_depth
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
                cross_attention_type=cross_attention_type
            )
            for i in range(args.decoder_depth)
        ])

        self.decoder_norms = nn.ModuleList([
            norm_layer(args.decoder_embed_dim) for _ in range(args.in_chans)
        ])

        self.decoder_heads = nn.ModuleList([
            nn.Linear(args.decoder_embed_dim, self.patch_area)
            for _ in range(args.in_chans)
        ])

        # Patch normalisation for MSE loss
        self.patch_norm = nn.LayerNorm(
            normalized_shape=(self.patch_area,), eps=1e-6, elementwise_affine=False
        )

        self.criterion = nn.MSELoss()

        # ------------------------------------------------------------------
        # FFT loss
        # ------------------------------------------------------------------
        self.use_fft_loss = getattr(args, 'use_fft_loss', True)
        self.fft_loss_weight = getattr(args, 'fft_loss_weight', 0.1)
        self.fft_use_log = getattr(args, 'fft_use_log', True)
        self._current_fft_weight = 0.0

        if self.use_fft_loss:
            self.fft_criterion = FFT2DLoss(
                use_log=self.fft_use_log,
                norm_type='ortho'
            )
            print(f"   FFT Loss enabled: weight={self.fft_loss_weight}, use_log={self.fft_use_log}")

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

    def set_fft_weight(self, weight: float):
        self._current_fft_weight = weight

    def get_fft_weight(self) -> float:
        return self._current_fft_weight

    def forward_encoder_no_mask(self, x):
        """
        Forward pass through encoder without masking (for embedding extraction).

        Args:
            x: [B, C, H, W]

        Returns:
            x_ch0: [B, 1+num_patches, encoder_embed_dim]
            x_ch1: [B, 1+num_patches, encoder_embed_dim]
        """
        args = self.args
        B = x.size(0)

        # Patchify per channel
        x_channels = patchify_channelwise_2d(x, self.patch_size)

        # Embed all patches (no masking)
        embedded_channels = []
        for x_ch, embed, cls_token in zip(x_channels, self.patch_embeds, self.cls_tokens):
            x_emb = embed(x_ch)
            cls = cls_token.expand(B, -1, -1)
            x_emb = torch.cat([cls, x_emb], dim=1)

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

    # ------------------------------------------------------------------
    # Forward helpers
    # ------------------------------------------------------------------

    def _reconstruct_full_image_per_channel(self, out_ch, x_ch, unshuffle_indices):
        """
        Reconstruct full 2D image for one channel from decoder output.

        Args:
            out_ch:             [B, 1+num_patches, patch_area]  (includes CLS)
            x_ch:               [B, num_patches,   patch_area]  original patches
            unshuffle_indices:  [B, num_patches]

        Returns:
            [B, H, W]
        """
        # Unshuffle (skip CLS)
        recon = out_ch[:, 1:, :].gather(
            dim=1,
            index=unshuffle_indices[:, :, None].expand(-1, -1, self.patch_area)
        )
        # Denormalise
        recon = recon * (x_ch.var(dim=-1, unbiased=True, keepdim=True).sqrt() + 1e-6)
        recon = recon + x_ch.mean(dim=-1, keepdim=True)

        # Unpatchify → [B, H, W]
        recon_image = unpatchify_channelwise_2d(
            [recon], self.patch_size, self.grid_size
        )[:, 0]
        return recon_image

    def _get_original_image_per_channel(self, x_ch):
        """[B, num_patches, patch_area] → [B, H, W]"""
        return unpatchify_channelwise_2d(
            [x_ch], self.patch_size, self.grid_size
        )[:, 0]

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x, return_image=False):
        """
        Args:
            x:            [B, C, H, W]
            return_image: If True, also return patch tensors for visualisation

        Returns (return_image=False):
            loss, loss_mse, loss_fft

        Returns (return_image=True):
            loss, loss_mse, loss_fft, original_patches, recon_patches, masked_patches
            where *_patches have shape [B, num_patches, patch_area * C]
        """
        args = self.args
        B = x.size(0)
        assert x.size(1) == args.in_chans

        # ---- Patchify -------------------------------------------------------
        x_channels = patchify_channelwise_2d(x, self.patch_size)

        # ---- Masking (shared across channels) --------------------------------
        length = self.num_patches
        sel_length = int(length * (1 - args.mask_ratio))
        msk_length = length - sel_length

        shuffle_indices = batched_shuffle_indices(B, length, x.device)
        unshuffle_indices = shuffle_indices.argsort(dim=1)

        sel_x_channels, msk_x_channels = [], []
        for x_ch in x_channels:
            shuffled = x_ch.gather(
                dim=1,
                index=shuffle_indices[:, :, None].expand(-1, -1, self.patch_area)
            )
            sel_x_channels.append(shuffled[:, :sel_length, :])
            msk_x_channels.append(shuffled[:, -msk_length:, :])

        sel_indices = shuffle_indices[:, :sel_length]

        # ---- Encoder --------------------------------------------------------
        embedded_channels = []
        for sel_x, embed, cls_token in zip(sel_x_channels, self.patch_embeds, self.cls_tokens):
            x_emb = embed(sel_x)
            cls = cls_token.expand(B, -1, -1)
            x_emb = torch.cat([cls, x_emb], dim=1)

            sel_pos = self.encoder_pos_embed.expand(B, -1, -1).gather(
                dim=1,
                index=sel_indices[:, :, None].expand(-1, -1, args.encoder_embed_dim)
            )
            cls_pe = torch.zeros(B, 1, args.encoder_embed_dim, device=x.device)
            x_emb = self.pos_drop(x_emb + torch.cat([cls_pe, sel_pos], dim=1))
            embedded_channels.append(x_emb)

        x_ch0, x_ch1 = embedded_channels[0], embedded_channels[1]
        for block in self.encoder_blocks:
            x_ch0, x_ch1 = block(x_ch0, x_ch1)

        x_ch0 = self.encoder_to_decoder[0](self.encoder_norms[0](x_ch0))
        x_ch1 = self.encoder_to_decoder[1](self.encoder_norms[1](x_ch1))

        # ---- Decoder --------------------------------------------------------
        dec_channels = []
        for x_enc, mask_token in zip([x_ch0, x_ch1], self.mask_tokens):
            all_x = torch.cat([x_enc, mask_token.expand(B, msk_length, -1)], dim=1)
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

        # ---- MSE loss (masked patches only) ---------------------------------
        loss_ch0_mse = self.criterion(
            out_ch0[:, -msk_length:, :],
            self.patch_norm(msk_x_channels[0].detach())
        )
        loss_ch1_mse = self.criterion(
            out_ch1[:, -msk_length:, :],
            self.patch_norm(msk_x_channels[1].detach())
        )
        loss_mse = (loss_ch0_mse + loss_ch1_mse) / 2

        # ---- FFT loss (full reconstruction) ---------------------------------
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

        loss = loss_mse + self._current_fft_weight * loss_fft

        if return_image:
            recon_channels, masked_channels, original_channels = [], [], []
            for out_ch, x_ch in zip([out_ch0, out_ch1], x_channels):
                # Unshuffle and denormalise reconstruction
                recon = out_ch[:, 1:, :].gather(
                    dim=1,
                    index=unshuffle_indices[:, :, None].expand(-1, -1, self.patch_area)
                )
                recon = recon * (x_ch.var(dim=-1, unbiased=True, keepdim=True).sqrt() + 1e-6)
                recon = recon + x_ch.mean(dim=-1, keepdim=True)
                recon_channels.append(recon)

                # Masked version (zeros for masked patches)
                shuffled_vis = x_ch.gather(
                    dim=1,
                    index=shuffle_indices[:, :, None].expand(-1, -1, self.patch_area)
                )
                masked = torch.cat([
                    shuffled_vis[:, :sel_length, :],
                    torch.zeros(B, msk_length, self.patch_area, device=x.device)
                ], dim=1).gather(
                    dim=1,
                    index=unshuffle_indices[:, :, None].expand(-1, -1, self.patch_area)
                )
                masked_channels.append(masked)
                original_channels.append(x_ch)

            # Concatenate channels → [B, num_patches, patch_area * C]
            original = torch.cat(original_channels, dim=-1)
            recon = torch.cat(recon_channels, dim=-1)
            masked = torch.cat(masked_channels, dim=-1)

            return loss, loss_mse, loss_fft, original.detach(), recon.detach(), masked.detach()

        return loss, loss_mse, loss_fft
