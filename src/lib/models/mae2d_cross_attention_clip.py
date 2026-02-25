"""
MAE2D with Channel Cross-Attention and CLIP-based ESM2 Integration.

2D adaptation of MAE3DChannelCrossAttentionCLIP for max-projected OpenCell images.

Architecture:
- Encoder: Standard dual-channel cross-attention (no ESM2, no FFT)
- Decoder: ESM2 embedding concatenated as extra token per channel
- CLIP head: InfoNCE contrastive loss between image and ESM2 embeddings

Compatible with MAE2DChannelCrossAttentionFFT checkpoints (encoder weights
are shared; FFT and CLIP-specific keys are handled via strict=False loading).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from functools import partial

from timm.models.layers.helpers import to_2tuple

from lib.models.mae2d import build_2d_sincos_position_embedding
from lib.models.mae2d_cross_attention_fft import (
    PatchEmbedChannelwise2D,
    patchify_channelwise_2d,
    batched_shuffle_indices,
)
from lib.networks.cross_attention import DualChannelTransformerBlock


__all__ = ["MAE2DChannelCrossAttentionCLIP"]


class MAE2DChannelCrossAttentionCLIP(nn.Module):
    """
    2D Masked Autoencoder with Channel Cross-Attention and CLIP-based ESM2 Integration.

    Processes nucleus and protein channels as separate token streams with
    position-wise cross-attention at every encoder/decoder layer.
    ESM2 is injected as an extra decoder token and used in an InfoNCE
    contrastive loss against the pooled image embedding.

    Input:  [B, 2, H, W]   (max-projected, H=W=176)
    Patch:  [8, 8]         -> 22x22 = 484 patches per channel
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

        # ------------------------------------------------------------------
        # Encoder
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
        # Decoder
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

        self.patch_norm = nn.LayerNorm(
            normalized_shape=(self.patch_area,), eps=1e-6, elementwise_affine=False
        )
        self.criterion = nn.MSELoss()

        # ------------------------------------------------------------------
        # ESM2 decoder components
        # ------------------------------------------------------------------
        self.esm2_embed_dim = getattr(args, 'esm2_embed_dim', 1280)
        self.esm2_decoder_proj = nn.Linear(self.esm2_embed_dim, args.decoder_embed_dim)

        # ------------------------------------------------------------------
        # CLIP / InfoNCE components
        # ------------------------------------------------------------------
        self.clip_embed_dim = getattr(args, 'clip_embed_dim', 768)
        image_embed_dim = args.encoder_embed_dim * args.in_chans  # 384 * 2 = 768
        self.image_proj = nn.Linear(image_embed_dim, self.clip_embed_dim)
        self.esm2_proj = nn.Linear(self.esm2_embed_dim, self.clip_embed_dim)

        clip_temperature_init = getattr(args, 'clip_temperature_init', 0.07)
        self.log_temperature = nn.Parameter(
            torch.tensor(math.log(1.0 / clip_temperature_init))
        )

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
        nn.init.xavier_uniform_(self.esm2_decoder_proj.weight)
        nn.init.zeros_(self.esm2_decoder_proj.bias)
        nn.init.xavier_uniform_(self.image_proj.weight)
        nn.init.zeros_(self.image_proj.bias)
        nn.init.xavier_uniform_(self.esm2_proj.weight)
        nn.init.zeros_(self.esm2_proj.bias)

    def get_image_embedding(self, x_ch0, x_ch1):
        """
        Global-pool encoder outputs and concatenate channels.

        Args:
            x_ch0: [B, 1+N, encoder_embed_dim]
            x_ch1: [B, 1+N, encoder_embed_dim]

        Returns:
            [B, encoder_embed_dim * 2]
        """
        feat_ch0 = x_ch0[:, 1:, :].mean(dim=1)
        feat_ch1 = x_ch1[:, 1:, :].mean(dim=1)
        return torch.cat([feat_ch0, feat_ch1], dim=-1)

    def info_nce_loss(self, image_emb, esm2_emb):
        """Symmetric InfoNCE loss."""
        B = image_emb.size(0)
        image_emb = F.normalize(image_emb, dim=-1)
        esm2_emb = F.normalize(esm2_emb, dim=-1)
        temperature = self.log_temperature.exp()
        logits = image_emb @ esm2_emb.T * temperature
        labels = torch.arange(B, device=logits.device)
        loss_i2e = F.cross_entropy(logits, labels)
        loss_e2i = F.cross_entropy(logits.T, labels)
        return (loss_i2e + loss_e2i) / 2

    def forward_encoder_no_mask(self, x):
        """
        Encode all patches without masking (for embedding extraction).

        Args:
            x: [B, C, H, W]

        Returns:
            x_ch0: [B, 1+num_patches, encoder_embed_dim]
            x_ch1: [B, 1+num_patches, encoder_embed_dim]
        """
        args = self.args
        B = x.size(0)

        x_channels = patchify_channelwise_2d(x, self.patch_size)

        embedded_channels = []
        for x_ch, embed, cls_token in zip(x_channels, self.patch_embeds, self.cls_tokens):
            x_emb = embed(x_ch)
            cls = cls_token.expand(B, -1, -1)
            x_emb = torch.cat([cls, x_emb], dim=1)
            cls_pe = torch.zeros(B, 1, args.encoder_embed_dim, device=x.device)
            pos_embed_full = torch.cat([cls_pe, self.encoder_pos_embed.expand(B, -1, -1)], dim=1)
            x_emb = self.pos_drop(x_emb + pos_embed_full)
            embedded_channels.append(x_emb)

        x_ch0, x_ch1 = embedded_channels[0], embedded_channels[1]
        for block in self.encoder_blocks:
            x_ch0, x_ch1 = block(x_ch0, x_ch1)

        x_ch0 = self.encoder_norms[0](x_ch0)
        x_ch1 = self.encoder_norms[1](x_ch1)

        return x_ch0, x_ch1

    def forward(self, x, esm2_emb=None, return_image=False, return_clip_loss=False):
        """
        Args:
            x:               [B, C, H, W]
            esm2_emb:        [B, esm2_embed_dim] or None
            return_image:    If True, also return patch tensors for visualization
            return_clip_loss: If True, return CLIP loss separately

        Returns (return_clip_loss=True, esm2_emb provided):
            recon_loss, clip_loss
        Returns (return_image=True, return_clip_loss=True):
            recon_loss, clip_loss, original_patches, recon_patches, masked_patches
        """
        args = self.args
        B = x.size(0)
        assert x.size(1) == args.in_chans

        # ---- Patchify -------------------------------------------------------
        x_channels = patchify_channelwise_2d(x, self.patch_size)

        # ---- Masking --------------------------------------------------------
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

        x_ch0_norm = self.encoder_norms[0](x_ch0)
        x_ch1_norm = self.encoder_norms[1](x_ch1)

        # ---- CLIP loss (optional) -------------------------------------------
        clip_loss = None
        if return_clip_loss and esm2_emb is not None:
            image_emb = self.get_image_embedding(x_ch0_norm, x_ch1_norm)
            image_proj = self.image_proj(image_emb)
            esm2_proj = self.esm2_proj(esm2_emb)
            clip_loss = self.info_nce_loss(image_proj, esm2_proj)

        # ---- Project to decoder dim -----------------------------------------
        x_ch0_dec = self.encoder_to_decoder[0](x_ch0_norm)
        x_ch1_dec = self.encoder_to_decoder[1](x_ch1_norm)

        # ---- Decoder (with optional ESM2 token) -----------------------------
        esm2_dec_token = None
        if esm2_emb is not None:
            esm2_dec_token = self.esm2_decoder_proj(esm2_emb).unsqueeze(1)  # [B, 1, dec_dim]

        dec_channels = []
        for x_enc, mask_token in zip([x_ch0_dec, x_ch1_dec], self.mask_tokens):
            if esm2_dec_token is not None:
                all_x = torch.cat([
                    x_enc,
                    esm2_dec_token,
                    mask_token.expand(B, msk_length, -1)
                ], dim=1)
            else:
                all_x = torch.cat([x_enc, mask_token.expand(B, msk_length, -1)], dim=1)
            dec_channels.append(all_x)

        shuffled_dec_pos = self.decoder_pos_embed.expand(B, -1, -1).gather(
            dim=1,
            index=shuffle_indices[:, :, None].expand(-1, -1, args.decoder_embed_dim)
        )

        if esm2_dec_token is not None:
            for i in range(len(dec_channels)):
                dec_channels[i][:, 1:1+sel_length, :] = (
                    dec_channels[i][:, 1:1+sel_length, :] + shuffled_dec_pos[:, :sel_length, :]
                )
                dec_channels[i][:, 2+sel_length:, :] = (
                    dec_channels[i][:, 2+sel_length:, :] + shuffled_dec_pos[:, sel_length:, :]
                )
        else:
            for i in range(len(dec_channels)):
                dec_channels[i][:, 1:, :] = dec_channels[i][:, 1:, :] + shuffled_dec_pos

        x_ch0_d, x_ch1_d = dec_channels[0], dec_channels[1]
        for block in self.decoder_blocks:
            x_ch0_d, x_ch1_d = block(x_ch0_d, x_ch1_d)

        # Remove ESM2 token before head
        if esm2_dec_token is not None:
            out_ch0 = torch.cat([x_ch0_d[:, :1+sel_length, :], x_ch0_d[:, 2+sel_length:, :]], dim=1)
            out_ch1 = torch.cat([x_ch1_d[:, :1+sel_length, :], x_ch1_d[:, 2+sel_length:, :]], dim=1)
        else:
            out_ch0 = x_ch0_d
            out_ch1 = x_ch1_d

        out_ch0 = self.decoder_heads[0](self.decoder_norms[0](out_ch0))
        out_ch1 = self.decoder_heads[1](self.decoder_norms[1](out_ch1))

        # ---- Reconstruction loss (masked patches only) ----------------------
        loss_ch0 = self.criterion(
            out_ch0[:, -msk_length:, :],
            self.patch_norm(msk_x_channels[0].detach())
        )
        loss_ch1 = self.criterion(
            out_ch1[:, -msk_length:, :],
            self.patch_norm(msk_x_channels[1].detach())
        )
        recon_loss = (loss_ch0 + loss_ch1) / 2

        if return_image:
            recon_channels, masked_channels, original_channels = [], [], []
            for out_ch, sel_x, x_ch in zip([out_ch0, out_ch1], sel_x_channels, x_channels):
                recon = out_ch[:, 1:, :].gather(
                    dim=1,
                    index=unshuffle_indices[:, :, None].expand(-1, -1, self.patch_area)
                )
                recon = recon * (x_ch.var(dim=-1, unbiased=True, keepdim=True).sqrt() + 1e-6)
                recon = recon + x_ch.mean(dim=-1, keepdim=True)
                recon_channels.append(recon)

                shuffled_visible = x_ch.gather(
                    dim=1,
                    index=shuffle_indices[:, :, None].expand(-1, -1, self.patch_area)
                )
                masked = torch.cat([
                    shuffled_visible[:, :sel_length, :],
                    torch.zeros(B, msk_length, self.patch_area, device=x.device)
                ], dim=1).gather(
                    dim=1,
                    index=unshuffle_indices[:, :, None].expand(-1, -1, self.patch_area)
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
