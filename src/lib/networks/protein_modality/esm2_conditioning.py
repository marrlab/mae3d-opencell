"""
ESM2 protein embedding conditioning modules for cross-attention.

This module provides cross-attention mechanisms that allow image tokens
to attend to ESM2 protein embeddings as an additional conditioning signal.

The ESM2 embedding is projected to the image embedding dimension and used
as key/value for cross-attention, while image tokens serve as queries.
This enables one-way conditioning from protein to image representation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from timm.models.layers import DropPath
from timm.models.vision_transformer import Mlp

from lib.networks.cross_attention import (
    PositionWiseCrossAttention,
    CrossAttention
)


class ESM2CrossAttention(nn.Module):
    """
    Cross-attention where image tokens (query) attend to projected ESM2 embedding (key/value).

    ESM2 embedding [B, esm2_dim] is projected to [B, 1, embed_dim], then all image
    tokens attend to this single protein context token.

    This provides a global protein-level conditioning signal to all image tokens.
    """

    def __init__(self, dim, num_heads=8, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Query from image tokens
        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        # Key, Value from ESM2 context
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, query, esm2_ctx):
        """
        Cross-attention from image tokens to ESM2 context.

        Args:
            query: Image tokens [B, N, D] - tokens that will be updated
            esm2_ctx: ESM2 context [B, 1, D] - projected ESM2 embedding

        Returns:
            Updated query tensor [B, N, D]
        """
        B, N, D = query.shape
        M = esm2_ctx.shape[1]  # Should be 1

        # Project queries from image, keys/values from ESM2
        q = self.q_proj(query).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(esm2_ctx).reshape(B, M, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(esm2_ctx).reshape(B, M, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # Attention: [B, num_heads, N, head_dim] @ [B, num_heads, head_dim, M] -> [B, num_heads, N, M]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # Apply attention to values: [B, num_heads, N, M] @ [B, num_heads, M, head_dim] -> [B, num_heads, N, head_dim]
        x = (attn @ v).transpose(1, 2).reshape(B, N, D)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class ESM2ConditionedChannelBlock(nn.Module):
    """
    A transformer block with self-attention, cross-channel attention, optional ESM2 attention, and MLP.

    Extends ChannelCrossAttentionBlock with optional ESM2 cross-attention step:
        self_attn -> cross_channel_attn -> [esm2_attn] -> mlp

    The ESM2 attention is only applied if esm2_ctx is provided and use_esm2 is True.
    """

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=True,
                 drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 cross_attention_type='position_wise',
                 use_esm2=False):
        super().__init__()
        self.cross_attention_type = cross_attention_type
        self.use_esm2 = use_esm2

        # Self-attention components
        self.norm1 = norm_layer(dim)
        self.self_attn = nn.MultiheadAttention(dim, num_heads, dropout=attn_drop,
                                                batch_first=True, bias=qkv_bias)
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # Cross-attention to other channel
        self.norm2 = norm_layer(dim)
        if cross_attention_type == 'position_wise':
            self.cross_attn = PositionWiseCrossAttention(
                dim, num_heads=num_heads, qkv_bias=qkv_bias,
                attn_drop=attn_drop, proj_drop=drop
            )
        else:
            self.cross_attn = CrossAttention(
                dim, num_heads=num_heads, qkv_bias=qkv_bias,
                attn_drop=attn_drop, proj_drop=drop
            )
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # ESM2 cross-attention (optional)
        if use_esm2:
            self.norm_esm2 = norm_layer(dim)
            self.esm2_attn = ESM2CrossAttention(
                dim, num_heads=num_heads, qkv_bias=qkv_bias,
                attn_drop=attn_drop, proj_drop=drop
            )
            self.drop_path_esm2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # MLP
        self.norm3 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)
        self.drop_path3 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, cross_x, esm2_ctx=None):
        """
        Args:
            x: Tensor [B, N, D] - tokens for this channel
            cross_x: Tensor [B, N, D] - tokens from the other channel
            esm2_ctx: Optional tensor [B, 1, D] - projected ESM2 embedding

        Returns:
            Updated x tensor [B, N, D]
        """
        # Self-attention
        x_norm = self.norm1(x)
        x_attn, _ = self.self_attn(x_norm, x_norm, x_norm)
        x = x + self.drop_path1(x_attn)

        # Cross-channel attention
        x = x + self.drop_path2(self.cross_attn(self.norm2(x), cross_x))

        # ESM2 cross-attention (optional)
        if self.use_esm2 and esm2_ctx is not None:
            x = x + self.drop_path_esm2(self.esm2_attn(self.norm_esm2(x), esm2_ctx))

        # MLP
        x = x + self.drop_path3(self.mlp(self.norm3(x)))

        return x


class DualChannelTransformerBlockESM2(nn.Module):
    """
    A block that processes two channel token streams with cross-attention and optional ESM2 conditioning.

    Extends DualChannelTransformerBlock to accept optional esm2_emb and pass to both channel blocks.

    For each channel:
        1. Self-attention (global spatial within channel)
        2. Cross-channel attention (to the other channel)
        3. ESM2 cross-attention (optional, to protein context)
        4. MLP
    """

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=True,
                 drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 cross_attention_type='position_wise',
                 use_esm2=False):
        super().__init__()
        self.cross_attention_type = cross_attention_type
        self.use_esm2 = use_esm2

        # Block for channel 0 (nucleus)
        self.block_ch0 = ESM2ConditionedChannelBlock(
            dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias, drop=drop, attn_drop=attn_drop,
            drop_path=drop_path, act_layer=act_layer, norm_layer=norm_layer,
            cross_attention_type=cross_attention_type,
            use_esm2=use_esm2
        )

        # Block for channel 1 (protein)
        self.block_ch1 = ESM2ConditionedChannelBlock(
            dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias, drop=drop, attn_drop=attn_drop,
            drop_path=drop_path, act_layer=act_layer, norm_layer=norm_layer,
            cross_attention_type=cross_attention_type,
            use_esm2=use_esm2
        )

    def forward(self, x_ch0, x_ch1, esm2_ctx=None):
        """
        Args:
            x_ch0: Tensor [B, N, D] - tokens for channel 0 (nucleus)
            x_ch1: Tensor [B, N, D] - tokens for channel 1 (protein)
            esm2_ctx: Optional tensor [B, 1, D] - projected ESM2 embedding

        Returns:
            Tuple of (updated x_ch0, updated x_ch1)
        """
        # Process both channels, each attending to the other at the same position
        # and optionally to the ESM2 context
        x_ch0_new = self.block_ch0(x_ch0, x_ch1, esm2_ctx)
        x_ch1_new = self.block_ch1(x_ch1, x_ch0, esm2_ctx)

        return x_ch0_new, x_ch1_new
