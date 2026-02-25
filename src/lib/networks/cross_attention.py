"""
Cross-attention modules for channel-wise token processing.

This module provides building blocks for cross-attention between channels
in multi-channel image data (e.g., nucleus and protein channels in OpenCell).

Supports two modes:
1. Full cross-attention: Each position attends to all positions in the other channel
2. Position-wise cross-attention: Each position only attends to the same position
   in the other channel (more memory efficient, captures co-localization)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from timm.models.vision_transformer import Block, Mlp
from timm.models.layers import DropPath


class PositionWiseCrossAttention(nn.Module):
    """
    Position-wise cross-attention module.

    Each spatial position in the query attends ONLY to the corresponding
    position in the key/value sequence. This is O(N) instead of O(N²).

    This captures co-localization: how nucleus and protein relate at the
    same spatial location (same z, y, x position).

    Architecture:
        1. Project both channels to Q, K, V
        2. At each position i: attention(q_i, k_i) -> weighted v_i
        3. Since there's only one key per query, this simplifies to a
           gating mechanism based on feature similarity
    """

    def __init__(self, dim, num_heads=8, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Projections for query channel
        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        # Projections for key/value channel
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, query, key_value):
        """
        Position-wise cross-attention.

        Args:
            query: [B, N, D] - tokens from one channel
            key_value: [B, N, D] - tokens from the other channel (same N!)

        Returns:
            [B, N, D] - updated query tokens
        """
        B, N, D = query.shape
        assert key_value.shape[1] == N, "Position-wise attention requires same sequence length"

        # Project to Q, K, V
        # [B, N, D] -> [B, N, num_heads, head_dim] -> [B, num_heads, N, head_dim]
        q = self.q_proj(query).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(key_value).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(key_value).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # Position-wise attention: each position only attends to same position
        # q[b, h, i, :] attends to k[b, h, i, :] only
        # Compute attention scores: [B, num_heads, N]
        # Element-wise dot product along head_dim, then scale
        attn_scores = (q * k).sum(dim=-1) * self.scale  # [B, num_heads, N]

        # Softmax is applied per-position (but since there's only 1 key per query,
        # softmax(x) = 1. Instead, we use sigmoid for gating behavior)
        attn_weights = torch.sigmoid(attn_scores)  # [B, num_heads, N]
        attn_weights = self.attn_drop(attn_weights)

        # Apply attention weights to values
        # v: [B, num_heads, N, head_dim]
        # attn_weights: [B, num_heads, N] -> [B, num_heads, N, 1]
        x = v * attn_weights.unsqueeze(-1)  # [B, num_heads, N, head_dim]

        # Reshape back
        x = x.permute(0, 2, 1, 3).reshape(B, N, D)  # [B, N, D]
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class CrossAttention(nn.Module):
    """
    Full cross-attention module where queries come from one source and keys/values from another.

    Each position in query attends to ALL positions in key_value.
    This is O(N²) in memory and computation.

    Used for cross-channel attention where each channel attends to the other channel.
    """

    def __init__(self, dim, num_heads=8, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Separate projections for Q (from target) and K,V (from source)
        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, query, key_value):
        """
        Args:
            query: Tensor of shape [B, N, D] - the sequence that will be updated
            key_value: Tensor of shape [B, M, D] - the sequence to attend to

        Returns:
            Updated query tensor of shape [B, N, D]
        """
        B, N, D = query.shape
        M = key_value.shape[1]

        # Project queries from target, keys and values from source
        q = self.q_proj(query).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(key_value).reshape(B, M, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(key_value).reshape(B, M, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(B, N, D)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class ChannelCrossAttentionBlock(nn.Module):
    """
    A transformer block with both self-attention and position-wise cross-attention.

    For each channel:
    1. Self-attention within the channel (spatial attention across all positions)
    2. Position-wise cross-attention to the other channel (same spatial position only)
    3. MLP

    This allows each channel to:
    - Process its own spatial information globally (self-attention)
    - Incorporate information from the other channel at the same location (cross-attention)

    The position-wise cross-attention captures co-localization: how nucleus and protein
    relate at the same (z, y, x) position.
    """

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=True,
                 drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 cross_attention_type='position_wise'):
        super().__init__()
        self.cross_attention_type = cross_attention_type

        # Self-attention components
        self.norm1 = norm_layer(dim)
        self.self_attn = nn.MultiheadAttention(dim, num_heads, dropout=attn_drop,
                                                batch_first=True, bias=qkv_bias)
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # Cross-attention components (position-wise by default)
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

        # MLP
        self.norm3 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)
        self.drop_path3 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, cross_x):
        """
        Args:
            x: Tensor of shape [B, N, D] - tokens for this channel
            cross_x: Tensor of shape [B, N, D] - tokens from the other channel
                     (must have same N for position-wise attention)

        Returns:
            Updated x tensor of shape [B, N, D]
        """
        # Self-attention (global spatial attention within this channel)
        x_norm = self.norm1(x)
        x_attn, _ = self.self_attn(x_norm, x_norm, x_norm)
        x = x + self.drop_path1(x_attn)

        # Position-wise cross-attention (attend to same position in other channel)
        x = x + self.drop_path2(self.cross_attn(self.norm2(x), cross_x))

        # MLP
        x = x + self.drop_path3(self.mlp(self.norm3(x)))

        return x


class DualChannelTransformerBlock(nn.Module):
    """
    A block that processes two channel token streams in parallel with position-wise cross-attention.

    Both channels undergo:
    1. Self-attention (global spatial attention within channel)
    2. Position-wise cross-attention (to the same position in the other channel)
    3. MLP

    Position-wise cross-attention means:
    - Nucleus token at position (z=5, y=10, x=15) attends to protein token at (z=5, y=10, x=15)
    - This captures co-localization of nucleus and protein signals
    - Memory efficient: O(N) instead of O(N²)

    This is the main building block for channel-aware processing.
    """

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=True,
                 drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 cross_attention_type='position_wise'):
        super().__init__()
        self.cross_attention_type = cross_attention_type

        # Block for channel 0 (nucleus)
        self.block_ch0 = ChannelCrossAttentionBlock(
            dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias, drop=drop, attn_drop=attn_drop,
            drop_path=drop_path, act_layer=act_layer, norm_layer=norm_layer,
            cross_attention_type=cross_attention_type
        )

        # Block for channel 1 (protein)
        self.block_ch1 = ChannelCrossAttentionBlock(
            dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias, drop=drop, attn_drop=attn_drop,
            drop_path=drop_path, act_layer=act_layer, norm_layer=norm_layer,
            cross_attention_type=cross_attention_type
        )

    def forward(self, x_ch0, x_ch1):
        """
        Args:
            x_ch0: Tensor of shape [B, N, D] - tokens for channel 0 (nucleus)
            x_ch1: Tensor of shape [B, N, D] - tokens for channel 1 (protein)

        Returns:
            Tuple of (updated x_ch0, updated x_ch1)
        """
        # Process both channels, each attending to the other at the same position
        # Note: we use the input states for cross-attention (not updated ones)
        # to avoid information leakage within a single block
        x_ch0_new = self.block_ch0(x_ch0, x_ch1)
        x_ch1_new = self.block_ch1(x_ch1, x_ch0)

        return x_ch0_new, x_ch1_new


class ChannelCrossAttentionEncoder(nn.Module):
    """
    Encoder with separate token streams per channel and cross-attention between them.

    Architecture:
    - Separate patch embeddings per channel
    - Dual-stream transformer blocks with cross-attention
    - Optional: merge channels at the end or keep separate
    """

    def __init__(self, patch_size, in_chans=2, embed_dim=384, depth=6,
                 num_heads=6, mlp_ratio=4., qkv_bias=True,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 norm_layer=None, act_layer=None, embed_layer=None,
                 merge_channels=False):
        super().__init__()

        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.merge_channels = merge_channels
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        # Separate patch embeddings for each channel
        # Each processes single-channel patches
        self.patch_embeds = nn.ModuleList([
            embed_layer(img_size=patch_size, patch_size=patch_size,
                       in_chans=1, embed_dim=embed_dim)
            for _ in range(in_chans)
        ])

        # CLS tokens for each channel
        self.cls_tokens = nn.ParameterList([
            nn.Parameter(torch.zeros(1, 1, embed_dim))
            for _ in range(in_chans)
        ])

        # Initialize CLS tokens
        for cls_token in self.cls_tokens:
            nn.init.normal_(cls_token, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # Stochastic depth decay
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        # Dual-channel transformer blocks
        self.blocks = nn.ModuleList([
            DualChannelTransformerBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[i], act_layer=act_layer, norm_layer=norm_layer
            )
            for i in range(depth)
        ])

        # Final normalization for each channel
        self.norms = nn.ModuleList([norm_layer(embed_dim) for _ in range(in_chans)])

        # Optional: merge layer to combine channels
        if merge_channels:
            self.merge_proj = nn.Linear(embed_dim * in_chans, embed_dim)

        self._init_weights()

    def _init_weights(self):
        for embed in self.patch_embeds:
            if hasattr(embed, 'proj') and hasattr(embed.proj, 'weight'):
                w = embed.proj.weight.data
                nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

    def forward(self, x_channels, pos_embeds):
        """
        Args:
            x_channels: List of [B, N, patch_dim] tensors, one per channel
            pos_embeds: List of position embeddings, one per channel

        Returns:
            If merge_channels: [B, N+1, embed_dim]
            Else: List of [B, N+1, embed_dim] per channel
        """
        B = x_channels[0].shape[0]

        # Embed patches for each channel
        embedded = []
        for i, (x, embed, cls_token, pos_embed) in enumerate(
            zip(x_channels, self.patch_embeds, self.cls_tokens, pos_embeds)):

            # Patch embedding
            x = embed(x)  # [B*L, embed_dim] -> reshape needed
            L = x_channels[i].shape[1]
            x = x.reshape(B, L, self.embed_dim)

            # Add CLS token
            cls = cls_token.expand(B, -1, -1)
            x = torch.cat([cls, x], dim=1)

            # Add position embedding (with zero for CLS)
            if pos_embed.size(1) != x.size(1):
                cls_pe = torch.zeros(B, 1, self.embed_dim, device=x.device)
                pos_embed_full = torch.cat([cls_pe, pos_embed], dim=1)
            else:
                pos_embed_full = pos_embed

            x = self.pos_drop(x + pos_embed_full)
            embedded.append(x)

        # Process through dual-channel blocks
        x_ch0, x_ch1 = embedded[0], embedded[1]
        for block in self.blocks:
            x_ch0, x_ch1 = block(x_ch0, x_ch1)

        # Normalize
        x_ch0 = self.norms[0](x_ch0)
        x_ch1 = self.norms[1](x_ch1)

        if self.merge_channels:
            # Merge channels by concatenating and projecting
            x = torch.cat([x_ch0, x_ch1], dim=-1)
            x = self.merge_proj(x)
            return x
        else:
            return [x_ch0, x_ch1]


class ChannelCrossAttentionDecoder(nn.Module):
    """
    Decoder with separate token streams per channel and cross-attention.

    Similar structure to encoder but for reconstruction.
    """

    def __init__(self, patch_size, num_classes, in_chans=2, embed_dim=192,
                 depth=4, num_heads=6, mlp_ratio=4., qkv_bias=True,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 norm_layer=None, act_layer=None):
        super().__init__()

        self.in_chans = in_chans
        self.embed_dim = embed_dim
        # num_classes is the output dimension per patch (single channel)
        self.num_classes_per_channel = num_classes // in_chans

        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.pos_drop = nn.Dropout(p=drop_rate)

        # Stochastic depth decay
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        # Dual-channel transformer blocks
        self.blocks = nn.ModuleList([
            DualChannelTransformerBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[i], act_layer=act_layer, norm_layer=norm_layer
            )
            for i in range(depth)
        ])

        # Final normalization and projection heads for each channel
        self.norms = nn.ModuleList([norm_layer(embed_dim) for _ in range(in_chans)])
        self.heads = nn.ModuleList([
            nn.Linear(embed_dim, self.num_classes_per_channel)
            for _ in range(in_chans)
        ])

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(self, x_channels):
        """
        Args:
            x_channels: List of [B, N, embed_dim] tensors, one per channel

        Returns:
            List of [B, N, num_classes_per_channel] per channel
        """
        x_ch0, x_ch1 = x_channels[0], x_channels[1]

        # Process through dual-channel blocks
        for block in self.blocks:
            x_ch0, x_ch1 = block(x_ch0, x_ch1)

        # Normalize and project to output dimension
        out_ch0 = self.heads[0](self.norms[0](x_ch0))
        out_ch1 = self.heads[1](self.norms[1](x_ch1))

        return [out_ch0, out_ch1]
