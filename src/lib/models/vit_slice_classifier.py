"""
ViT Classifier with Slice Aggregation.

This model processes multiple 2D slices from a 3D volume,
extracts embeddings for each slice using a 2D ViT encoder,
and aggregates them (e.g., mean pooling) for classification.
"""

import torch
import torch.nn as nn
from functools import partial
from timm.models.vision_transformer import Block
import numpy as np

from lib.networks.patch_embed_layers import PatchEmbed2D
from lib.networks.mae_vit import build_2d_sincos_position_embedding


class ViT2DSliceAggregateClassifier(nn.Module):
    """
    2D Vision Transformer Classifier with Slice Aggregation.

    For each input volume:
    1. Process each 2D slice through the ViT encoder
    2. Extract embeddings for each slice
    3. Aggregate slice embeddings (mean pooling)
    4. Pass through classification head

    This preserves 3D information while using a 2D encoder pretrained
    on individual slices (MAE2D slices).
    """

    def __init__(self,
                 input_size=(176, 176),
                 patch_size=(8, 8),
                 in_chans=2,
                 num_classes=17,
                 embed_dim=384,
                 depth=6,
                 num_heads=6,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 pos_embed_type='sincos',
                 use_global_pool=True,
                 aggregation='mean'):
        """
        Args:
            input_size: Input image size (H, W)
            patch_size: Patch size (ph, pw)
            in_chans: Number of input channels
            num_classes: Number of output classes
            embed_dim: Embedding dimension
            depth: Number of transformer blocks
            num_heads: Number of attention heads
            mlp_ratio: MLP hidden dimension ratio
            qkv_bias: Enable bias in QKV projection
            drop_rate: Dropout rate
            attn_drop_rate: Attention dropout rate
            drop_path_rate: Stochastic depth rate
            pos_embed_type: Type of positional embedding ('sincos' or 'learnable')
            use_global_pool: If True, use global average pooling
            aggregation: How to aggregate slice embeddings ('mean', 'max', 'attention')
        """
        super().__init__()

        if isinstance(input_size, int):
            input_size = (input_size, input_size)
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)

        self.input_size = input_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.use_global_pool = use_global_pool
        self.aggregation = aggregation

        # Calculate grid size
        grid_h = input_size[0] // patch_size[0]
        grid_w = input_size[1] // patch_size[1]
        self.grid_size = (grid_h, grid_w)
        self.num_patches = grid_h * grid_w

        # Patch embedding
        self.patch_embed = PatchEmbed2D(
            img_size=patch_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim
        )

        # CLS token (optional)
        if not use_global_pool:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            nn.init.normal_(self.cls_token, std=.02)
        else:
            self.cls_token = None

        # Positional embedding
        if pos_embed_type == 'sincos':
            with torch.no_grad():
                num_tokens = 0 if use_global_pool else 1
                self.pos_embed = build_2d_sincos_position_embedding(
                    grid_h, embed_dim, num_tokens=num_tokens
                )
        elif pos_embed_type == 'learnable':
            num_tokens = 0 if use_global_pool else 1
            self.pos_embed = nn.Parameter(
                torch.zeros(1, self.num_patches + num_tokens, embed_dim)
            )
            nn.init.normal_(self.pos_embed, std=.02)
        else:
            raise ValueError(f"Unknown pos_embed_type: {pos_embed_type}")

        self.pos_drop = nn.Dropout(p=drop_rate)

        # Transformer blocks
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        act_layer = nn.GELU
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer
            )
            for i in range(depth)
        ])

        self.norm = norm_layer(embed_dim)

        # Attention-based aggregation (optional)
        if aggregation == 'attention':
            self.slice_attention = nn.Sequential(
                nn.Linear(embed_dim, embed_dim // 4),
                nn.Tanh(),
                nn.Linear(embed_dim // 4, 1)
            )
        else:
            self.slice_attention = None

        # Classification head
        self.head = nn.Linear(embed_dim, num_classes)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify_image(self, x):
        """Convert 2D image to patches."""
        B, C, H, W = x.shape
        gh, gw = self.grid_size
        ph, pw = self.patch_size

        x = x.reshape(B, C, gh, ph, gw, pw)
        x = x.permute(0, 2, 4, 3, 5, 1).reshape(B, self.num_patches, ph * pw * C)

        return x

    def forward_single_slice(self, x):
        """
        Extract embedding from a single 2D slice.

        Args:
            x: Input tensor [B, C, H, W]

        Returns:
            Embedding tensor [B, embed_dim]
        """
        B = x.shape[0]

        # Patchify and embed
        x = self.patchify_image(x)
        x = self.patch_embed(x)
        x = x.reshape(B, self.num_patches, self.embed_dim)

        # Add CLS token if not using global pooling
        if not self.use_global_pool:
            cls_token = self.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_token, x), dim=1)

        # Add positional embedding
        x = self.pos_drop(x + self.pos_embed)

        # Apply transformer blocks
        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)

        # Global pooling or CLS token
        if self.use_global_pool:
            x = x.mean(dim=1)
        else:
            x = x[:, 0]

        return x

    def aggregate_slice_embeddings(self, embeddings, mask=None):
        """
        Aggregate slice embeddings into a single volume embedding.

        Args:
            embeddings: Slice embeddings [B, num_slices, embed_dim]
            mask: Valid slice mask [B, num_slices] (True for valid slices)

        Returns:
            Aggregated embedding [B, embed_dim]
        """
        if mask is not None:
            # Expand mask for broadcasting
            mask_expanded = mask.unsqueeze(-1).float()  # [B, num_slices, 1]

            if self.aggregation == 'mean':
                # Masked mean
                sum_emb = (embeddings * mask_expanded).sum(dim=1)
                count = mask_expanded.sum(dim=1).clamp(min=1)
                return sum_emb / count

            elif self.aggregation == 'max':
                # Masked max (set invalid slices to very negative value)
                masked_emb = embeddings.clone()
                masked_emb[~mask] = float('-inf')
                return masked_emb.max(dim=1)[0]

            elif self.aggregation == 'attention':
                # Attention-weighted aggregation
                attn_scores = self.slice_attention(embeddings).squeeze(-1)  # [B, num_slices]
                attn_scores = attn_scores.masked_fill(~mask, float('-inf'))
                attn_weights = torch.softmax(attn_scores, dim=1).unsqueeze(-1)
                return (embeddings * attn_weights).sum(dim=1)

        else:
            # No mask - all slices valid
            if self.aggregation == 'mean':
                return embeddings.mean(dim=1)
            elif self.aggregation == 'max':
                return embeddings.max(dim=1)[0]
            elif self.aggregation == 'attention':
                attn_scores = self.slice_attention(embeddings).squeeze(-1)
                attn_weights = torch.softmax(attn_scores, dim=1).unsqueeze(-1)
                return (embeddings * attn_weights).sum(dim=1)

        raise ValueError(f"Unknown aggregation: {self.aggregation}")

    def forward(self, slices, mask=None):
        """
        Forward pass for slice aggregation.

        Args:
            slices: Input tensor [B, num_slices, C, H, W]
            mask: Valid slice mask [B, num_slices] (optional)

        Returns:
            Logits tensor [B, num_classes]
        """
        B, num_slices, C, H, W = slices.shape

        # Process all slices at once by reshaping
        # Reshape: [B, num_slices, C, H, W] -> [B * num_slices, C, H, W]
        slices_flat = slices.reshape(B * num_slices, C, H, W)

        # Extract embeddings for all slices
        embeddings_flat = self.forward_single_slice(slices_flat)  # [B * num_slices, embed_dim]

        # Reshape back: [B * num_slices, embed_dim] -> [B, num_slices, embed_dim]
        embeddings = embeddings_flat.reshape(B, num_slices, self.embed_dim)

        # Aggregate slice embeddings
        volume_embedding = self.aggregate_slice_embeddings(embeddings, mask)  # [B, embed_dim]

        # Classification
        logits = self.head(volume_embedding)

        return logits

    def forward_embedding(self, slices, mask=None):
        """
        Extract volume embedding without classification head.

        Args:
            slices: Input tensor [B, num_slices, C, H, W]
            mask: Valid slice mask [B, num_slices] (optional)

        Returns:
            Volume embedding [B, embed_dim]
        """
        B, num_slices, C, H, W = slices.shape

        slices_flat = slices.reshape(B * num_slices, C, H, W)
        embeddings_flat = self.forward_single_slice(slices_flat)
        embeddings = embeddings_flat.reshape(B, num_slices, self.embed_dim)

        return self.aggregate_slice_embeddings(embeddings, mask)

    def get_num_layers(self):
        return len(self.blocks)

    def load_mae_encoder(self, state_dict, strict=False):
        """Load pretrained MAE2D encoder weights."""
        encoder_dict = {}
        for key, value in state_dict.items():
            if key.startswith('encoder.'):
                new_key = key.replace('encoder.', '')
                if not new_key.startswith('head'):
                    encoder_dict[new_key] = value

        msg = self.load_state_dict(encoder_dict, strict=strict)
        print(f"Loaded MAE2D encoder weights: {msg}")
        return msg
