"""
ViT3D Cross-Attention Classifier with SubCell Embedding Fusion.

Combines MAE3D cross-attention encoder features with precomputed SubCell embeddings
for protein localization classification.

Fusion strategy: Concatenate both embeddings → classification head
"""

import torch
import torch.nn as nn
import numpy as np
from functools import partial
from timm.models.layers.helpers import to_3tuple

from lib.models.mae3d import build_3d_sincos_position_embedding
from lib.networks.cross_attention import DualChannelTransformerBlock
from lib.models.vit_classifier import PatchEmbedChannelwise


__all__ = ['ViT3DCrossAttentionSubCellClassifier']


class ViT3DCrossAttentionSubCellClassifier(nn.Module):
    """
    3D Vision Transformer Classifier with Channel Cross-Attention and SubCell Fusion.

    This classifier:
    1. Extracts features from images using MAE3D cross-attention encoder
    2. Takes precomputed SubCell embeddings as additional input
    3. Concatenates both feature vectors
    4. Passes through classification head

    Architecture:
    - Image → MAE3D encoder → global pool + concat → [768]
    - SubCell embedding (precomputed) → [1536]
    - Concatenate → [2304] → Linear → [num_classes]
    """

    def __init__(self,
                 input_size=(100, 176, 176),
                 patch_size=(10, 8, 8),
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
                 cross_attention_type='position_wise',
                 pool_mode='concat',
                 subcell_embed_dim=1536,
                 subcell_proj_dim=None,
                 fusion_type='concat'):
        """
        Args:
            input_size: Input image size (D, H, W)
            patch_size: Patch size (pd, ph, pw)
            in_chans: Number of input channels (must be 2 for cross-attention)
            num_classes: Number of output classes
            embed_dim: Embedding dimension for ViT encoder
            depth: Number of transformer blocks
            num_heads: Number of attention heads
            mlp_ratio: MLP hidden dimension ratio
            qkv_bias: Enable bias in QKV projection
            drop_rate: Dropout rate
            attn_drop_rate: Attention dropout rate
            drop_path_rate: Stochastic depth rate
            pos_embed_type: Type of positional embedding ('sincos' or 'learnable')
            use_global_pool: If True, use global average pooling; If False, use CLS tokens
            cross_attention_type: 'position_wise' (O(N)) or 'full' (O(N²))
            pool_mode: How to combine channel features: 'concat', 'mean', 'sum'
            subcell_embed_dim: Dimension of SubCell embeddings (1536)
            subcell_proj_dim: If set, project SubCell to this dim before fusion (None = no projection)
            fusion_type: How to fuse features: 'concat' (default), 'add' (requires same dim)
        """
        super().__init__()

        assert in_chans == 2, "Cross-attention classifier requires exactly 2 input channels"

        input_size = to_3tuple(input_size)
        patch_size = to_3tuple(patch_size)

        self.input_size = input_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.use_global_pool = use_global_pool
        self.cross_attention_type = cross_attention_type
        self.pool_mode = pool_mode
        self.subcell_embed_dim = subcell_embed_dim
        self.subcell_proj_dim = subcell_proj_dim
        self.fusion_type = fusion_type

        # Patch volume per channel
        self.patch_volume = np.prod(patch_size)

        # Calculate grid size
        grid_size = []
        for in_size, pa_size in zip(input_size, patch_size):
            assert in_size % pa_size == 0
            grid_size.append(in_size // pa_size)
        self.grid_size = grid_size
        self.num_patches = np.prod(grid_size)

        # ============ MAE3D Encoder Components ============
        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        # Per-channel patch embeddings
        self.patch_embeds = nn.ModuleList([
            PatchEmbedChannelwise(patch_size, embed_dim, norm_layer)
            for _ in range(in_chans)
        ])

        # CLS tokens for each channel
        self.cls_tokens = nn.ParameterList([
            nn.Parameter(torch.zeros(1, 1, embed_dim))
            for _ in range(in_chans)
        ])

        # Positional embedding
        if pos_embed_type == 'sincos':
            with torch.no_grad():
                self.encoder_pos_embed = build_3d_sincos_position_embedding(
                    grid_size, embed_dim, num_tokens=0
                )
        else:
            self.encoder_pos_embed = nn.Parameter(
                torch.zeros(1, self.num_patches, embed_dim)
            )
            nn.init.normal_(self.encoder_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # Dual-channel transformer blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.encoder_blocks = nn.ModuleList([
            DualChannelTransformerBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                cross_attention_type=cross_attention_type
            )
            for i in range(depth)
        ])

        # Encoder output normalization
        self.encoder_norms = nn.ModuleList([
            norm_layer(embed_dim)
            for _ in range(in_chans)
        ])

        # ============ SubCell Projection (optional) ============
        if subcell_proj_dim is not None:
            self.subcell_proj = nn.Linear(subcell_embed_dim, subcell_proj_dim)
            subcell_final_dim = subcell_proj_dim
        else:
            self.subcell_proj = None
            subcell_final_dim = subcell_embed_dim

        # ============ Classification Head ============
        # Calculate MAE3D feature dimension
        if pool_mode == 'concat':
            mae_feature_dim = embed_dim * in_chans  # 384 * 2 = 768
        else:
            mae_feature_dim = embed_dim  # 384

        # Calculate final feature dimension based on fusion type
        if fusion_type == 'concat':
            head_input_dim = mae_feature_dim + subcell_final_dim
        elif fusion_type == 'add':
            assert mae_feature_dim == subcell_final_dim, \
                f"For 'add' fusion, dims must match: {mae_feature_dim} vs {subcell_final_dim}"
            head_input_dim = mae_feature_dim
        else:
            raise ValueError(f"Unknown fusion_type: {fusion_type}")

        self.head = nn.Linear(head_input_dim, num_classes)

        # Store dimensions for logging
        self.mae_feature_dim = mae_feature_dim
        self.subcell_final_dim = subcell_final_dim
        self.head_input_dim = head_input_dim

        # Initialize weights
        self._init_weights()

        print(f"   MAE3D feature dim: {mae_feature_dim}")
        print(f"   SubCell feature dim: {subcell_final_dim}")
        print(f"   Fusion type: {fusion_type}")
        print(f"   Classification head input dim: {head_input_dim}")

    def _init_weights(self):
        for cls_token in self.cls_tokens:
            nn.init.normal_(cls_token, std=.02)
        for embed in self.patch_embeds:
            nn.init.xavier_uniform_(embed.proj.weight)
        nn.init.xavier_uniform_(self.head.weight)
        if self.head.bias is not None:
            nn.init.constant_(self.head.bias, 0)
        if self.subcell_proj is not None:
            nn.init.xavier_uniform_(self.subcell_proj.weight)
            nn.init.constant_(self.subcell_proj.bias, 0)

    def patchify_image_channelwise(self, x):
        """Patchify 3D image into separate token streams per channel."""
        B, C, D, H, W = x.shape
        gd, gh, gw = self.grid_size
        pd, ph, pw = self.patch_size

        x = x.reshape(B, C, gd, pd, gh, ph, gw, pw)
        x = x.permute(0, 1, 2, 4, 6, 3, 5, 7)
        x = x.reshape(B, C, gd * gh * gw, pd * ph * pw)

        channels = [x[:, c, :, :] for c in range(C)]
        return channels

    def forward_mae_features(self, x):
        """Extract features from image using MAE3D cross-attention encoder."""
        B = x.shape[0]

        # Patchify per channel
        x_channels = self.patchify_image_channelwise(x)

        # Embed patches and add CLS tokens
        embedded_channels = []
        for i, (x_ch, embed, cls_token) in enumerate(
            zip(x_channels, self.patch_embeds, self.cls_tokens)):

            x_emb = embed(x_ch)
            cls = cls_token.expand(B, -1, -1)
            x_emb = torch.cat([cls, x_emb], dim=1)

            pos_embed = self.encoder_pos_embed.expand(B, -1, -1)
            cls_pe = torch.zeros(B, 1, self.embed_dim, device=x.device)
            pos_embed_full = torch.cat([cls_pe, pos_embed], dim=1)

            x_emb = self.pos_drop(x_emb + pos_embed_full)
            embedded_channels.append(x_emb)

        # Process through encoder blocks
        x_ch0, x_ch1 = embedded_channels[0], embedded_channels[1]
        for block in self.encoder_blocks:
            x_ch0, x_ch1 = block(x_ch0, x_ch1)

        # Normalize
        x_ch0 = self.encoder_norms[0](x_ch0)
        x_ch1 = self.encoder_norms[1](x_ch1)

        # Pool features
        if self.use_global_pool:
            feat_ch0 = x_ch0[:, 1:, :].mean(dim=1)
            feat_ch1 = x_ch1[:, 1:, :].mean(dim=1)
        else:
            feat_ch0 = x_ch0[:, 0]
            feat_ch1 = x_ch1[:, 0]

        # Combine channel features
        if self.pool_mode == 'concat':
            features = torch.cat([feat_ch0, feat_ch1], dim=-1)
        elif self.pool_mode == 'mean':
            features = (feat_ch0 + feat_ch1) / 2
        elif self.pool_mode == 'sum':
            features = feat_ch0 + feat_ch1
        else:
            raise ValueError(f"Unknown pool_mode: {self.pool_mode}")

        return features

    def forward(self, x, subcell_emb):
        """
        Forward pass with image and SubCell embedding fusion.

        Args:
            x: Input image [B, C, D, H, W]
            subcell_emb: Precomputed SubCell embedding [B, subcell_embed_dim]

        Returns:
            logits: [B, num_classes]
        """
        # Extract MAE3D features
        mae_features = self.forward_mae_features(x)  # [B, mae_feature_dim]

        # Project SubCell if needed
        if self.subcell_proj is not None:
            subcell_features = self.subcell_proj(subcell_emb)
        else:
            subcell_features = subcell_emb

        # Fuse features
        if self.fusion_type == 'concat':
            fused = torch.cat([mae_features, subcell_features], dim=-1)
        elif self.fusion_type == 'add':
            fused = mae_features + subcell_features
        else:
            raise ValueError(f"Unknown fusion_type: {self.fusion_type}")

        # Classification
        logits = self.head(fused)

        return logits

    def forward_from_embeddings(self, mae_emb, subcell_emb):
        """
        Forward pass using precomputed MAE embeddings (fast mode).

        Use this when MAE embeddings are precomputed and saved to disk.
        Skips the entire MAE encoder forward pass.

        Args:
            mae_emb: Precomputed MAE embedding [B, mae_feature_dim]
            subcell_emb: Precomputed SubCell embedding [B, subcell_embed_dim]

        Returns:
            logits: [B, num_classes]
        """
        # Project SubCell if needed
        if self.subcell_proj is not None:
            subcell_features = self.subcell_proj(subcell_emb)
        else:
            subcell_features = subcell_emb

        # Fuse features
        if self.fusion_type == 'concat':
            fused = torch.cat([mae_emb, subcell_features], dim=-1)
        elif self.fusion_type == 'add':
            fused = mae_emb + subcell_features
        else:
            raise ValueError(f"Unknown fusion_type: {self.fusion_type}")

        # Classification
        logits = self.head(fused)

        return logits

    def get_num_layers(self):
        return len(self.encoder_blocks)

    def load_mae_encoder(self, state_dict, strict=False):
        """
        Load pretrained MAE3DChannelCrossAttention encoder weights.

        Maps weights from MAE3DChannelCrossAttention to this classifier.
        """
        encoder_dict = {}

        for key, value in state_dict.items():
            if any(skip in key for skip in ['decoder', 'mask_token', 'encoder_to_decoder']):
                continue

            if key.startswith('patch_embeds.'):
                encoder_dict[key] = value
            elif key.startswith('cls_tokens.'):
                encoder_dict[key] = value
            elif key.startswith('encoder_pos_embed'):
                encoder_dict[key] = value
            elif key.startswith('encoder_blocks.'):
                encoder_dict[key] = value
            elif key.startswith('encoder_norms.'):
                encoder_dict[key] = value
            elif key.startswith('pos_drop.'):
                encoder_dict[key] = value

        msg = self.load_state_dict(encoder_dict, strict=strict)
        print(f"Loaded MAE cross-attention encoder weights:")
        print(f"  Missing keys: {len(msg.missing_keys)}")
        print(f"  Unexpected keys: {len(msg.unexpected_keys)}")

        return msg
