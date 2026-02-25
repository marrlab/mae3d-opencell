"""
Protein-Protein Interaction Metric Learning Models.

Implements Siamese-style networks with MLP projection heads for PPI prediction.
Architecture: ViT encoder -> MLP projection -> L2 normalize -> Cosine similarity
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from timm.models.vision_transformer import Block
from timm.models.layers.helpers import to_3tuple
import numpy as np

from lib.networks.patch_embed_layers import PatchEmbed3D, PatchEmbed2D
from lib.models.mae3d import build_3d_sincos_position_embedding
from lib.networks.cross_attention import DualChannelTransformerBlock


__all__ = ['PPIMetric3D', 'PPIMetric2D', 'PPIMetric3DCrossAttention']


class MLPProjectionHead(nn.Module):
    """
    MLP Projection Head for metric learning.

    Maps encoder embeddings to a normalized embedding space.
    """

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
        """
        Args:
            input_dim: Input embedding dimension from encoder
            hidden_dim: Hidden layer dimension
            output_dim: Output embedding dimension (for similarity computation)
            num_layers: Number of layers (2 or 3)
        """
        super().__init__()

        if num_layers == 2:
            self.mlp = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, output_dim)
            )
        elif num_layers == 3:
            self.mlp = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, output_dim)
            )
        else:
            raise ValueError(f"num_layers must be 2 or 3, got {num_layers}")

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input embeddings [B, input_dim]

        Returns:
            L2-normalized output embeddings [B, output_dim]
        """
        z = self.mlp(x)
        z = F.normalize(z, p=2, dim=-1)  # L2 normalize
        return z


class PPIMetric3D(nn.Module):
    """
    3D PPI Metric Learning Model.

    Architecture:
    1. ViT3D Encoder (pretrained from MAE)
    2. MLP Projection Head
    3. L2 Normalization
    4. Cosine Similarity for pair scoring

    Training uses contrastive/margin loss on protein pairs.
    """

    def __init__(self,
                 input_size=(100, 176, 176),
                 patch_size=(10, 8, 8),
                 in_chans=2,
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
                 proj_hidden_dim=512,
                 proj_output_dim=128,
                 proj_num_layers=2):
        """
        Args:
            input_size: Input image size (D, H, W)
            patch_size: Patch size (pd, ph, pw)
            in_chans: Number of input channels
            embed_dim: ViT embedding dimension
            depth: Number of transformer blocks
            num_heads: Number of attention heads
            mlp_ratio: MLP hidden dimension ratio
            qkv_bias: Enable bias in QKV projection
            drop_rate: Dropout rate
            attn_drop_rate: Attention dropout rate
            drop_path_rate: Stochastic depth rate
            pos_embed_type: Position embedding type
            use_global_pool: Use global pooling vs CLS token
            proj_hidden_dim: MLP projection hidden dimension
            proj_output_dim: Final embedding dimension for similarity
            proj_num_layers: Number of MLP layers (2 or 3)
        """
        super().__init__()

        input_size = to_3tuple(input_size)
        patch_size = to_3tuple(patch_size)

        self.input_size = input_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.use_global_pool = use_global_pool
        self.proj_output_dim = proj_output_dim

        # Calculate grid size
        grid_size = []
        for in_size, pa_size in zip(input_size, patch_size):
            assert in_size % pa_size == 0
            grid_size.append(in_size // pa_size)
        self.grid_size = grid_size
        self.num_patches = np.prod(grid_size)

        # Patch embedding
        self.patch_embed = PatchEmbed3D(
            img_size=patch_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim
        )

        # CLS token
        if not use_global_pool:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            nn.init.normal_(self.cls_token, std=.02)
        else:
            self.cls_token = None

        # Positional embedding
        if pos_embed_type == 'sincos':
            with torch.no_grad():
                num_tokens = 0 if use_global_pool else 1
                self.pos_embed = build_3d_sincos_position_embedding(
                    grid_size, embed_dim, num_tokens=num_tokens
                )
        elif pos_embed_type == 'learnable':
            num_tokens = 0 if use_global_pool else 1
            self.pos_embed = nn.Parameter(
                torch.zeros(1, self.num_patches + num_tokens, embed_dim)
            )
            nn.init.normal_(self.pos_embed, std=.02)

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

        # MLP Projection Head
        self.projection_head = MLPProjectionHead(
            input_dim=embed_dim,
            hidden_dim=proj_hidden_dim,
            output_dim=proj_output_dim,
            num_layers=proj_num_layers
        )

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
        """Convert image to patches."""
        B, C, D, H, W = x.shape
        gd, gh, gw = self.grid_size
        pd, ph, pw = self.patch_size

        x = x.reshape(B, C, gd, pd, gh, ph, gw, pw)
        x = x.permute(0, 2, 4, 6, 3, 5, 7, 1).reshape(B, self.num_patches, pd * ph * pw * C)
        return x

    def forward_encoder(self, x):
        """Extract encoder features."""
        B = x.shape[0]

        x = self.patchify_image(x)
        x = self.patch_embed(x)
        x = x.reshape(B, self.num_patches, self.embed_dim)

        if not self.use_global_pool:
            cls_token = self.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_token, x), dim=1)

        x = self.pos_drop(x + self.pos_embed)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)

        if self.use_global_pool:
            x = x.mean(dim=1)
        else:
            x = x[:, 0]

        return x

    def forward_embedding(self, x):
        """Get normalized embedding for a single image."""
        e = self.forward_encoder(x)
        z = self.projection_head(e)
        return z

    def forward_from_embeddings(self, e1, e2):
        """
        Forward pass using precomputed MAE embeddings (fast mode).

        Args:
            e1: Precomputed encoder embeddings for protein 1 [B, embed_dim]
            e2: Precomputed encoder embeddings for protein 2 [B, embed_dim]

        Returns:
            (z1, z2, similarity) — projection-head outputs and cosine similarity
        """
        z1 = self.projection_head(e1)
        z2 = self.projection_head(e2)
        similarity = (z1 * z2).sum(dim=-1)
        return z1, z2, similarity

    def forward(self, x1, x2=None):
        """
        Forward pass for pair or single image.

        Args:
            x1: First image [B, C, D, H, W]
            x2: Second image [B, C, D, H, W] or None

        Returns:
            If x2 is None: normalized embeddings z1 [B, proj_dim]
            If x2 is given: (z1, z2, similarity) where similarity is [B]
        """
        z1 = self.forward_embedding(x1)

        if x2 is None:
            return z1

        z2 = self.forward_embedding(x2)

        # Cosine similarity (embeddings are already L2 normalized)
        similarity = (z1 * z2).sum(dim=-1)

        return z1, z2, similarity

    def compute_similarity(self, z1, z2):
        """Compute cosine similarity between embeddings."""
        return (z1 * z2).sum(dim=-1)

    def get_num_layers(self):
        return len(self.blocks)

    def load_mae_encoder(self, state_dict, strict=False):
        """Load pretrained MAE3D encoder weights."""
        encoder_dict = {}
        for key, value in state_dict.items():
            if key.startswith('encoder.'):
                new_key = key.replace('encoder.', '')
                if not new_key.startswith('head'):
                    encoder_dict[new_key] = value

        # Map to our model
        model_dict = {}
        for key, value in encoder_dict.items():
            if key.startswith('blocks.') or key.startswith('norm.') or key.startswith('patch_embed.'):
                model_dict[key] = value
            elif key == 'cls_token' and self.cls_token is not None:
                model_dict[key] = value

        msg = self.load_state_dict(model_dict, strict=False)
        print(f"Loaded MAE encoder weights: {msg}")
        return msg


class PPIMetric2D(nn.Module):
    """
    2D PPI Metric Learning Model for max-projected images.
    """

    def __init__(self,
                 input_size=(176, 176),
                 patch_size=(8, 8),
                 in_chans=2,
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
                 proj_hidden_dim=512,
                 proj_output_dim=128,
                 proj_num_layers=2):
        """2D ViT + MLP projection for PPI metric learning."""
        super().__init__()

        from lib.networks.mae_vit import build_2d_sincos_position_embedding

        if isinstance(input_size, int):
            input_size = (input_size, input_size)
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)

        self.input_size = input_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.use_global_pool = use_global_pool
        self.proj_output_dim = proj_output_dim

        grid_h = input_size[0] // patch_size[0]
        grid_w = input_size[1] // patch_size[1]
        self.grid_size = (grid_h, grid_w)
        self.num_patches = grid_h * grid_w

        self.patch_embed = PatchEmbed2D(
            img_size=patch_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim
        )

        if not use_global_pool:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            nn.init.normal_(self.cls_token, std=.02)
        else:
            self.cls_token = None

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

        self.pos_drop = nn.Dropout(p=drop_rate)

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

        self.projection_head = MLPProjectionHead(
            input_dim=embed_dim,
            hidden_dim=proj_hidden_dim,
            output_dim=proj_output_dim,
            num_layers=proj_num_layers
        )

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

    def forward_encoder(self, x):
        """Extract encoder features."""
        B = x.shape[0]

        x = self.patchify_image(x)
        x = self.patch_embed(x)
        x = x.reshape(B, self.num_patches, self.embed_dim)

        if not self.use_global_pool:
            cls_token = self.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_token, x), dim=1)

        x = self.pos_drop(x + self.pos_embed)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)

        if self.use_global_pool:
            x = x.mean(dim=1)
        else:
            x = x[:, 0]

        return x

    def forward_embedding(self, x):
        """Get normalized embedding."""
        e = self.forward_encoder(x)
        z = self.projection_head(e)
        return z

    def forward_from_embeddings(self, e1, e2):
        """
        Forward pass using precomputed MAE embeddings (fast mode).

        Args:
            e1: Precomputed encoder embeddings for protein 1 [B, embed_dim]
            e2: Precomputed encoder embeddings for protein 2 [B, embed_dim]

        Returns:
            (z1, z2, similarity) — projection-head outputs and cosine similarity
        """
        z1 = self.projection_head(e1)
        z2 = self.projection_head(e2)
        similarity = (z1 * z2).sum(dim=-1)
        return z1, z2, similarity

    def forward(self, x1, x2=None):
        """Forward pass."""
        z1 = self.forward_embedding(x1)

        if x2 is None:
            return z1

        z2 = self.forward_embedding(x2)
        similarity = (z1 * z2).sum(dim=-1)

        return z1, z2, similarity

    def compute_similarity(self, z1, z2):
        """Compute cosine similarity."""
        return (z1 * z2).sum(dim=-1)

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

        model_dict = {}
        for key, value in encoder_dict.items():
            if key.startswith('blocks.') or key.startswith('norm.') or key.startswith('patch_embed.'):
                model_dict[key] = value
            elif key == 'cls_token' and self.cls_token is not None:
                model_dict[key] = value

        msg = self.load_state_dict(model_dict, strict=False)
        print(f"Loaded MAE encoder weights: {msg}")
        return msg


class PPIMetric3DCrossAttention(nn.Module):
    """
    3D PPI Metric Learning Model with Cross-Attention between channels.

    Uses the dual-stream architecture from MAE3DChannelCrossAttention.
    """

    def __init__(self,
                 input_size=(100, 176, 176),
                 patch_size=(10, 8, 8),
                 in_chans=2,
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
                 proj_hidden_dim=512,
                 proj_output_dim=128,
                 proj_num_layers=2):
        """Cross-attention ViT3D + MLP projection for PPI metric learning."""
        super().__init__()

        from lib.models.vit_classifier import PatchEmbedChannelwise

        assert in_chans == 2, "Cross-attention requires exactly 2 input channels"

        input_size = to_3tuple(input_size)
        patch_size = to_3tuple(patch_size)

        self.input_size = input_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.use_global_pool = use_global_pool
        self.cross_attention_type = cross_attention_type
        self.pool_mode = pool_mode
        self.proj_output_dim = proj_output_dim

        self.patch_volume = np.prod(patch_size)

        grid_size = []
        for in_size, pa_size in zip(input_size, patch_size):
            assert in_size % pa_size == 0
            grid_size.append(in_size // pa_size)
        self.grid_size = grid_size
        self.num_patches = np.prod(grid_size)

        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        self.patch_embeds = nn.ModuleList([
            PatchEmbedChannelwise(patch_size, embed_dim, norm_layer)
            for _ in range(in_chans)
        ])

        self.cls_tokens = nn.ParameterList([
            nn.Parameter(torch.zeros(1, 1, embed_dim))
            for _ in range(in_chans)
        ])

        if pos_embed_type == 'sincos':
            with torch.no_grad():
                self.encoder_pos_embed = build_3d_sincos_position_embedding(
                    grid_size, embed_dim, num_tokens=0
                )
        elif pos_embed_type == 'learnable':
            self.encoder_pos_embed = nn.Parameter(
                torch.zeros(1, self.num_patches, embed_dim)
            )
            nn.init.normal_(self.encoder_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

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

        self.encoder_norms = nn.ModuleList([
            norm_layer(embed_dim)
            for _ in range(in_chans)
        ])

        # Projection head input dimension
        if pool_mode == 'concat':
            encoder_output_dim = embed_dim * in_chans
        else:
            encoder_output_dim = embed_dim

        self.projection_head = MLPProjectionHead(
            input_dim=encoder_output_dim,
            hidden_dim=proj_hidden_dim,
            output_dim=proj_output_dim,
            num_layers=proj_num_layers
        )

        self._init_weights()

    def _init_weights(self):
        for cls_token in self.cls_tokens:
            nn.init.normal_(cls_token, std=.02)
        for embed in self.patch_embeds:
            nn.init.xavier_uniform_(embed.proj.weight)

    def patchify_image_channelwise(self, x):
        """Patchify per channel."""
        B, C, D, H, W = x.shape
        gd, gh, gw = self.grid_size
        pd, ph, pw = self.patch_size

        x = x.reshape(B, C, gd, pd, gh, ph, gw, pw)
        x = x.permute(0, 1, 2, 4, 6, 3, 5, 7)
        x = x.reshape(B, C, gd * gh * gw, pd * ph * pw)

        channels = [x[:, c, :, :] for c in range(C)]
        return channels

    def forward_encoder(self, x):
        """Extract encoder features."""
        B = x.shape[0]

        x_channels = self.patchify_image_channelwise(x)

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

        x_ch0, x_ch1 = embedded_channels[0], embedded_channels[1]
        for block in self.encoder_blocks:
            x_ch0, x_ch1 = block(x_ch0, x_ch1)

        x_ch0 = self.encoder_norms[0](x_ch0)
        x_ch1 = self.encoder_norms[1](x_ch1)

        if self.use_global_pool:
            feat_ch0 = x_ch0[:, 1:, :].mean(dim=1)
            feat_ch1 = x_ch1[:, 1:, :].mean(dim=1)
        else:
            feat_ch0 = x_ch0[:, 0]
            feat_ch1 = x_ch1[:, 0]

        if self.pool_mode == 'concat':
            features = torch.cat([feat_ch0, feat_ch1], dim=-1)
        elif self.pool_mode == 'mean':
            features = (feat_ch0 + feat_ch1) / 2
        elif self.pool_mode == 'sum':
            features = feat_ch0 + feat_ch1
        else:
            raise ValueError(f"Unknown pool_mode: {self.pool_mode}")

        return features

    def forward_embedding(self, x):
        """Get normalized embedding."""
        e = self.forward_encoder(x)
        z = self.projection_head(e)
        return z

    def forward_from_embeddings(self, e1, e2):
        """
        Forward pass using precomputed MAE embeddings (fast mode).

        Args:
            e1: Precomputed encoder embeddings for protein 1 [B, embed_dim * 2]
            e2: Precomputed encoder embeddings for protein 2 [B, embed_dim * 2]

        Returns:
            (z1, z2, similarity) — projection-head outputs and cosine similarity
        """
        z1 = self.projection_head(e1)
        z2 = self.projection_head(e2)
        similarity = (z1 * z2).sum(dim=-1)
        return z1, z2, similarity

    def forward(self, x1, x2=None):
        """Forward pass."""
        z1 = self.forward_embedding(x1)

        if x2 is None:
            return z1

        z2 = self.forward_embedding(x2)
        similarity = (z1 * z2).sum(dim=-1)

        return z1, z2, similarity

    def compute_similarity(self, z1, z2):
        """Compute cosine similarity."""
        return (z1 * z2).sum(dim=-1)

    def get_num_layers(self):
        return len(self.encoder_blocks)

    def load_mae_encoder(self, state_dict, strict=False):
        """Load pretrained MAE3DChannelCrossAttention encoder weights."""
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
