import torch
import torch.nn as nn
from functools import partial
from timm.models.vision_transformer import Block
from timm.models.layers.helpers import to_3tuple
import numpy as np

from lib.networks.patch_embed_layers import PatchEmbed3D, PatchEmbed2D
from lib.models.mae3d import build_3d_sincos_position_embedding
from lib.networks.cross_attention import DualChannelTransformerBlock


__all__ = ['ViT3DClassifier', 'ViT2DClassifier', 'ViT3DCrossAttentionClassifier']


class ViT3DClassifier(nn.Module):
    """
    3D Vision Transformer for classification.
    Can load pretrained MAE3D encoder weights.
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
                 use_global_pool=True):
        """
        Args:
            input_size: Input image size (D, H, W)
            patch_size: Patch size (pd, ph, pw)
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
            use_global_pool: If True, use global average pooling + classification head
                            If False, use CLS token for classification
        """
        super().__init__()

        input_size = to_3tuple(input_size)
        patch_size = to_3tuple(patch_size)

        self.input_size = input_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.use_global_pool = use_global_pool

        # Calculate grid size
        grid_size = []
        for in_size, pa_size in zip(input_size, patch_size):
            assert in_size % pa_size == 0, f"input size {in_size} must be divisible by patch size {pa_size}"
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

        # CLS token (optional, only used if not using global pooling)
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

        # Classification head
        self.head = nn.Linear(embed_dim, num_classes)

        # Initialize weights
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
        assert D == self.input_size[0] and H == self.input_size[1] and W == self.input_size[2], \
            f"Input size mismatch: expected {self.input_size}, got ({D}, {H}, {W})"

        gd, gh, gw = self.grid_size
        pd, ph, pw = self.patch_size

        # Reshape: [B, C, D, H, W] -> [B, C, gd, pd, gh, ph, gw, pw]
        x = x.reshape(B, C, gd, pd, gh, ph, gw, pw)
        # Permute and reshape: -> [B, gd*gh*gw, pd*ph*pw*C]
        x = x.permute(0, 2, 4, 6, 3, 5, 7, 1).reshape(B, self.num_patches, pd * ph * pw * C)

        return x

    def forward_features(self, x):
        """Extract features from input."""
        B = x.shape[0]

        # Patchify input
        x = self.patchify_image(x)  # [B, num_patches, patch_dim]

        # Embed patches
        x = self.patch_embed(x)  # [B*num_patches, embed_dim]
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
            x = x.mean(dim=1)  # Global average pooling
        else:
            x = x[:, 0]  # CLS token

        return x

    def forward(self, x):
        """Forward pass."""
        x = self.forward_features(x)
        x = self.head(x)
        return x

    def forward_from_embeddings(self, embeddings):
        """
        Forward pass using precomputed embeddings (fast mode).

        Args:
            embeddings: Precomputed MAE embeddings [B, embed_dim]

        Returns:
            logits: [B, num_classes]
        """
        return self.head(embeddings)

    def get_num_layers(self):
        return len(self.blocks)

    def get_feature_dim(self):
        """Return the feature dimension (embedding dimension)."""
        return self.embed_dim

    def load_mae_encoder(self, state_dict, strict=False):
        """
        Load pretrained MAE3D encoder weights.

        Args:
            state_dict: State dict from MAE3D checkpoint
            strict: Whether to strictly enforce key matching
        """
        # Filter encoder weights
        encoder_dict = {}
        for key, value in state_dict.items():
            if key.startswith('encoder.'):
                new_key = key.replace('encoder.', '')
                # Skip the head since we have a different classification head
                if not new_key.startswith('head'):
                    encoder_dict[new_key] = value

        # Load weights (allow missing keys for head and cls_token if using global pooling)
        msg = self.load_state_dict(encoder_dict, strict=strict)
        print(f"Loaded MAE encoder weights: {msg}")
        return msg


class ViT2DClassifier(nn.Module):
    """
    2D Vision Transformer for classification.
    Can load pretrained MAE2D encoder weights.
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
                 use_global_pool=True):
        """
        2D ViT classifier for max-projected images.
        Similar to ViT3DClassifier but for 2D inputs.
        """
        super().__init__()

        from lib.networks.mae_vit import build_2d_sincos_position_embedding

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
                # Use grid_h for sincos (assuming square grid)
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

        # Reshape: [B, C, H, W] -> [B, C, gh, ph, gw, pw]
        x = x.reshape(B, C, gh, ph, gw, pw)
        # Permute and reshape: -> [B, gh*gw, ph*pw*C]
        x = x.permute(0, 2, 4, 3, 5, 1).reshape(B, self.num_patches, ph * pw * C)

        return x

    def forward_features(self, x):
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

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

    def forward_from_embeddings(self, embeddings):
        """
        Forward pass using precomputed embeddings (fast mode).

        Args:
            embeddings: Precomputed MAE embeddings [B, embed_dim]

        Returns:
            logits: [B, num_classes]
        """
        return self.head(embeddings)

    def get_num_layers(self):
        return len(self.blocks)

    def get_feature_dim(self):
        """Return the feature dimension (embedding dimension)."""
        return self.embed_dim

    def load_mae_encoder(self, state_dict, strict=False):
        """Load pretrained MAE2D encoder weights."""
        encoder_dict = {}
        for key, value in state_dict.items():
            if key.startswith('encoder.'):
                new_key = key.replace('encoder.', '')
                if not new_key.startswith('head'):
                    encoder_dict[new_key] = value

        msg = self.load_state_dict(encoder_dict, strict=strict)
        print(f"Loaded MAE encoder weights: {msg}")
        return msg


class PatchEmbedChannelwise(nn.Module):
    """
    Patch embedding for single-channel input.
    Projects a flattened patch to embedding dimension.
    Must match MAE3DChannelCrossAttention architecture.
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


class ViT3DCrossAttentionClassifier(nn.Module):
    """
    3D Vision Transformer Classifier with Channel Cross-Attention.

    This classifier uses the same dual-stream architecture as MAE3DChannelCrossAttention,
    allowing it to load pretrained weights from that model.

    Key features:
    - Channel-wise tokens: Each channel (nucleus, protein) has its own spatial tokens
    - Cross-attention: Channels attend to each other at each transformer layer
    - Classification: Uses global pooling or CLS tokens from both channels

    Architecture matches MAE3DChannelCrossAttention encoder exactly for weight loading.
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
                 pool_mode='concat'):
        """
        Args:
            input_size: Input image size (D, H, W)
            patch_size: Patch size (pd, ph, pw)
            in_chans: Number of input channels (must be 2 for cross-attention)
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
            use_global_pool: If True, use global average pooling; If False, use CLS tokens
            cross_attention_type: 'position_wise' (O(N)) or 'full' (O(N²))
            pool_mode: How to combine channel features for classification:
                       'concat' - concatenate channel features (default)
                       'mean' - average channel features
                       'sum' - sum channel features
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

        # Patch volume per channel (single channel patches)
        self.patch_volume = np.prod(patch_size)

        # Calculate grid size
        grid_size = []
        for in_size, pa_size in zip(input_size, patch_size):
            assert in_size % pa_size == 0, f"input size {in_size} must be divisible by patch size {pa_size}"
            grid_size.append(in_size // pa_size)
        self.grid_size = grid_size
        self.num_patches = np.prod(grid_size)

        # Per-channel patch embeddings (matches MAE3DChannelCrossAttention)
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.patch_embeds = nn.ModuleList([
            PatchEmbedChannelwise(patch_size, embed_dim, norm_layer)
            for _ in range(in_chans)
        ])

        # CLS tokens for each channel (matches MAE3DChannelCrossAttention)
        self.cls_tokens = nn.ParameterList([
            nn.Parameter(torch.zeros(1, 1, embed_dim))
            for _ in range(in_chans)
        ])

        # Positional embedding (shared across channels, matches MAE3DChannelCrossAttention)
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
        else:
            raise ValueError(f"Unknown pos_embed_type: {pos_embed_type}")

        self.pos_drop = nn.Dropout(p=drop_rate)

        # Dual-channel transformer blocks (matches MAE3DChannelCrossAttention)
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

        # Encoder output normalization (per channel, matches MAE3DChannelCrossAttention)
        self.encoder_norms = nn.ModuleList([
            norm_layer(embed_dim)
            for _ in range(in_chans)
        ])

        # Classification head
        # Input dimension depends on pool_mode
        if pool_mode == 'concat':
            head_input_dim = embed_dim * in_chans
        else:
            head_input_dim = embed_dim

        self.head = nn.Linear(head_input_dim, num_classes)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for cls_token in self.cls_tokens:
            nn.init.normal_(cls_token, std=.02)
        for embed in self.patch_embeds:
            nn.init.xavier_uniform_(embed.proj.weight)
        nn.init.xavier_uniform_(self.head.weight)
        if self.head.bias is not None:
            nn.init.constant_(self.head.bias, 0)

    def patchify_image_channelwise(self, x):
        """
        Patchify 3D image into separate token streams per channel.

        Args:
            x: Input tensor [B, C, D, H, W]

        Returns:
            List of C tensors, each of shape [B, num_patches, patch_volume]
        """
        B, C, D, H, W = x.shape
        assert D == self.input_size[0] and H == self.input_size[1] and W == self.input_size[2], \
            f"Input size mismatch: expected {self.input_size}, got ({D}, {H}, {W})"

        gd, gh, gw = self.grid_size
        pd, ph, pw = self.patch_size

        # Reshape to extract patches per channel
        # [B, C, D, H, W] -> [B, C, gd, pd, gh, ph, gw, pw]
        x = x.reshape(B, C, gd, pd, gh, ph, gw, pw)
        # [B, C, gd, gh, gw, pd, ph, pw]
        x = x.permute(0, 1, 2, 4, 6, 3, 5, 7)
        # [B, C, num_patches, patch_volume]
        x = x.reshape(B, C, gd * gh * gw, pd * ph * pw)

        # Split into list per channel
        channels = [x[:, c, :, :] for c in range(C)]
        return channels

    def forward_features(self, x):
        """Extract features from input using dual-stream encoder."""
        B = x.shape[0]

        # Patchify per channel: List of [B, num_patches, patch_volume]
        x_channels = self.patchify_image_channelwise(x)

        # Embed patches and add CLS tokens
        embedded_channels = []
        for i, (x_ch, embed, cls_token) in enumerate(
            zip(x_channels, self.patch_embeds, self.cls_tokens)):

            # Project patches
            x_emb = embed(x_ch)  # [B, num_patches, embed_dim]

            # Add CLS token
            cls = cls_token.expand(B, -1, -1)
            x_emb = torch.cat([cls, x_emb], dim=1)  # [B, 1 + num_patches, embed_dim]

            # Add position embedding (zero for CLS position)
            pos_embed = self.encoder_pos_embed.expand(B, -1, -1)
            cls_pe = torch.zeros(B, 1, self.embed_dim, device=x.device)
            pos_embed_full = torch.cat([cls_pe, pos_embed], dim=1)

            x_emb = self.pos_drop(x_emb + pos_embed_full)
            embedded_channels.append(x_emb)

        # Process through dual-channel transformer blocks
        x_ch0, x_ch1 = embedded_channels[0], embedded_channels[1]
        for block in self.encoder_blocks:
            x_ch0, x_ch1 = block(x_ch0, x_ch1)

        # Normalize
        x_ch0 = self.encoder_norms[0](x_ch0)
        x_ch1 = self.encoder_norms[1](x_ch1)

        # Pool features
        if self.use_global_pool:
            # Global average pooling (skip CLS token)
            feat_ch0 = x_ch0[:, 1:, :].mean(dim=1)  # [B, embed_dim]
            feat_ch1 = x_ch1[:, 1:, :].mean(dim=1)  # [B, embed_dim]
        else:
            # Use CLS tokens
            feat_ch0 = x_ch0[:, 0]  # [B, embed_dim]
            feat_ch1 = x_ch1[:, 0]  # [B, embed_dim]

        # Combine channel features
        if self.pool_mode == 'concat':
            features = torch.cat([feat_ch0, feat_ch1], dim=-1)  # [B, embed_dim * 2]
        elif self.pool_mode == 'mean':
            features = (feat_ch0 + feat_ch1) / 2  # [B, embed_dim]
        elif self.pool_mode == 'sum':
            features = feat_ch0 + feat_ch1  # [B, embed_dim]
        else:
            raise ValueError(f"Unknown pool_mode: {self.pool_mode}")

        return features

    def forward(self, x):
        """Forward pass."""
        x = self.forward_features(x)
        x = self.head(x)
        return x

    def forward_from_embeddings(self, embeddings):
        """
        Forward pass using precomputed embeddings (fast mode).

        Args:
            embeddings: Precomputed MAE embeddings [B, feature_dim]
                       feature_dim = embed_dim * 2 if pool_mode='concat' else embed_dim

        Returns:
            logits: [B, num_classes]
        """
        return self.head(embeddings)

    def get_num_layers(self):
        return len(self.encoder_blocks)

    def get_feature_dim(self):
        """Return the feature dimension based on pool mode."""
        if self.pool_mode == 'concat':
            return self.embed_dim * self.in_chans
        return self.embed_dim

    def load_mae_encoder(self, state_dict, strict=False):
        """
        Load pretrained MAE3DChannelCrossAttention encoder weights.

        The weights are loaded from the encoder part of MAE3DChannelCrossAttention.
        Key mappings:
        - patch_embeds -> patch_embeds
        - cls_tokens -> cls_tokens
        - encoder_pos_embed -> encoder_pos_embed
        - encoder_blocks -> encoder_blocks
        - encoder_norms -> encoder_norms

        Args:
            state_dict: State dict from MAE3DChannelCrossAttention checkpoint
            strict: Whether to strictly enforce key matching
        """
        encoder_dict = {}

        for key, value in state_dict.items():
            # Skip decoder and other non-encoder weights
            if any(skip in key for skip in ['decoder', 'mask_token', 'encoder_to_decoder']):
                continue

            # Map weights directly (architecture matches exactly)
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

        # Load weights
        msg = self.load_state_dict(encoder_dict, strict=strict)
        print(f"Loaded MAE cross-attention encoder weights:")
        print(f"  Missing keys: {len(msg.missing_keys)}")
        print(f"  Unexpected keys: {len(msg.unexpected_keys)}")

        if msg.missing_keys:
            print(f"  Missing: {msg.missing_keys[:5]}...")  # Show first 5
        if msg.unexpected_keys:
            print(f"  Unexpected: {msg.unexpected_keys[:5]}...")

        return msg
