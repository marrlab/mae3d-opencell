"""
PPI Metric Learning Model using SubCell Embeddings.

Architecture: SubCell Embedding -> MLP Projection -> L2 Normalize -> Cosine Similarity

This is a lightweight model that only trains a projection head on top of
precomputed SubCell embeddings for fair comparison with MAE-based models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PPIMetricSubCell(nn.Module):
    """
    PPI Metric Learning model using precomputed SubCell embeddings.

    Only contains a projection head (MLP) that maps embeddings to a
    normalized space where cosine similarity indicates interaction likelihood.
    """

    def __init__(
        self,
        embed_dim=384,  # SubCell embedding dimension (after PCA)
        proj_hidden_dim=512,
        proj_output_dim=128,
        proj_num_layers=2,
        dropout=0.1,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.proj_output_dim = proj_output_dim

        # Build projection head (MLP)
        layers = []
        in_dim = embed_dim

        for i in range(proj_num_layers - 1):
            layers.extend([
                nn.Linear(in_dim, proj_hidden_dim),
                nn.LayerNorm(proj_hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            in_dim = proj_hidden_dim

        # Final projection layer
        layers.append(nn.Linear(in_dim, proj_output_dim))

        self.projection_head = nn.Sequential(*layers)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward_embedding(self, embedding):
        """
        Forward pass to get normalized embeddings.

        Args:
            embedding: SubCell embedding [B, embed_dim]

        Returns:
            Normalized projection [B, proj_output_dim]
        """
        z = self.projection_head(embedding)
        z = F.normalize(z, dim=-1)
        return z

    def forward(self, embedding1, embedding2):
        """
        Forward pass for a pair of embeddings.

        Args:
            embedding1: First protein embedding [B, embed_dim]
            embedding2: Second protein embedding [B, embed_dim]

        Returns:
            z1: Normalized embedding of first protein [B, proj_output_dim]
            z2: Normalized embedding of second protein [B, proj_output_dim]
            similarity: Cosine similarity [B]
        """
        z1 = self.forward_embedding(embedding1)
        z2 = self.forward_embedding(embedding2)

        # Cosine similarity (embeddings are already normalized)
        similarity = (z1 * z2).sum(dim=-1)

        return z1, z2, similarity

    def get_num_params(self):
        """Return number of parameters."""
        return sum(p.numel() for p in self.parameters())

    def get_trainable_params(self):
        """Return number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
