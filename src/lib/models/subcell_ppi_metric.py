"""
SubCell PPI Metric Learning Model.

Uses precomputed embeddings with MLP projection head for PPI prediction.
Architecture: Embedding -> MLP Projection -> L2 Normalize -> Cosine Similarity
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SubCellPPIMetric(nn.Module):
    """
    PPI Metric Learning Model for precomputed SubCell embeddings.

    Architecture:
    1. MLP Projection Head (maps embeddings to similarity space)
    2. L2 Normalization
    3. Cosine Similarity for pair scoring

    This is equivalent to the projection head of PPIMetric2D/3D,
    but without the ViT encoder (since embeddings are precomputed).
    """

    def __init__(self,
                 input_dim=384,
                 proj_hidden_dim=512,
                 proj_output_dim=128,
                 proj_num_layers=2):
        """
        Args:
            input_dim: Input embedding dimension (e.g., 384 for PCA-reduced SubCell)
            proj_hidden_dim: Hidden dimension in projection MLP
            proj_output_dim: Output dimension for similarity computation
            proj_num_layers: Number of MLP layers (2 or 3)
        """
        super().__init__()

        self.input_dim = input_dim
        self.proj_hidden_dim = proj_hidden_dim
        self.proj_output_dim = proj_output_dim
        self.proj_num_layers = proj_num_layers

        # Build projection head
        if proj_num_layers == 2:
            self.projection = nn.Sequential(
                nn.Linear(input_dim, proj_hidden_dim),
                nn.BatchNorm1d(proj_hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(proj_hidden_dim, proj_output_dim)
            )
        elif proj_num_layers == 3:
            self.projection = nn.Sequential(
                nn.Linear(input_dim, proj_hidden_dim),
                nn.BatchNorm1d(proj_hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(proj_hidden_dim, proj_hidden_dim),
                nn.BatchNorm1d(proj_hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(proj_hidden_dim, proj_output_dim)
            )
        else:
            raise ValueError(f"proj_num_layers must be 2 or 3, got {proj_num_layers}")

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward_one(self, x):
        """
        Forward pass for a single embedding.

        Args:
            x: Input embedding [B, input_dim]

        Returns:
            L2-normalized projection [B, proj_output_dim]
        """
        z = self.projection(x)
        z = F.normalize(z, p=2, dim=-1)
        return z

    def forward(self, emb1, emb2):
        """
        Forward pass for a pair of embeddings.

        Args:
            emb1: First embedding [B, input_dim]
            emb2: Second embedding [B, input_dim]

        Returns:
            similarity: Cosine similarity scores [B]
            z1, z2: Normalized projections [B, proj_output_dim]
        """
        z1 = self.forward_one(emb1)
        z2 = self.forward_one(emb2)

        # Cosine similarity (dot product of normalized vectors)
        similarity = (z1 * z2).sum(dim=-1)

        return similarity, z1, z2

    def get_embedding(self, x):
        """
        Get the projected embedding for a single input.

        Args:
            x: Input embedding [B, input_dim]

        Returns:
            L2-normalized projection [B, proj_output_dim]
        """
        return self.forward_one(x)
