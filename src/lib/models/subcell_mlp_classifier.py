"""
MLP Classifier for protein localization using precomputed SubCell embeddings.
Uses the same MLP structure as the PPI metric learning projection head.
"""

import torch
import torch.nn as nn


class SubCellMLPClassifier(nn.Module):
    """
    MLP Classifier for protein localization from SubCell embeddings.

    Architecture follows the same pattern as MLPProjectionHead:
    - 2-layer: Input -> Linear -> BatchNorm -> ReLU -> Linear -> Output
    - 3-layer: Input -> Linear -> BatchNorm -> ReLU -> Linear -> BatchNorm -> ReLU -> Linear -> Output

    Unlike MLPProjectionHead, this classifier does NOT apply L2 normalization
    at the output (since we need raw logits for BCE loss).

    Optionally supports a projection layer to reduce embedding dimension before
    classification, enabling fair comparison with other models (e.g., MAE3D with 384-dim).
    """

    def __init__(self, input_dim, hidden_dim, num_classes, num_layers=2, dropout=0.0,
                 project_dim=None):
        """
        Args:
            input_dim: Input embedding dimension (e.g., 1536 for SubCell)
            hidden_dim: Hidden layer dimension (used when num_layers > 1)
            num_classes: Number of output classes (e.g., 17 for localization)
            num_layers: Number of layers (1, 2, or 3)
            dropout: Dropout rate (applied after ReLU)
            project_dim: If specified, project input to this dimension first.
                        This enables fair comparison with other models.
                        E.g., project_dim=384 to match MAE3D embedding dimension.
                        The projection is a simple linear layer (no activation).
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.project_dim = project_dim

        # Optional projection layer to reduce dimension
        if project_dim is not None:
            self.projection = nn.Linear(input_dim, project_dim)
            classifier_input_dim = project_dim
            print(f"  Using projection: {input_dim} -> {project_dim}")
        else:
            self.projection = None
            classifier_input_dim = input_dim

        # Build classifier
        if num_layers == 1:
            # Simple linear classifier (like the original ViT classifier head)
            # This matches MAE3D: Linear(embed_dim, num_classes)
            self.mlp = nn.Linear(classifier_input_dim, num_classes)
        elif num_layers == 2:
            layers = [
                nn.Linear(classifier_input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
            ]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(hidden_dim, num_classes))
            self.mlp = nn.Sequential(*layers)
        elif num_layers == 3:
            layers = [
                nn.Linear(classifier_input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
            ]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
            ])
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(hidden_dim, num_classes))
            self.mlp = nn.Sequential(*layers)
        else:
            raise ValueError(f"num_layers must be 1, 2 or 3, got {num_layers}")

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

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input embeddings [B, input_dim]

        Returns:
            Logits [B, num_classes]
        """
        # Apply projection if specified
        if self.projection is not None:
            x = self.projection(x)

        return self.mlp(x)


# Alias for consistency with other models
MLPLocalizationClassifier = SubCellMLPClassifier
