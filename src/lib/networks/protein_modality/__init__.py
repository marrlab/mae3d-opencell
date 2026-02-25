"""
Protein modality conditioning modules for cross-attention.

This module provides ESM2 protein embedding conditioning components
for use with channel cross-attention models.
"""

from .esm2_conditioning import (
    ESM2CrossAttention,
    ESM2ConditionedChannelBlock,
    DualChannelTransformerBlockESM2
)

__all__ = [
    'ESM2CrossAttention',
    'ESM2ConditionedChannelBlock',
    'DualChannelTransformerBlockESM2'
]
