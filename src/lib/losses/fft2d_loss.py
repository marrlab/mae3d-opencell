"""
FFT 2D Loss for MAE training on 2D images (e.g. max-projected OpenCell).

Computes L1 loss between log-transformed 2D FFT magnitudes.
Design choices are identical to FFT3DLoss; only the FFT dims differ.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FFT2DLoss(nn.Module):
    """
    2D FFT-based frequency domain loss.

    Computes L1 loss between log-transformed FFT magnitudes of predicted
    and target 2D images.

    Args:
        use_log: If True, apply log1p transform to compress dynamic range
        norm_type: FFT normalization ('ortho' recommended for stability)
        eps: Small constant for numerical stability
    """

    def __init__(self, use_log: bool = True, norm_type: str = 'ortho', eps: float = 1e-8):
        super().__init__()
        self.use_log = use_log
        self.norm_type = norm_type
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred:   [B, H, W]
            target: [B, H, W]

        Returns:
            Scalar loss value
        """
        fft_dims = (-2, -1)

        pred_fft = torch.fft.fftn(pred, dim=fft_dims, norm=self.norm_type)
        target_fft = torch.fft.fftn(target, dim=fft_dims, norm=self.norm_type)

        pred_mag = torch.abs(pred_fft) + self.eps
        target_mag = torch.abs(target_fft) + self.eps

        if self.use_log:
            pred_mag = torch.log1p(pred_mag)
            target_mag = torch.log1p(target_mag)

        return F.l1_loss(pred_mag, target_mag)
