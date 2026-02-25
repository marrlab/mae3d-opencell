"""
FFT 3D Loss for MAE training.

This loss computes the L1 distance between FFT magnitudes of predicted and target 3D volumes.
Using FFT loss encourages the model to learn proper frequency distributions:
- Low frequencies: global structure
- High frequencies: sharp edges and fine details

Key design choices for stability:
1. Log transform: Compresses dynamic range of FFT magnitudes
2. L1 loss: More robust to outliers than L2
3. Magnitude only: Phase matching is unstable
4. Orthonormal FFT: Keeps values in reasonable range
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FFT3DLoss(nn.Module):
    """
    3D FFT-based frequency domain loss.

    Computes L1 loss between log-transformed FFT magnitudes of predicted and target volumes.

    Args:
        use_log: If True, apply log1p transform to compress dynamic range (recommended)
        norm_type: FFT normalization type ('ortho' recommended for stability)
        eps: Small constant for numerical stability in gradient computation
    """

    def __init__(self, use_log: bool = True, norm_type: str = 'ortho', eps: float = 1e-8):
        super().__init__()
        self.use_log = use_log
        self.norm_type = norm_type
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute FFT loss between predicted and target 3D volumes.

        Args:
            pred: Predicted volume [B, D, H, W] or [B, C, D, H, W]
            target: Target volume [B, D, H, W] or [B, C, D, H, W]

        Returns:
            Scalar loss value
        """
        # Handle both 4D and 5D inputs
        if pred.dim() == 5:
            # [B, C, D, H, W] - compute FFT over spatial dims only
            fft_dims = (-3, -2, -1)
        elif pred.dim() == 4:
            # [B, D, H, W]
            fft_dims = (-3, -2, -1)
        else:
            raise ValueError(f"Expected 4D or 5D tensor, got {pred.dim()}D")

        # Compute 3D FFT
        pred_fft = torch.fft.fftn(pred, dim=fft_dims, norm=self.norm_type)
        target_fft = torch.fft.fftn(target, dim=fft_dims, norm=self.norm_type)

        # Take magnitude with eps for gradient stability
        # When magnitude is very small, gradients of abs() can be unstable
        pred_mag = torch.abs(pred_fft) + self.eps
        target_mag = torch.abs(target_fft) + self.eps

        # Log transform to compress dynamic range
        if self.use_log:
            pred_mag = torch.log1p(pred_mag)
            target_mag = torch.log1p(target_mag)

        # L1 loss (more stable than L2 for frequency domain)
        loss = F.l1_loss(pred_mag, target_mag)

        return loss


class FFT3DLossWeighted(FFT3DLoss):
    """
    FFT3DLoss with frequency band weighting.

    Allows emphasizing certain frequency ranges (e.g., high frequencies for sharper edges).

    Args:
        use_log: If True, apply log1p transform
        norm_type: FFT normalization type
        eps: Small constant for numerical stability
        low_freq_weight: Weight for low frequency components
        high_freq_weight: Weight for high frequency components
        cutoff_ratio: Ratio of frequency spectrum to consider as "low" (0-1)
    """

    def __init__(
        self,
        use_log: bool = True,
        norm_type: str = 'ortho',
        eps: float = 1e-8,
        low_freq_weight: float = 1.0,
        high_freq_weight: float = 1.0,
        cutoff_ratio: float = 0.25
    ):
        super().__init__(use_log=use_log, norm_type=norm_type, eps=eps)
        self.low_freq_weight = low_freq_weight
        self.high_freq_weight = high_freq_weight
        self.cutoff_ratio = cutoff_ratio
        self._freq_mask = None
        self._cached_shape = None

    def _get_freq_mask(self, shape: tuple, device: torch.device) -> torch.Tensor:
        """Create frequency mask for weighting (cached for efficiency)."""
        if self._freq_mask is not None and self._cached_shape == shape:
            return self._freq_mask.to(device)

        # Create frequency distance from center
        d, h, w = shape[-3:]
        freq_d = torch.fft.fftfreq(d).reshape(-1, 1, 1)
        freq_h = torch.fft.fftfreq(h).reshape(1, -1, 1)
        freq_w = torch.fft.fftfreq(w).reshape(1, 1, -1)

        # Normalized distance from DC component
        freq_dist = torch.sqrt(freq_d**2 + freq_h**2 + freq_w**2)
        max_freq = freq_dist.max()
        freq_dist = freq_dist / (max_freq + self.eps)

        # Create weight mask
        low_mask = (freq_dist <= self.cutoff_ratio).float()
        high_mask = (freq_dist > self.cutoff_ratio).float()

        weight_mask = low_mask * self.low_freq_weight + high_mask * self.high_freq_weight

        self._freq_mask = weight_mask
        self._cached_shape = shape

        return weight_mask.to(device)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute weighted FFT loss."""
        if pred.dim() == 5:
            fft_dims = (-3, -2, -1)
        elif pred.dim() == 4:
            fft_dims = (-3, -2, -1)
        else:
            raise ValueError(f"Expected 4D or 5D tensor, got {pred.dim()}D")

        # Compute 3D FFT
        pred_fft = torch.fft.fftn(pred, dim=fft_dims, norm=self.norm_type)
        target_fft = torch.fft.fftn(target, dim=fft_dims, norm=self.norm_type)

        # Take magnitude with eps for gradient stability
        pred_mag = torch.abs(pred_fft) + self.eps
        target_mag = torch.abs(target_fft) + self.eps

        # Log transform
        if self.use_log:
            pred_mag = torch.log1p(pred_mag)
            target_mag = torch.log1p(target_mag)

        # Get frequency weights
        weight_mask = self._get_freq_mask(pred.shape, pred.device)

        # Weighted L1 loss
        diff = torch.abs(pred_mag - target_mag)
        weighted_diff = diff * weight_mask
        loss = weighted_diff.mean()

        return loss
