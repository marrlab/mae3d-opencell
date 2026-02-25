"""
Loss functions for MAE training.
"""

from .fft3d_loss import FFT3DLoss
from .fft2d_loss import FFT2DLoss

__all__ = ['FFT3DLoss', 'FFT2DLoss']
