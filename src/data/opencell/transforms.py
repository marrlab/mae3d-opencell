"""
OpenCell-specific data transforms with proper channel-wise normalization.

This module provides transform pipelines for OpenCell dataset, including:
- 3D transforms for full volumetric data
- 2D transforms for max-projected images
- Proper channel-wise normalization (CRITICAL for multi-channel data)
- Intensity-based data augmentation
"""

from monai import transforms
import numpy as np
import torch
from typing import Optional, Tuple


class CheckForNaNd(transforms.MapTransform):
    """Check for NaN/Inf values in the data."""

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            if isinstance(d[key], torch.Tensor):
                img = d[key]
            else:
                img = np.asarray(d[key])

            if np.any(np.isnan(img)) or np.any(np.isinf(img)):
                raise ValueError(f"Found NaN or Inf in {key}! "
                               f"NaN count: {np.isnan(img).sum()}, "
                               f"Inf count: {np.isinf(img).sum()}")
        return d


# ========================================
# 3D Transforms for Volumetric Data
# ========================================

def get_opencell_train_transforms(
    flip_prob: float = 0.2,
    rotate_prob: float = 0.2,
    channel_wise_norm: bool = True,
    intensity_clipping: Optional[Tuple[float, float, float, float]] = None,
    intensity_augmentation: bool = True,
    scale_intensity_prob: float = 0.1,
    scale_intensity_factor: float = 0.1,
    shift_intensity_prob: float = 0.1,
    shift_intensity_offset: float = 0.1,
    resize_to: Optional[Tuple[int, int, int]] = None,
    pad_crop_to: Optional[Tuple[int, int, int]] = None,
):
    """
    Transforms for pre-loaded TIFF arrays (training).

    Args:
        flip_prob: Probability of random flipping
        rotate_prob: Probability of random 90-degree rotation
        channel_wise_norm: If True, normalize each channel independently (RECOMMENDED)
        intensity_clipping: Optional (a_min, a_max, b_min, b_max) for intensity scaling
        intensity_augmentation: Whether to apply intensity augmentation
        scale_intensity_prob: Probability of scaling intensity
        scale_intensity_factor: Factor for random intensity scaling
        shift_intensity_prob: Probability of shifting intensity
        shift_intensity_offset: Offset for random intensity shifting

    Expects input: {"image": numpy array of shape (Z, C, Y, X)}
    Returns: {"image": tensor of shape (C, Z, Y, X)}

    Key improvements:
    - Channel-wise normalization: Each channel (nucleus, protein) normalized independently
    - Intensity augmentation: Random scale/shift for robustness
    - Optional intensity clipping: Standardize intensity ranges if needed
    """
    transform_list = [
        transforms.EnsureTyped(keys=["image"]),
        transforms.CastToTyped(keys=["image"], dtype=np.float32),

        # Check for NaN/Inf after loading
        CheckForNaNd(keys=["image"]),

        # CRITICAL: Transpose (Z,C,Y,X) → (C,Z,Y,X)
        transforms.Transposed(keys=["image"], indices=(1, 0, 2, 3)),
    ]

    # Optional: spatial harmonization (C,Z,Y,X) → (C, *target).
    # pad_crop_to center pads/crops with NO scaling (preserves physical pixel size);
    # resize_to interpolates. Prefer pad_crop_to when feeding pixel size to the model.
    if pad_crop_to is not None:
        transform_list.append(
            transforms.ResizeWithPadOrCropd(keys=["image"], spatial_size=pad_crop_to)
        )
    elif resize_to is not None:
        transform_list.append(
            transforms.Resized(keys=["image"], spatial_size=resize_to)
        )

    # Optional: Intensity clipping before normalization
    if intensity_clipping is not None:
        a_min, a_max, b_min, b_max = intensity_clipping
        transform_list.append(
            transforms.ScaleIntensityRanged(
                keys=["image"],
                a_min=a_min,
                a_max=a_max,
                b_min=b_min,
                b_max=b_max,
                clip=True,
            )
        )

    # CRITICAL: Channel-wise normalization
    # This ensures nucleus and protein channels are normalized independently
    # Without this, channels with different intensity distributions will bias the model
    transform_list.append(
        transforms.NormalizeIntensityd(
            keys="image",
            nonzero=True,  # Only normalize non-zero pixels (avoid background)
            channel_wise=channel_wise_norm,  # Normalize each channel independently
        )
    )

    # Spatial augmentation (after normalization: shape is (C, Z, Y, X))
    # spatial_axis 0=Z, 1=Y, 2=X
    transform_list.extend([
        transforms.RandFlipd(keys=["image"], prob=flip_prob, spatial_axis=0),  # Flip Z
        transforms.RandFlipd(keys=["image"], prob=flip_prob, spatial_axis=1),  # Flip Y
        transforms.RandFlipd(keys=["image"], prob=flip_prob, spatial_axis=2),  # Flip X
        transforms.RandRotate90d(
            keys=["image"],
            prob=rotate_prob,
            spatial_axes=(1, 2)  # Rotate in Y-X plane
        ),
    ])

    # Intensity augmentation (for robustness to imaging variations)
    if intensity_augmentation:
        transform_list.extend([
            transforms.RandScaleIntensityd(
                keys="image",
                factors=scale_intensity_factor,
                prob=scale_intensity_prob
            ),
            transforms.RandShiftIntensityd(
                keys="image",
                offsets=shift_intensity_offset,
                prob=shift_intensity_prob
            ),
        ])

    transform_list.extend([
        transforms.EnsureTyped(keys=["image"]),
        transforms.ToTensord(keys=["image"]),
    ])

    return transforms.Compose(transform_list)


def get_opencell_val_transforms(
    channel_wise_norm: bool = True,
    intensity_clipping: Optional[Tuple[float, float, float, float]] = None,
    resize_to: Optional[Tuple[int, int, int]] = None,
    pad_crop_to: Optional[Tuple[int, int, int]] = None,
):
    """
    Transforms for pre-loaded TIFF arrays (validation).

    Args:
        channel_wise_norm: If True, normalize each channel independently (RECOMMENDED)
        intensity_clipping: Optional (a_min, a_max, b_min, b_max) for intensity scaling

    Expects input: {"image": numpy array of shape (Z, C, Y, X)}
    Returns: {"image": tensor of shape (C, Z, Y, X)}

    Note: No augmentation for validation, only normalization
    """
    transform_list = [
        transforms.EnsureTyped(keys=["image"]),
        transforms.CastToTyped(keys=["image"], dtype=np.float32),

        # Check for NaN/Inf after loading
        CheckForNaNd(keys=["image"]),

        # CRITICAL: Transpose (Z,C,Y,X) → (C,Z,Y,X)
        transforms.Transposed(keys=["image"], indices=(1, 0, 2, 3)),
    ]

    # Optional: spatial harmonization (C,Z,Y,X) → (C, *target).
    # pad_crop_to center pads/crops with NO scaling; resize_to interpolates.
    if pad_crop_to is not None:
        transform_list.append(
            transforms.ResizeWithPadOrCropd(keys=["image"], spatial_size=pad_crop_to)
        )
    elif resize_to is not None:
        transform_list.append(
            transforms.Resized(keys=["image"], spatial_size=resize_to)
        )

    # Optional: Intensity clipping
    if intensity_clipping is not None:
        a_min, a_max, b_min, b_max = intensity_clipping
        transform_list.append(
            transforms.ScaleIntensityRanged(
                keys=["image"],
                a_min=a_min,
                a_max=a_max,
                b_min=b_min,
                b_max=b_max,
                clip=True,
            )
        )

    # CRITICAL: Channel-wise normalization
    transform_list.append(
        transforms.NormalizeIntensityd(
            keys="image",
            nonzero=True,
            channel_wise=channel_wise_norm,
        )
    )

    transform_list.extend([
        transforms.EnsureTyped(keys=["image"]),
        transforms.ToTensord(keys=["image"]),
    ])

    return transforms.Compose(transform_list)


# ========================================
# 2D Transforms for Max-Projected Images
# ========================================

def get_opencell_2d_train_transforms(
    flip_prob: float = 0.2,
    rotate_prob: float = 0.2,
    channel_wise_norm: bool = True,
    intensity_clipping: Optional[Tuple[float, float, float, float]] = None,
    intensity_augmentation: bool = True,
    scale_intensity_prob: float = 0.1,
    scale_intensity_factor: float = 0.1,
    shift_intensity_prob: float = 0.1,
    shift_intensity_offset: float = 0.1,
):
    """
    Transforms for max-projected 2D images (training).

    Args:
        flip_prob: Probability of random flipping
        rotate_prob: Probability of random 90-degree rotation
        channel_wise_norm: If True, normalize each channel independently (RECOMMENDED)
        intensity_clipping: Optional (a_min, a_max, b_min, b_max) for intensity scaling
        intensity_augmentation: Whether to apply intensity augmentation
        scale_intensity_prob: Probability of scaling intensity
        scale_intensity_factor: Factor for random intensity scaling
        shift_intensity_prob: Probability of shifting intensity
        shift_intensity_offset: Offset for random intensity shifting

    Expects input: {"image": numpy array of shape (C, Y, X)} (after max projection)
    Returns: {"image": tensor of shape (C, Y, X)}
    """
    transform_list = [
        transforms.EnsureTyped(keys=["image"]),
        transforms.CastToTyped(keys=["image"], dtype=np.float32),

        # Check for NaN/Inf after loading
        CheckForNaNd(keys=["image"]),
    ]

    # Optional: Intensity clipping
    if intensity_clipping is not None:
        a_min, a_max, b_min, b_max = intensity_clipping
        transform_list.append(
            transforms.ScaleIntensityRanged(
                keys=["image"],
                a_min=a_min,
                a_max=a_max,
                b_min=b_min,
                b_max=b_max,
                clip=True,
            )
        )

    # CRITICAL: Channel-wise normalization
    transform_list.append(
        transforms.NormalizeIntensityd(
            keys="image",
            nonzero=True,
            channel_wise=channel_wise_norm,
        )
    )

    # Spatial augmentation (shape is already (C, Y, X))
    # spatial_axis 0=Y, 1=X
    transform_list.extend([
        transforms.RandFlipd(keys=["image"], prob=flip_prob, spatial_axis=0),  # Flip Y
        transforms.RandFlipd(keys=["image"], prob=flip_prob, spatial_axis=1),  # Flip X
        transforms.RandRotate90d(
            keys=["image"],
            prob=rotate_prob,
            spatial_axes=(0, 1)  # Rotate in Y-X plane
        ),
    ])

    # Intensity augmentation
    if intensity_augmentation:
        transform_list.extend([
            transforms.RandScaleIntensityd(
                keys="image",
                factors=scale_intensity_factor,
                prob=scale_intensity_prob
            ),
            transforms.RandShiftIntensityd(
                keys="image",
                offsets=shift_intensity_offset,
                prob=shift_intensity_prob
            ),
        ])

    transform_list.extend([
        transforms.EnsureTyped(keys=["image"]),
        transforms.ToTensord(keys=["image"]),
    ])

    return transforms.Compose(transform_list)


def get_opencell_2d_val_transforms(
    channel_wise_norm: bool = True,
    intensity_clipping: Optional[Tuple[float, float, float, float]] = None,
):
    """
    Transforms for max-projected 2D images (validation).

    Args:
        channel_wise_norm: If True, normalize each channel independently (RECOMMENDED)
        intensity_clipping: Optional (a_min, a_max, b_min, b_max) for intensity scaling

    Expects input: {"image": numpy array of shape (C, Y, X)} (after max projection)
    Returns: {"image": tensor of shape (C, Y, X)}
    """
    transform_list = [
        transforms.EnsureTyped(keys=["image"]),
        transforms.CastToTyped(keys=["image"], dtype=np.float32),

        # Check for NaN/Inf after loading
        CheckForNaNd(keys=["image"]),
    ]

    # Optional: Intensity clipping
    if intensity_clipping is not None:
        a_min, a_max, b_min, b_max = intensity_clipping
        transform_list.append(
            transforms.ScaleIntensityRanged(
                keys=["image"],
                a_min=a_min,
                a_max=a_max,
                b_min=b_min,
                b_max=b_max,
                clip=True,
            )
        )

    # CRITICAL: Channel-wise normalization
    transform_list.append(
        transforms.NormalizeIntensityd(
            keys="image",
            nonzero=True,
            channel_wise=channel_wise_norm,
        )
    )

    transform_list.extend([
        transforms.EnsureTyped(keys=["image"]),
        transforms.ToTensord(keys=["image"]),
    ])

    return transforms.Compose(transform_list)

# ========================================
# FOV (Field-of-View) Transforms
# ========================================

def get_opencell_fov_train_transforms(
    crop_size: Optional[Tuple[int, int, int]] = None,
    flip_prob: float = 0.2,
    rotate_prob: float = 0.2,
    channel_wise_norm: bool = True,
    intensity_clipping: Optional[Tuple[float, float, float, float]] = None,
    intensity_augmentation: bool = True,
    scale_intensity_prob: float = 0.1,
    scale_intensity_factor: float = 0.1,
    shift_intensity_prob: float = 0.1,
    shift_intensity_offset: float = 0.1,
):
    """
    Transforms for FOV (field-of-view) images (training).

    FOV images are larger than single cells:
    - FOV: (51, 2, 600, 600) - uint16
    - Single cell: (100, 2, 176, 176) - float16

    This transform includes random cropping to make FOV patches comparable
    to single-cell images for training.

    Args:
        crop_size: Size to crop from FOV (Z, H, W). Default (51, 176, 176) matches
                  single cell spatial size with FOV's native Z dimension
        flip_prob: Probability of random flipping
        rotate_prob: Probability of random 90-degree rotation
        channel_wise_norm: If True, normalize each channel independently (RECOMMENDED)
        intensity_clipping: Optional (a_min, a_max, b_min, b_max) for intensity scaling
        intensity_augmentation: Whether to apply intensity augmentation
        scale_intensity_prob: Probability of scaling intensity
        scale_intensity_factor: Factor for random intensity scaling
        shift_intensity_prob: Probability of shifting intensity
        shift_intensity_offset: Offset for random intensity shifting

    Expects input: {"image": numpy array of shape (Z, C, Y, X) = (51, 2, 600, 600), dtype=float32}
    Returns: {"image": tensor of shape (C, Z, Y, X) = (2, 51, 176, 176)}

    Key features:
    - Random spatial cropping to single-cell size (makes FOV compatible with single-cell models)
    - Channel-wise normalization (CRITICAL for multi-channel)
    - Same augmentations as single-cell for consistency
    """
    transform_list = [
        transforms.EnsureTyped(keys=["image"]),
        transforms.CastToTyped(keys=["image"], dtype=np.float32),

        # Check for NaN/Inf after loading
        CheckForNaNd(keys=["image"]),

        # CRITICAL: Transpose (Z,C,Y,X) → (C,Z,Y,X)
        transforms.Transposed(keys=["image"], indices=(1, 0, 2, 3)),
    ]

    # Optional: Intensity clipping before normalization
    if intensity_clipping is not None:
        a_min, a_max, b_min, b_max = intensity_clipping
        transform_list.append(
            transforms.ScaleIntensityRanged(
                keys=["image"],
                a_min=a_min,
                a_max=a_max,
                b_min=b_min,
                b_max=b_max,
                clip=True,
            )
        )

    # CRITICAL: Random spatial crop to single-cell size
    # This makes FOV patches comparable to single cells for training
    # Shape after transpose: (C, Z, Y, X) = (2, 51, 600, 600)
    # After crop: (2, 51, 176, 176) or custom crop_size
    transform_list.append(
        transforms.RandSpatialCropd(
            keys=["image"],
            roi_size=crop_size,  # (Z, H, W)
            random_center=True,   # Randomly select crop center
        )
    )

    # CRITICAL: Channel-wise normalization
    # Applied AFTER cropping for efficiency
    transform_list.append(
        transforms.NormalizeIntensityd(
            keys="image",
            nonzero=True,
            channel_wise=channel_wise_norm,
        )
    )

    # Spatial augmentation (after normalization and cropping)
    # spatial_axis 0=Z, 1=Y, 2=X (after transpose)
    transform_list.extend([
        transforms.RandFlipd(keys=["image"], prob=flip_prob, spatial_axis=0),  # Flip Z
        transforms.RandFlipd(keys=["image"], prob=flip_prob, spatial_axis=1),  # Flip Y
        transforms.RandFlipd(keys=["image"], prob=flip_prob, spatial_axis=2),  # Flip X
        transforms.RandRotate90d(
            keys=["image"],
            prob=rotate_prob,
            spatial_axes=(1, 2)  # Rotate in Y-X plane
        ),
    ])

    # Intensity augmentation
    if intensity_augmentation:
        transform_list.extend([
            transforms.RandScaleIntensityd(
                keys="image",
                factors=scale_intensity_factor,
                prob=scale_intensity_prob
            ),
            transforms.RandShiftIntensityd(
                keys="image",
                offsets=shift_intensity_offset,
                prob=shift_intensity_prob
            ),
        ])

    transform_list.extend([
        transforms.EnsureTyped(keys=["image"]),
        transforms.ToTensord(keys=["image"]),
    ])

    return transforms.Compose(transform_list)


def get_opencell_fov_val_transforms(
    crop_size: Optional[Tuple[int, int, int]] = None,
    channel_wise_norm: bool = True,
    intensity_clipping: Optional[Tuple[float, float, float, float]] = None,
    use_center_crop: bool = False,
):
    """
    Transforms for FOV images (validation).

    Args:
        crop_size: Size to crop from FOV (Z, H, W). If None, no cropping (uses full FOV)
        channel_wise_norm: If True, normalize each channel independently (RECOMMENDED)
        intensity_clipping: Optional (a_min, a_max, b_min, b_max) for intensity scaling
        use_center_crop: If True and crop_size is set, use center crop instead of random

    Expects input: {"image": numpy array of shape (Z, C, Y, X) = (51, 2, 600, 600), dtype=float32}
    Returns: {"image": tensor of shape (C, Z, Y, X)}

    Note: For validation, you can choose to:
    1. Crop to single-cell size (crop_size=(51, 176, 176)) - for single-cell model compatibility
    2. Use full FOV (crop_size=None) - for full-image reconstruction
    """
    transform_list = [
        transforms.EnsureTyped(keys=["image"]),
        transforms.CastToTyped(keys=["image"], dtype=np.float32),

        # Check for NaN/Inf
        CheckForNaNd(keys=["image"]),

        # CRITICAL: Transpose (Z,C,Y,X) → (C,Z,Y,X)
        transforms.Transposed(keys=["image"], indices=(1, 0, 2, 3)),
    ]

    # Optional: Intensity clipping
    if intensity_clipping is not None:
        a_min, a_max, b_min, b_max = intensity_clipping
        transform_list.append(
            transforms.ScaleIntensityRanged(
                keys=["image"],
                a_min=a_min,
                a_max=a_max,
                b_min=b_min,
                b_max=b_max,
                clip=True,
            )
        )

    # Optional: Cropping for validation
    if crop_size is not None:
        if use_center_crop:
            # Center crop (deterministic for validation)
            transform_list.append(
                transforms.CenterSpatialCropd(
                    keys=["image"],
                    roi_size=crop_size,
                )
            )
        else:
            # Random crop (for data augmentation during validation)
            transform_list.append(
                transforms.RandSpatialCropd(
                    keys=["image"],
                    roi_size=crop_size,
                    random_center=True,
                )
            )

    # CRITICAL: Channel-wise normalization
    transform_list.append(
        transforms.NormalizeIntensityd(
            keys="image",
            nonzero=True,
            channel_wise=channel_wise_norm,
        )
    )

    transform_list.extend([
        transforms.EnsureTyped(keys=["image"]),
        transforms.ToTensord(keys=["image"]),
    ])

    return transforms.Compose(transform_list)


# FOV 2D Transforms (Max-Projected) - Similar to above but for 2D
def get_opencell_fov_2d_train_transforms(
    crop_size: Optional[Tuple[int, int]] = None,
    flip_prob: float = 0.2,
    rotate_prob: float = 0.2,
    channel_wise_norm: bool = True,
    intensity_clipping: Optional[Tuple[float, float, float, float]] = None,
    intensity_augmentation: bool = True,
    scale_intensity_prob: float = 0.1,
    scale_intensity_factor: float = 0.1,
    shift_intensity_prob: float = 0.1,
    shift_intensity_offset: float = 0.1,
):
    """Transforms for max-projected FOV 2D images (training)."""
    transform_list = [
        transforms.EnsureTyped(keys=["image"]),
        transforms.CastToTyped(keys=["image"], dtype=np.float32),
        CheckForNaNd(keys=["image"]),
    ]

    if intensity_clipping is not None:
        a_min, a_max, b_min, b_max = intensity_clipping
        transform_list.append(
            transforms.ScaleIntensityRanged(
                keys=["image"], a_min=a_min, a_max=a_max,
                b_min=b_min, b_max=b_max, clip=True,
            )
        )

    transform_list.append(
        transforms.RandSpatialCropd(keys=["image"], roi_size=crop_size, random_center=True)
    )

    transform_list.append(
        transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=channel_wise_norm)
    )

    transform_list.extend([
        transforms.RandFlipd(keys=["image"], prob=flip_prob, spatial_axis=0),
        transforms.RandFlipd(keys=["image"], prob=flip_prob, spatial_axis=1),
        transforms.RandRotate90d(keys=["image"], prob=rotate_prob, spatial_axes=(0, 1)),
    ])

    if intensity_augmentation:
        transform_list.extend([
            transforms.RandScaleIntensityd(keys="image", factors=scale_intensity_factor, prob=scale_intensity_prob),
            transforms.RandShiftIntensityd(keys="image", offsets=shift_intensity_offset, prob=shift_intensity_prob),
        ])

    transform_list.extend([
        transforms.EnsureTyped(keys=["image"]),
        transforms.ToTensord(keys=["image"]),
    ])

    return transforms.Compose(transform_list)


def get_opencell_fov_2d_val_transforms(
    crop_size: Optional[Tuple[int, int]] = None,
    channel_wise_norm: bool = True,
    intensity_clipping: Optional[Tuple[float, float, float, float]] = None,
    use_center_crop: bool = False,
):
    """Transforms for max-projected FOV 2D images (validation)."""
    transform_list = [
        transforms.EnsureTyped(keys=["image"]),
        transforms.CastToTyped(keys=["image"], dtype=np.float32),
        CheckForNaNd(keys=["image"]),
    ]

    if intensity_clipping is not None:
        a_min, a_max, b_min, b_max = intensity_clipping
        transform_list.append(
            transforms.ScaleIntensityRanged(
                keys=["image"], a_min=a_min, a_max=a_max,
                b_min=b_min, b_max=b_max, clip=True,
            )
        )

    if crop_size is not None:
        if use_center_crop:
            transform_list.append(transforms.CenterSpatialCropd(keys=["image"], roi_size=crop_size))
        else:
            transform_list.append(transforms.RandSpatialCropd(keys=["image"], roi_size=crop_size, random_center=True))

    transform_list.append(
        transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=channel_wise_norm)
    )

    transform_list.extend([
        transforms.EnsureTyped(keys=["image"]),
        transforms.ToTensord(keys=["image"]),
    ])

    return transforms.Compose(transform_list)


# ========================================
# Simple Transform Classes for VLM
# ========================================
# These transforms handle dictionary input {'image': array}
# for use with VLM datasets

class Compose:
    """Compose multiple transforms together."""
    def __init__(self, transforms_list):
        self.transforms = transforms_list

    def __call__(self, data):
        for t in self.transforms:
            data = t(data)
        return data


class ToTensor:
    """Convert numpy array to torch tensor."""
    def __call__(self, data):
        if isinstance(data, dict):
            x = data['image']
            if isinstance(x, np.ndarray):
                data['image'] = torch.from_numpy(x.copy()).float()
            return data
        else:
            if isinstance(data, np.ndarray):
                return torch.from_numpy(data.copy()).float()
            return data


class ChannelWiseNormalize:
    """Normalize each channel independently to zero mean and unit variance."""
    def __init__(self, eps: float = 1e-8):
        self.eps = eps

    def __call__(self, data):
        if isinstance(data, dict):
            x = data['image']
        else:
            x = data

        if isinstance(x, torch.Tensor):
            x = x.numpy()

        # x shape: (C, Z, Y, X) or (C, Y, X)
        normalized = np.zeros_like(x, dtype=np.float32)
        for c in range(x.shape[0]):
            channel = x[c]
            # Only normalize non-zero values
            nonzero_mask = channel != 0
            if nonzero_mask.sum() > 0:
                mean = channel[nonzero_mask].mean()
                std = channel[nonzero_mask].std() + self.eps
                normalized[c] = (channel - mean) / std
            else:
                normalized[c] = channel

        if isinstance(data, dict):
            data['image'] = normalized
            return data
        return normalized


class Normalize3D:
    """Normalize entire volume (all channels together) to zero mean and unit variance."""
    def __init__(self, eps: float = 1e-8):
        self.eps = eps

    def __call__(self, data):
        if isinstance(data, dict):
            x = data['image']
        else:
            x = data

        if isinstance(x, torch.Tensor):
            x = x.numpy()

        nonzero_mask = x != 0
        if nonzero_mask.sum() > 0:
            mean = x[nonzero_mask].mean()
            std = x[nonzero_mask].std() + self.eps
            result = ((x - mean) / std).astype(np.float32)
        else:
            result = x.astype(np.float32)

        if isinstance(data, dict):
            data['image'] = result
            return data
        return result


class RandomFlip3D:
    """Random flip along spatial axes."""
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, data):
        if isinstance(data, dict):
            x = data['image']
        else:
            x = data

        if isinstance(x, torch.Tensor):
            x = x.numpy()

        # x shape: (C, Z, Y, X)
        if np.random.random() < self.p:
            x = np.flip(x, axis=1).copy()  # Flip Z
        if np.random.random() < self.p:
            x = np.flip(x, axis=2).copy()  # Flip Y
        if np.random.random() < self.p:
            x = np.flip(x, axis=3).copy()  # Flip X

        if isinstance(data, dict):
            data['image'] = x
            return data
        return x


class RandomRotate90_3D:
    """Random 90-degree rotation in Y-X plane."""
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, data):
        if isinstance(data, dict):
            x = data['image']
        else:
            x = data

        if isinstance(x, torch.Tensor):
            x = x.numpy()

        if np.random.random() < self.p:
            k = np.random.randint(1, 4)  # 90, 180, or 270 degrees
            # Rotate in Y-X plane (axes 2 and 3 for shape C, Z, Y, X)
            x = np.rot90(x, k=k, axes=(2, 3)).copy()

        if isinstance(data, dict):
            data['image'] = x
            return data
        return x
