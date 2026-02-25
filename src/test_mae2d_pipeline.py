"""
Test script for MAE2D pipeline.

This script tests:
1. Data loading with max projection
2. Model creation
3. Forward pass
4. Unpatchify/visualization

Usage:
    python src/test_mae2d_pipeline.py
"""

import os
import sys
import torch
import numpy as np

# Add src to path
sys.path.insert(0, os.path.dirname(__file__))

from utils.utils import get_conf
from lib.models.mae2d import MAE2D
from lib.networks.mae_vit import MAEViTEncoder, MAEViTDecoder
from data.opencell.dataset import OpenCellDataset
from data.opencell.transforms import get_opencell_2d_train_transforms


def test_data_loading():
    """Test data loading with max projection."""
    print("\n" + "="*60)
    print("TEST 1: Data Loading with Max Projection")
    print("="*60)

    csv_path = "/ictstr01/groups/labs/marr/qscd01/datasets/SingleCellImagesDataset/opencell/opencell_dataset/single_cells/metadata/dataset1/train.csv"

    # Create dataset with max projection
    transform = get_opencell_2d_train_transforms(flip_prob=0.0, rotate_prob=0.0)
    dataset = OpenCellDataset(
        csv_path=csv_path,
        split='train',
        transform=transform,
        cache_rate=0.0,
        num_workers=0,
        use_max_projection=True
    )

    print(f"✓ Dataset created: {len(dataset)} samples")

    # Load one sample
    sample = dataset[0]
    image = sample['image']

    print(f"✓ Sample loaded")
    print(f"  Shape: {image.shape}")
    print(f"  Expected: [C, Y, X] = [2, 176, 176]")
    print(f"  Dtype: {image.dtype}")
    print(f"  Min: {image.min():.4f}, Max: {image.max():.4f}, Mean: {image.mean():.4f}")

    # Check shape is correct for 2D
    assert len(image.shape) == 3, f"Expected 3D tensor (C, Y, X), got {image.shape}"
    assert image.shape[0] == 2, f"Expected 2 channels, got {image.shape[0]}"

    print("✓ Data loading test PASSED!")
    return dataset


def test_model_creation(args):
    """Test model creation."""
    print("\n" + "="*60)
    print("TEST 2: Model Creation")
    print("="*60)

    model = MAE2D(
        encoder=MAEViTEncoder,
        decoder=MAEViTDecoder,
        args=args
    )

    print(f"✓ Model created")
    print(f"  Encoder embed dim: {args.encoder_embed_dim}")
    print(f"  Decoder embed dim: {args.decoder_embed_dim}")
    print(f"  Patch size: {args.patch_size}")
    print(f"  Input size: {args.input_size}")
    print(f"  Grid size: {model.grid_size}")
    print(f"  Mask ratio: {args.mask_ratio}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    print("✓ Model creation test PASSED!")
    return model


def test_forward_pass(model, dataset, device='cpu'):
    """Test forward pass."""
    print("\n" + "="*60)
    print("TEST 3: Forward Pass")
    print("="*60)

    # Move model to device
    model = model.to(device)
    model.eval()

    # Create a small batch
    batch_size = 2
    batch_images = []
    for i in range(batch_size):
        sample = dataset[i]
        batch_images.append(sample['image'])

    # Stack into batch
    batch = torch.stack(batch_images).to(device)

    print(f"✓ Batch created: {batch.shape}")

    # Forward pass (without returning images)
    with torch.no_grad():
        loss = model(batch, return_image=False)

    print(f"✓ Forward pass (loss only)")
    print(f"  Loss: {loss.item():.4f}")

    # Forward pass (with returning images)
    with torch.no_grad():
        loss, original_patches, recon_patches, masked_patches = model(batch, return_image=True)

    print(f"✓ Forward pass (with reconstruction)")
    print(f"  Loss: {loss.item():.4f}")
    print(f"  Original patches shape: {original_patches.shape}")
    print(f"  Reconstructed patches shape: {recon_patches.shape}")
    print(f"  Masked patches shape: {masked_patches.shape}")

    print("✓ Forward pass test PASSED!")
    return loss, original_patches, recon_patches, masked_patches


def test_unpatchify(model, patches):
    """Test unpatchifying patches back to images."""
    print("\n" + "="*60)
    print("TEST 4: Unpatchify")
    print("="*60)

    # Unpatchify
    from timm.models.layers.helpers import to_2tuple

    B = patches.shape[0]
    H, W = model.args.input_size  # Y, X
    ph, pw = model.args.patch_size
    gh, gw = H // ph, W // pw
    in_chans = model.args.in_chans

    print(f"  Patch shape: {patches.shape}")
    print(f"  Expected: [B={B}, gh*gw={gh*gw}, C*ph*pw={in_chans*ph*pw}]")

    # Unpatchify
    x = patches.reshape(B, gh, gw, in_chans, ph, pw)
    x = x.permute(0, 3, 1, 4, 2, 5)
    x = x.reshape(B, in_chans, H, W)

    print(f"✓ Unpatchified shape: {x.shape}")
    print(f"  Expected: [B={B}, C={in_chans}, Y={H}, X={W}]")

    assert x.shape == (B, in_chans, H, W), f"Unexpected shape: {x.shape}"

    print("✓ Unpatchify test PASSED!")
    return x


def main():
    print("\n" + "="*60)
    print("MAE2D Pipeline Test Suite")
    print("="*60)

    # Load config
    code_base_path = '/lustre/groups/aih/amirhossein.kardoost/codes/single-cell-foundation-model/single-cell-foundation-model/'
    config_path = os.path.join(code_base_path, 'configs/opencell_2d.yaml')
    args = get_conf(config_path)

    print(f"\nLoaded config from: {config_path}")
    print(f"  Input size: {args.input_size}")
    print(f"  Patch size: {args.patch_size}")
    print(f"  In channels: {args.in_chans}")
    print(f"  Mask ratio: {args.mask_ratio}")

    # Determine device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")

    # Run tests
    dataset = test_data_loading()
    model = test_model_creation(args)
    loss, original_patches, recon_patches, masked_patches = test_forward_pass(model, dataset, device)
    unpatchified = test_unpatchify(model, original_patches)

    # Final summary
    print("\n" + "="*60)
    print("ALL TESTS PASSED! ✓")
    print("="*60)
    print("\nThe MAE2D pipeline is ready for training!")
    print(f"\nTo start training, run:")
    print(f"  sbatch scripts/train_mae2d_1gpu.sbatch")
    print(f"\nOr for interactive testing:")
    print(f"  python src/train_mae2d_opencell.py")
    print()


if __name__ == '__main__':
    main()
