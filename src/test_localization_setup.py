#!/usr/bin/env python3
"""
Test script to verify localization classification setup.
Run this before starting training to catch any issues.

Usage:
    python src/test_localization_setup.py
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import numpy as np

print("=" * 80)
print("Testing OpenCell Localization Classification Setup")
print("=" * 80)

# Test 1: Import required modules
print("\n[1/8] Testing imports...")
try:
    from utils.utils import get_conf
    from lib.models import ViT3DClassifier, ViT2DClassifier
    from lib.trainers import LocalizationTrainer
    from data.opencell.localization_dataset import (
        OpenCellLocalizationDataset,
        LOCALIZATION_LABELS,
        LABEL_TO_IDX
    )
    from data.opencell.transforms import (
        get_opencell_train_transforms,
        get_opencell_val_transforms
    )
    print("   ✓ All imports successful")
except Exception as e:
    print(f"   ✗ Import failed: {e}")
    sys.exit(1)

# Test 2: Check localization labels
print("\n[2/8] Checking localization labels...")
print(f"   Number of labels: {len(LOCALIZATION_LABELS)}")
print(f"   Labels: {', '.join(LOCALIZATION_LABELS[:5])}... (showing first 5)")
assert len(LOCALIZATION_LABELS) == 17, "Should have 17 localization labels"
assert len(LABEL_TO_IDX) == 17, "Label mapping should have 17 entries"
print("   ✓ Label configuration correct")

# Test 3: Check config file
print("\n[3/8] Testing config loading...")
try:
    config_path = 'configs/opencell_localization_3d.yaml'
    args = get_conf(config_path)
    print(f"   Config loaded: {config_path}")
    print(f"   - Trainer: {args.trainer_name}")
    print(f"   - Architecture: {args.arch}")
    print(f"   - Num classes: {args.num_classes}")
    print(f"   - Batch size: {args.batch_size}")
    assert args.num_classes == 17, "num_classes should be 17"
    assert args.trainer_name == 'LocalizationTrainer', "trainer_name should be LocalizationTrainer"
    print("   ✓ Config loading successful")
except Exception as e:
    print(f"   ✗ Config loading failed: {e}")
    sys.exit(1)

# Test 4: Check pretrained checkpoint
print("\n[4/8] Checking pretrained checkpoint...")
pretrain_path = args.pretrain
if pretrain_path and os.path.exists(pretrain_path):
    print(f"   Checkpoint exists: {pretrain_path}")
    try:
        checkpoint = torch.load(pretrain_path, map_location='cpu')
        print(f"   Checkpoint keys: {list(checkpoint.keys())}")
        if 'state_dict' in checkpoint:
            print(f"   State dict has {len(checkpoint['state_dict'])} parameters")
        print("   ✓ Checkpoint valid")
    except Exception as e:
        print(f"   ✗ Checkpoint loading failed: {e}")
else:
    print(f"   ⚠ Checkpoint not found: {pretrain_path}")
    print("   Will train from scratch (not recommended)")

# Test 5: Test model creation
print("\n[5/8] Testing model creation...")
try:
    model = ViT3DClassifier(
        input_size=(100, 176, 176),
        patch_size=(10, 8, 8),
        in_chans=2,
        num_classes=17,
        embed_dim=384,
        depth=6,
        num_heads=6
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Model created: ViT3DClassifier")
    print(f"   Total parameters: {total_params:,}")
    print("   ✓ Model creation successful")
except Exception as e:
    print(f"   ✗ Model creation failed: {e}")
    sys.exit(1)

# Test 6: Test forward pass
print("\n[6/8] Testing forward pass...")
try:
    dummy_input = torch.randn(2, 2, 100, 176, 176)  # [B, C, D, H, W]
    with torch.no_grad():
        output = model(dummy_input)
    print(f"   Input shape: {dummy_input.shape}")
    print(f"   Output shape: {output.shape}")
    assert output.shape == (2, 17), f"Expected output shape (2, 17), got {output.shape}"
    print("   ✓ Forward pass successful")
except Exception as e:
    print(f"   ✗ Forward pass failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 7: Test data paths
print("\n[7/8] Checking data paths...")
csv_path = args.csv_path
loc_csv_path = args.localization_csv_path

train_csv = os.path.join(csv_path, 'train.csv')
val_csv = os.path.join(csv_path, 'val.csv')

paths_to_check = [
    ('CSV directory', csv_path),
    ('Train CSV', train_csv),
    ('Val CSV', val_csv),
    ('Localization CSV', loc_csv_path)
]

all_exist = True
for name, path in paths_to_check:
    if os.path.exists(path):
        print(f"   ✓ {name}: {path}")
    else:
        print(f"   ✗ {name} not found: {path}")
        all_exist = False

if not all_exist:
    print("   ⚠ Some data files are missing!")
    print("   Please check data paths in config file")

# Test 8: Test dataset creation (if data exists)
print("\n[8/8] Testing dataset creation...")
if all_exist:
    try:
        transform = get_opencell_val_transforms()
        dataset = OpenCellLocalizationDataset(
            csv_path=train_csv,
            localization_csv_path=loc_csv_path,
            split='train',
            transform=transform,
            cache_rate=0.0,
            use_max_projection=False
        )
        print(f"   Dataset created with {len(dataset)} samples")

        # Test loading one sample
        sample = dataset[0]
        print(f"   Sample keys: {list(sample.keys())}")
        print(f"   Image shape: {sample['image'].shape}")
        print(f"   Label shape: {sample['label'].shape}")
        print(f"   Label sum: {sample['label'].sum():.2f} (weighted count of labels)")

        # Check label distribution
        label_dist = dataset.get_label_distribution()
        print(f"   Top 3 most common labels:")
        for i, (label, count) in enumerate(sorted(label_dist.items(), key=lambda x: -x[1])[:3]):
            print(f"     {i+1}. {label}: {count} samples")

        print("   ✓ Dataset creation successful")
    except Exception as e:
        print(f"   ✗ Dataset creation failed: {e}")
        import traceback
        traceback.print_exc()
else:
    print("   ⊘ Skipping dataset test (data files not found)")

# Summary
print("\n" + "=" * 80)
print("Setup Test Complete!")
print("=" * 80)
print("\nNext steps:")
print("1. Verify all data paths are correct in config file")
print("2. Ensure pretrained MAE3D checkpoint exists")
print("3. Start training with:")
print("   python src/train_localization.py configs/opencell_localization_3d.yaml")
print()
