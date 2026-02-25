"""
Quick verification script to test that the trainer pattern is set up correctly.

This script verifies:
1. All imports work
2. Trainer classes can be instantiated
3. Configuration loading works
4. Basic method structure is correct

Run this before attempting actual training to catch any setup issues.
"""

import os
import sys

# Add src to path
sys.path.insert(0, os.path.dirname(__file__))

def test_imports():
    """Test that all necessary imports work."""
    print("Testing imports...")
    try:
        from lib.trainers import BaseTrainer, MAE3DTrainer
        from lib.models.mae3d import MAE3D
        from lib.networks.mae_vit import MAEViTEncoder, MAEViTDecoder
        from data.opencell.dataset import OpenCellDataset
        from data.opencell.transforms import get_opencell_train_transforms
        from utils.utils import get_conf, set_seed
        print("✓ All imports successful")
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False


def test_config_loading():
    """Test that configuration can be loaded."""
    print("\nTesting configuration loading...")
    try:
        from utils.utils import get_conf
        code_base_path = '/lustre/groups/aih/amirhossein.kardoost/codes/single-cell-foundation-model/single-cell-foundation-model/'
        config_path = os.path.join(code_base_path, 'configs/opencell_3d.yaml')

        if not os.path.exists(config_path):
            print(f"✗ Config file not found: {config_path}")
            return False

        args = get_conf(config_path)

        # Check required attributes
        required_attrs = ['arch', 'dataset', 'batch_size', 'epochs', 'lr',
                         'input_size', 'patch_size', 'in_chans']

        for attr in required_attrs:
            if not hasattr(args, attr):
                print(f"✗ Missing required config attribute: {attr}")
                return False

        print(f"✓ Configuration loaded successfully")
        print(f"  - Dataset: {args.dataset}")
        print(f"  - Architecture: {args.arch}")
        print(f"  - Batch size: {args.batch_size}")
        print(f"  - Epochs: {args.epochs}")
        print(f"  - Input size: {args.input_size}")
        return True
    except Exception as e:
        print(f"✗ Config loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_trainer_instantiation():
    """Test that trainers can be instantiated with mock args."""
    print("\nTesting trainer instantiation...")
    try:
        from lib.trainers import MAE3DTrainer
        from omegaconf import OmegaConf

        # Create minimal mock args
        mock_args = OmegaConf.create({
            'arch': 'vit_base',
            'enc_arch': 'MAEViTEncoder',
            'dec_arch': 'MAEViTDecoder',
            'dataset': 'opencell',
            'batch_size': 1,
            'epochs': 1,
            'lr': 1e-4,
            'beta1': 0.9,
            'beta2': 0.95,
            'weight_decay': 0.05,
            'warmup_epochs': 1,
            'input_size': [100, 176, 176],
            'patch_size': [10, 8, 8],
            'in_chans': 2,
            'encoder_embed_dim': 384,
            'encoder_depth': 6,
            'encoder_num_heads': 6,
            'decoder_embed_dim': 192,
            'decoder_depth': 4,
            'decoder_num_heads': 6,
            'mask_ratio': 0.75,
            'pos_embed_type': 'sincos',
            'patchembed': 'PatchEmbed3D',
            'ckpt_dir': '/tmp/test_ckpt',
            'rank': 0,
            'world_size': 1,
            'gpu': 0,
            'workers': 4,
            'cache_rate': 0.0,
            'csv_path': '/tmp',
            'RandFlipd_prob': 0.2,
            'RandRotate90d_prob': 0.2,
            'vis_freq': 100,
            'print_freq': 10,
            'save_freq': 1,
            'start_epoch': 0,
            'resume': None,
            'proj_name': 'test',
            'run_name': 'test_run',
        })

        # Try to instantiate trainer
        trainer = MAE3DTrainer(mock_args)

        # Check that trainer has required methods
        required_methods = ['build_model', 'build_optimizer', 'build_dataloader',
                           'epoch_train', 'run', 'wrap_model', 'save_checkpoint', 'resume']

        for method in required_methods:
            if not hasattr(trainer, method):
                print(f"✗ Trainer missing required method: {method}")
                return False
            if not callable(getattr(trainer, method)):
                print(f"✗ Trainer attribute {method} is not callable")
                return False

        print("✓ Trainer instantiated successfully")
        print(f"  - Trainer type: {type(trainer).__name__}")
        print(f"  - Has all required methods: {', '.join(required_methods)}")
        return True
    except Exception as e:
        print(f"✗ Trainer instantiation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_creation():
    """Test that MAE3D model can be created."""
    print("\nTesting MAE3D model creation...")
    try:
        from lib.models.mae3d import MAE3D
        from lib.networks.mae_vit import MAEViTEncoder, MAEViTDecoder
        from omegaconf import OmegaConf

        # Create minimal args for model
        args = OmegaConf.create({
            'input_size': [100, 176, 176],
            'patch_size': [10, 8, 8],
            'in_chans': 2,
            'encoder_embed_dim': 384,
            'encoder_depth': 6,
            'encoder_num_heads': 6,
            'decoder_embed_dim': 192,
            'decoder_depth': 4,
            'decoder_num_heads': 6,
            'mask_ratio': 0.75,
            'pos_embed_type': 'sincos',
            'patchembed': 'PatchEmbed3D',
        })

        model = MAE3D(encoder=MAEViTEncoder, decoder=MAEViTDecoder, args=args)

        print("✓ MAE3D model created successfully")
        print(f"  - Model type: {type(model).__name__}")
        print(f"  - Number of parameters: {sum(p.numel() for p in model.parameters()):,}")
        return True
    except Exception as e:
        print(f"✗ Model creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("="*60)
    print("Trainer Pattern Setup Verification")
    print("="*60)

    tests = [
        ("Imports", test_imports),
        ("Configuration Loading", test_config_loading),
        ("Trainer Instantiation", test_trainer_instantiation),
        ("Model Creation", test_model_creation),
    ]

    results = []
    for name, test_func in tests:
        result = test_func()
        results.append((name, result))

    print("\n" + "="*60)
    print("Summary")
    print("="*60)

    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")

    all_passed = all(result for _, result in results)

    print("\n" + "="*60)
    if all_passed:
        print("✓ All tests passed! Trainer pattern is set up correctly.")
        print("\nYou can now run:")
        print("  python src/train_with_trainer.py")
    else:
        print("✗ Some tests failed. Please fix the issues before training.")
    print("="*60)

    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())
