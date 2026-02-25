#!/usr/bin/env python3
"""
Evaluation script for OpenCell protein abundance prediction.
Evaluates a trained model on the test set and saves detailed results.

Usage:
    python src/evaluate_abundance.py \
        --config configs/opencell/opencell_abundance_3d.yaml \
        --checkpoint /path/to/checkpoint.pth.tar \
        --output results_test
"""

import os
import sys
import argparse
import json
from pathlib import Path
import torch
import numpy as np
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from utils.utils import get_conf
from lib.models import ViT3DClassifier, ViT2DClassifier, ViT3DCrossAttentionClassifier
from data.opencell.abundance_dataset import OpenCellAbundanceDataset
from data.opencell.transforms import (
    get_opencell_val_transforms,
    get_opencell_2d_val_transforms
)


def evaluate_model(model, dataloader, device):
    """Evaluate model on a dataset."""
    model.eval()

    all_predictions = []
    all_targets = []

    print("Running evaluation...")
    with torch.no_grad():
        for data in tqdm(dataloader, desc="Evaluating"):
            images = data['image'].to(device)
            targets = data['label'].to(device)

            # Forward pass
            predictions = model(images)

            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    # Concatenate all batches
    all_predictions = np.concatenate(all_predictions, axis=0).flatten()
    all_targets = np.concatenate(all_targets, axis=0).flatten()

    return all_predictions, all_targets


def compute_metrics(predictions, targets):
    """Compute regression metrics."""
    metrics = {}

    # Basic metrics
    metrics['mse'] = float(np.mean((predictions - targets) ** 2))
    metrics['rmse'] = float(np.sqrt(metrics['mse']))
    metrics['mae'] = float(np.mean(np.abs(predictions - targets)))

    # Correlation metrics
    try:
        pearson_corr, pearson_p = pearsonr(predictions, targets)
        metrics['pearson'] = float(pearson_corr)
        metrics['pearson_p'] = float(pearson_p)
    except:
        metrics['pearson'] = 0.0
        metrics['pearson_p'] = 1.0

    try:
        spearman_corr, spearman_p = spearmanr(predictions, targets)
        metrics['spearman'] = float(spearman_corr)
        metrics['spearman_p'] = float(spearman_p)
    except:
        metrics['spearman'] = 0.0
        metrics['spearman_p'] = 1.0

    # R-squared
    ss_res = np.sum((targets - predictions) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    metrics['r_squared'] = float(1 - (ss_res / (ss_tot + 1e-8)))

    # Additional metrics
    metrics['mean_target'] = float(np.mean(targets))
    metrics['std_target'] = float(np.std(targets))
    metrics['mean_prediction'] = float(np.mean(predictions))
    metrics['std_prediction'] = float(np.std(predictions))

    return metrics


def plot_scatter(predictions, targets, output_path, title="Predictions vs Targets"):
    """Create scatter plot of predictions vs targets."""
    plt.figure(figsize=(8, 8))
    plt.scatter(targets, predictions, alpha=0.5, s=10)

    # Add diagonal line
    min_val = min(targets.min(), predictions.min())
    max_val = max(targets.max(), predictions.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect prediction')

    # Compute correlation for title
    pearson_r, _ = pearsonr(predictions, targets)

    plt.xlabel('Ground Truth (normalized log concentration)')
    plt.ylabel('Predicted (normalized log concentration)')
    plt.title(f'{title}\nPearson r = {pearson_r:.4f}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Scatter plot saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate OpenCell Abundance Model')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint file')
    parser.add_argument('--output', type=str, default='test_results', help='Output file prefix')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'],
                       help='Which split to evaluate on')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for evaluation')
    args_cmd = parser.parse_args()

    # Load config
    args = get_conf(args_cmd.config)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Build model
    print("Building model...")
    use_2d = getattr(args, 'use_2d', False)
    arch = getattr(args, 'arch', 'ViT3DClassifier')

    model_params = {
        'input_size': tuple(args.input_size) if hasattr(args, 'input_size') else (100, 176, 176),
        'patch_size': tuple(args.patch_size) if hasattr(args, 'patch_size') else (10, 8, 8),
        'in_chans': args.in_chans,
        'num_classes': 1,  # Regression
        'embed_dim': args.encoder_embed_dim,
        'depth': args.encoder_depth,
        'num_heads': args.encoder_num_heads,
        'drop_path_rate': getattr(args, 'drop_path', 0.0),
        'pos_embed_type': getattr(args, 'pos_embed_type', 'sincos'),
        'use_global_pool': getattr(args, 'use_global_pool', True),
    }

    if arch == 'ViT3DCrossAttentionClassifier':
        model_params['cross_attention_type'] = getattr(args, 'cross_attention_type', 'position_wise')
        model_params['pool_mode'] = getattr(args, 'pool_mode', 'concat')
        model = ViT3DCrossAttentionClassifier(**model_params)
        print(f"  Using ViT3DCrossAttentionClassifier")
    elif use_2d:
        model_params['input_size'] = model_params['input_size'][1:] if len(model_params['input_size']) == 3 else model_params['input_size']
        model_params['patch_size'] = model_params['patch_size'][1:] if len(model_params['patch_size']) == 3 else model_params['patch_size']
        model = ViT2DClassifier(**model_params)
        print(f"  Using ViT2DClassifier")
    else:
        model = ViT3DClassifier(**model_params)
        print(f"  Using ViT3DClassifier")

    # Load checkpoint
    print(f"Loading checkpoint from {args_cmd.checkpoint}...")
    checkpoint = torch.load(args_cmd.checkpoint, map_location='cpu')
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    print(f"Checkpoint loaded (epoch {checkpoint.get('epoch', 'unknown')})")

    # Build dataset
    print(f"Loading {args_cmd.split} dataset...")
    if use_2d:
        transform = get_opencell_2d_val_transforms()
    else:
        transform = get_opencell_val_transforms()

    csv_path = os.path.join(args.csv_path, f'{args_cmd.split}.csv')
    dataset = OpenCellAbundanceDataset(
        csv_path=csv_path,
        abundance_csv_path=args.abundance_csv_path,
        split=args_cmd.split,
        transform=transform,
        cache_rate=0.0,
        use_max_projection=use_2d,
        target_column=args.target_column,
        log_transform=getattr(args, 'log_transform', True),
        normalize_target=getattr(args, 'normalize_target', True)
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args_cmd.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    print(f"Dataset size: {len(dataset)} samples")

    # Evaluate
    predictions, targets = evaluate_model(model, dataloader, device)

    # Compute metrics
    print("\nComputing metrics...")
    metrics = compute_metrics(predictions, targets)

    # Print results
    print("\n" + "="*80)
    print(f"RESULTS ON {args_cmd.split.upper()} SET")
    print("="*80)
    print(f"MSE:         {metrics['mse']:.6f}")
    print(f"RMSE:        {metrics['rmse']:.6f}")
    print(f"MAE:         {metrics['mae']:.6f}")
    print(f"Pearson r:   {metrics['pearson']:.4f} (p={metrics['pearson_p']:.2e})")
    print(f"Spearman rho:{metrics['spearman']:.4f} (p={metrics['spearman_p']:.2e})")
    print(f"R-squared:   {metrics['r_squared']:.4f}")
    print("="*80)

    # Save results
    output_prefix = args_cmd.output.replace('.json', '').replace('.csv', '')

    # Save JSON results
    results = {
        'config': args_cmd.config,
        'checkpoint': args_cmd.checkpoint,
        'split': args_cmd.split,
        'num_samples': len(dataset),
        'target_column': args.target_column,
        'metrics': metrics
    }

    json_path = f"{output_prefix}.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nJSON results saved to: {json_path}")

    # Save predictions CSV
    predictions_df = pd.DataFrame({
        'target': targets,
        'prediction': predictions,
        'error': predictions - targets
    })
    csv_path = f"{output_prefix}_predictions.csv"
    predictions_df.to_csv(csv_path, index=False)
    print(f"Predictions CSV saved to: {csv_path}")

    # Save scatter plot
    plot_path = f"{output_prefix}_scatter.png"
    plot_scatter(predictions, targets, plot_path,
                 title=f"Abundance Prediction ({args_cmd.split} set)")

    # Save summary CSV
    summary_csv_path = f"{output_prefix}_summary.csv"
    summary_data = [
        {'Metric': 'MSE', 'Value': f"{metrics['mse']:.6f}"},
        {'Metric': 'RMSE', 'Value': f"{metrics['rmse']:.6f}"},
        {'Metric': 'MAE', 'Value': f"{metrics['mae']:.6f}"},
        {'Metric': 'Pearson r', 'Value': f"{metrics['pearson']:.4f}"},
        {'Metric': 'Spearman rho', 'Value': f"{metrics['spearman']:.4f}"},
        {'Metric': 'R-squared', 'Value': f"{metrics['r_squared']:.4f}"},
        {'Metric': 'Num Samples', 'Value': str(len(dataset))},
    ]
    df_summary = pd.DataFrame(summary_data)
    df_summary.to_csv(summary_csv_path, index=False)
    print(f"Summary CSV saved to: {summary_csv_path}")

    print(f"\nEvaluation complete!")


if __name__ == '__main__':
    main()
