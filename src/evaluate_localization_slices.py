#!/usr/bin/env python3
"""
Evaluation script for OpenCell protein localization with slice aggregation.
Evaluates a trained ViT2DSliceAggregateClassifier on the test set.

The model processes slices 45-55 from each volume, extracts embeddings
using a pretrained MAE2D encoder, aggregates them via mean pooling,
and classifies using an MLP head.

Usage:
    python src/evaluate_localization_slices.py \
        --config configs/opencell/opencell_localization_2d_slices.yaml \
        --checkpoint /path/to/checkpoint.pth.tar \
        --output results_test.json
"""

import os
import sys
import argparse
import json
from pathlib import Path
import torch
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score
from tqdm import tqdm
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from utils.utils import get_conf
from lib.models import ViT2DSliceAggregateClassifier
from data.opencell.localization_slice_dataset import (
    OpenCellLocalizationSliceDataset,
    collate_slices
)
from data.opencell.localization_dataset import LOCALIZATION_LABELS
from data.opencell.transforms import get_opencell_2d_val_transforms


def evaluate_model(model, dataloader, device):
    """Evaluate model on a dataset."""
    model.eval()

    all_logits = []
    all_targets = []
    all_protein_names = []

    print("Running evaluation...")
    with torch.no_grad():
        for data in tqdm(dataloader, desc="Evaluating"):
            slices = data['slices'].to(device)
            mask = data['mask'].to(device)
            targets = data['label'].to(device)

            # Forward pass with slice aggregation
            with torch.cuda.amp.autocast(True):
                logits = model(slices, mask=mask)

            all_logits.append(logits.cpu())
            all_targets.append(targets.cpu())

    # Concatenate all batches
    all_logits = torch.cat(all_logits, dim=0).numpy()
    all_targets = torch.cat(all_targets, dim=0).numpy()

    return all_logits, all_targets


def compute_metrics(logits, targets, threshold=0.5):
    """Compute comprehensive metrics."""
    # Convert logits to probabilities
    probs = torch.sigmoid(torch.from_numpy(logits)).numpy()

    # Binarize targets (any weight > 0 means label is present)
    binary_targets = (targets > 0).astype(float)

    # Binary predictions
    binary_preds = (probs > threshold).astype(float)

    # Compute metrics
    metrics = {}

    # Overall metrics
    try:
        metrics['mAP'] = average_precision_score(binary_targets, probs, average='macro')
        metrics['macro_AUC'] = roc_auc_score(binary_targets, probs, average='macro')
        metrics['micro_AUC'] = roc_auc_score(binary_targets, probs, average='micro')
    except:
        metrics['mAP'] = 0.0
        metrics['macro_AUC'] = 0.0
        metrics['micro_AUC'] = 0.0

    metrics['macro_F1'] = f1_score(binary_targets, binary_preds, average='macro', zero_division=0)
    metrics['micro_F1'] = f1_score(binary_targets, binary_preds, average='micro', zero_division=0)

    # Per-class metrics
    per_class_metrics = {}
    for i, label in enumerate(LOCALIZATION_LABELS):
        class_metrics = {}

        # Only compute if label exists in dataset
        if binary_targets[:, i].sum() > 0:
            try:
                class_metrics['AP'] = average_precision_score(binary_targets[:, i], probs[:, i])
                class_metrics['AUC'] = roc_auc_score(binary_targets[:, i], probs[:, i])
            except:
                class_metrics['AP'] = 0.0
                class_metrics['AUC'] = 0.0

            class_metrics['F1'] = f1_score(binary_targets[:, i], binary_preds[:, i], zero_division=0)
            class_metrics['precision'] = ((binary_preds[:, i] == 1) & (binary_targets[:, i] == 1)).sum() / max(1, (binary_preds[:, i] == 1).sum())
            class_metrics['recall'] = ((binary_preds[:, i] == 1) & (binary_targets[:, i] == 1)).sum() / max(1, (binary_targets[:, i] == 1).sum())
            class_metrics['support'] = int(binary_targets[:, i].sum())
        else:
            class_metrics['AP'] = 0.0
            class_metrics['AUC'] = 0.0
            class_metrics['F1'] = 0.0
            class_metrics['precision'] = 0.0
            class_metrics['recall'] = 0.0
            class_metrics['support'] = 0

        per_class_metrics[label] = class_metrics

    metrics['per_class'] = per_class_metrics

    return metrics


def main():
    parser = argparse.ArgumentParser(description='Evaluate OpenCell Localization Slices Model')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint file')
    parser.add_argument('--output', type=str, default='test_results_slices', help='Output file prefix')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'],
                       help='Which split to evaluate on')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for evaluation')
    parser.add_argument('--threshold', type=float, default=0.5, help='Classification threshold')
    args_cmd = parser.parse_args()

    # Load config
    args = get_conf(args_cmd.config)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Get slice parameters from config
    slice_start = getattr(args, 'slice_start', 45)
    slice_end = getattr(args, 'slice_end', 55)
    aggregation = getattr(args, 'aggregation', 'mean')

    print(f"\nSlice aggregation settings:")
    print(f"  Slice range: {slice_start} to {slice_end}")
    print(f"  Aggregation: {aggregation}")

    # Build model
    print("\nBuilding model...")
    model_params = {
        'input_size': tuple(args.input_size) if hasattr(args, 'input_size') else (176, 176),
        'patch_size': tuple(args.patch_size) if hasattr(args, 'patch_size') else (8, 8),
        'in_chans': args.in_chans,
        'num_classes': args.num_classes,
        'embed_dim': args.encoder_embed_dim,
        'depth': args.encoder_depth,
        'num_heads': args.encoder_num_heads,
        'drop_path_rate': getattr(args, 'drop_path', 0.0),
        'pos_embed_type': getattr(args, 'pos_embed_type', 'sincos'),
        'use_global_pool': getattr(args, 'use_global_pool', True),
        'aggregation': aggregation,
    }

    model = ViT2DSliceAggregateClassifier(**model_params)
    print(f"  Using ViT2DSliceAggregateClassifier with {aggregation} aggregation")

    # Load checkpoint
    print(f"\nLoading checkpoint from {args_cmd.checkpoint}...")
    checkpoint = torch.load(args_cmd.checkpoint, map_location='cpu')
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    # Handle DDP wrapper if needed
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict)
    model = model.to(device)
    model.eval()

    print(f"Checkpoint loaded (epoch {checkpoint.get('epoch', 'unknown')})")

    # Build dataset
    print(f"\nLoading {args_cmd.split} dataset...")
    transform = get_opencell_2d_val_transforms()

    csv_path = os.path.join(args.csv_path, f'{args_cmd.split}.csv')
    dataset = OpenCellLocalizationSliceDataset(
        csv_path=csv_path,
        localization_csv_path=args.localization_csv_path,
        split=args_cmd.split,
        transform=transform,
        slice_start=slice_start,
        slice_end=slice_end,
        grade_weights=getattr(args, 'grade_weights', None)
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args_cmd.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_slices
    )

    print(f"Dataset size: {len(dataset)} samples")
    print(f"Slices per sample: {slice_end - slice_start + 1}")

    # Evaluate
    logits, targets = evaluate_model(model, dataloader, device)

    # Compute metrics
    print("\nComputing metrics...")
    metrics = compute_metrics(logits, targets, threshold=args_cmd.threshold)

    # Print results
    print("\n" + "="*80)
    print(f"RESULTS ON {args_cmd.split.upper()} SET (Slice Aggregation: {aggregation})")
    print("="*80)
    print(f"mAP:        {metrics['mAP']:.4f}")
    print(f"Macro AUC:  {metrics['macro_AUC']:.4f}")
    print(f"Micro AUC:  {metrics['micro_AUC']:.4f}")
    print(f"Macro F1:   {metrics['macro_F1']:.4f}")
    print(f"Micro F1:   {metrics['micro_F1']:.4f}")

    print("\nPer-class Average Precision (sorted by AP):")
    print("-"*80)
    sorted_classes = sorted(metrics['per_class'].items(),
                           key=lambda x: x[1]['AP'], reverse=True)
    for label, class_metrics in sorted_classes:
        print(f"{label:20s}  AP: {class_metrics['AP']:.4f}  "
              f"AUC: {class_metrics['AUC']:.4f}  "
              f"F1: {class_metrics['F1']:.4f}  "
              f"Support: {class_metrics['support']:4d}")
    print("="*80)

    # Save results to JSON
    results = {
        'config': args_cmd.config,
        'checkpoint': args_cmd.checkpoint,
        'split': args_cmd.split,
        'threshold': args_cmd.threshold,
        'num_samples': len(dataset),
        'slice_start': slice_start,
        'slice_end': slice_end,
        'aggregation': aggregation,
        'metrics': metrics
    }

    # Remove .json or .csv extension if provided
    output_prefix = args_cmd.output.replace('.json', '').replace('.csv', '')

    json_path = f"{output_prefix}.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nJSON results saved to: {json_path}")

    # Save per-class results to CSV
    csv_path_out = f"{output_prefix}_per_class.csv"
    csv_data = []
    for label, class_metrics in sorted(metrics['per_class'].items(),
                                       key=lambda x: x[1]['AP'], reverse=True):
        csv_data.append({
            'Localization': label,
            'AP': f"{class_metrics['AP']:.4f}",
            'AUC': f"{class_metrics['AUC']:.4f}",
            'F1': f"{class_metrics['F1']:.4f}",
            'Precision': f"{class_metrics['precision']:.4f}",
            'Recall': f"{class_metrics['recall']:.4f}",
            'Support': class_metrics['support']
        })

    df = pd.DataFrame(csv_data)
    df.to_csv(csv_path_out, index=False)
    print(f"Per-class CSV saved to: {csv_path_out}")

    # Save summary statistics to CSV
    summary_csv_path = f"{output_prefix}_summary.csv"
    summary_data = [
        {'Metric': 'mAP', 'Value': f"{metrics['mAP']:.4f}"},
        {'Metric': 'Macro AUC', 'Value': f"{metrics['macro_AUC']:.4f}"},
        {'Metric': 'Micro AUC', 'Value': f"{metrics['micro_AUC']:.4f}"},
        {'Metric': 'Macro F1', 'Value': f"{metrics['macro_F1']:.4f}"},
        {'Metric': 'Micro F1', 'Value': f"{metrics['micro_F1']:.4f}"},
        {'Metric': 'Num Samples', 'Value': str(len(dataset))},
        {'Metric': 'Threshold', 'Value': f"{args_cmd.threshold:.2f}"},
        {'Metric': 'Slice Range', 'Value': f"{slice_start}-{slice_end}"},
        {'Metric': 'Aggregation', 'Value': aggregation},
    ]
    df_summary = pd.DataFrame(summary_data)
    df_summary.to_csv(summary_csv_path, index=False)
    print(f"Summary CSV saved to: {summary_csv_path}")

    print(f"\nEvaluation complete! Results saved to:")
    print(f"  - {json_path} (complete data)")
    print(f"  - {csv_path_out} (per-class metrics)")
    print(f"  - {summary_csv_path} (summary metrics)")


if __name__ == '__main__':
    main()
