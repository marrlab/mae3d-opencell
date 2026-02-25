#!/usr/bin/env python3
"""
Evaluation script for OpenCell protein localization classification.
Evaluates a trained model on the test set and saves detailed results.

Usage:
    python src/evaluate_localization.py \
        --config configs/opencell_localization_3d.yaml \
        --checkpoint /path/to/checkpoint.pth.tar \
        --output results_test.json
"""

import os
import sys
import argparse
import json
import csv
from pathlib import Path
import torch
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score, classification_report
from tqdm import tqdm
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from utils.utils import get_conf
from lib.models import ViT3DClassifier, ViT2DClassifier, ViT3DCrossAttentionClassifier, ViT3DCrossAttentionSubCellClassifier
from data.opencell.localization_dataset import OpenCellLocalizationDataset, LOCALIZATION_LABELS
from data.opencell.localization_fusion_dataset import OpenCellLocalizationFusionDataset
from data.opencell.transforms import (
    get_opencell_val_transforms,
    get_opencell_2d_val_transforms
)


def evaluate_model(model, dataloader, device, use_embedding_mode=False, is_fusion=False):
    """Evaluate model on a dataset."""
    model.eval()

    all_logits = []
    all_targets = []

    print(f"Running evaluation (mode: {'embedding' if use_embedding_mode else 'image'}, fusion: {is_fusion})...")
    with torch.no_grad():
        for data in tqdm(dataloader, desc="Evaluating"):
            targets = data['label'].to(device)
            subcell_emb = data['subcell_embedding'].to(device) if is_fusion else None

            with torch.cuda.amp.autocast(True):
                if use_embedding_mode and 'mae_embedding' in data:
                    # Embedding mode (fast) - skip encoder
                    mae_emb = data['mae_embedding'].to(device)
                    underlying_model = model.module if hasattr(model, 'module') else model
                    if is_fusion:
                        logits = underlying_model.forward_from_embeddings(mae_emb, subcell_emb)
                    else:
                        logits = underlying_model.forward_from_embeddings(mae_emb)
                else:
                    # Image mode (full forward)
                    images = data['image'].to(device)
                    if is_fusion:
                        logits = model(images, subcell_emb)
                    else:
                        logits = model(images)

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
    parser = argparse.ArgumentParser(description='Evaluate OpenCell Localization Model')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint file')
    parser.add_argument('--output', type=str, default='test_results', help='Output file prefix (will create .json and .csv)')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'],
                       help='Which split to evaluate on')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for evaluation')
    parser.add_argument('--threshold', type=float, default=0.5, help='Classification threshold')
    parser.add_argument('--mae_embedding_path', type=str, default=None,
                       help='Path to precomputed MAE embeddings directory (for fast evaluation)')
    parser.add_argument('--mae_embedding_csv_path', type=str, default=None,
                       help='Directory of CSVs used during embedding extraction (e.g. dataset1/). '
                            'When set, builds a combined lookup from all splits so kfold '
                            'evaluation CSVs can reference cells from any source split.')
    parser.add_argument('--csv_path', type=str, default=None,
                       help='Override config csv_path (e.g. kfold5/fold2)')
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
        'num_classes': args.num_classes,
        'embed_dim': args.encoder_embed_dim,
        'depth': args.encoder_depth,
        'num_heads': args.encoder_num_heads,
        'drop_path_rate': getattr(args, 'drop_path', 0.0),
        'pos_embed_type': getattr(args, 'pos_embed_type', 'sincos'),
        'use_global_pool': getattr(args, 'use_global_pool', True),
    }

    is_fusion = arch == 'ViT3DCrossAttentionSubCellClassifier'

    if arch == 'ViT3DCrossAttentionSubCellClassifier':
        # Cross-attention classifier with SubCell fusion
        model_params['cross_attention_type'] = getattr(args, 'cross_attention_type', 'position_wise')
        model_params['pool_mode'] = getattr(args, 'pool_mode', 'concat')
        model_params['subcell_embed_dim'] = getattr(args, 'subcell_embed_dim', 1536)
        model_params['subcell_proj_dim'] = getattr(args, 'subcell_proj_dim', None)
        model_params['fusion_type'] = getattr(args, 'fusion_type', 'concat')
        model = ViT3DCrossAttentionSubCellClassifier(**model_params)
        print(f"  Using ViT3DCrossAttentionSubCellClassifier")
        print(f"  Cross-attention type: {model_params['cross_attention_type']}")
        print(f"  Pool mode: {model_params['pool_mode']}")
        print(f"  Fusion type: {model_params['fusion_type']}")
    elif arch == 'ViT3DCrossAttentionClassifier':
        # Cross-attention classifier
        model_params['cross_attention_type'] = getattr(args, 'cross_attention_type', 'position_wise')
        model_params['pool_mode'] = getattr(args, 'pool_mode', 'concat')
        model = ViT3DCrossAttentionClassifier(**model_params)
        print(f"  Using ViT3DCrossAttentionClassifier")
        print(f"  Cross-attention type: {model_params['cross_attention_type']}")
        print(f"  Pool mode: {model_params['pool_mode']}")
    elif use_2d:
        model_params['input_size'] = model_params['input_size'][:2] if len(model_params['input_size']) == 3 else model_params['input_size']
        model_params['patch_size'] = model_params['patch_size'][:2] if len(model_params['patch_size']) == 3 else model_params['patch_size']
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

    # Override csv_path from CLI if provided
    if args_cmd.csv_path is not None:
        args.csv_path = args_cmd.csv_path

    # Check for embedding mode
    use_embedding_mode = args_cmd.mae_embedding_path is not None
    mae_emb_path = None
    combined_emb = None
    combined_lookup = None

    if use_embedding_mode:
        if args_cmd.mae_embedding_csv_path:
            # Kfold mode: a fold's eval CSV may reference cells from any source split.
            # Build a combined embedding array + lookup covering train+val+test so every
            # cell is found regardless of which dataset1 split it came from.
            print(f"Using embedding mode with combined kfold lookup...")
            print(f"  MAE embedding dir:  {args_cmd.mae_embedding_path}")
            print(f"  Source CSV dir:     {args_cmd.mae_embedding_csv_path}")
            all_embs = []
            combined_lookup = {}
            offset = 0
            for sname in ('train', 'val', 'test'):
                npy = os.path.join(args_cmd.mae_embedding_path, f'{sname}.npy')
                csv_s = os.path.join(args_cmd.mae_embedding_csv_path, f'{sname}.csv')
                if os.path.exists(npy) and os.path.exists(csv_s):
                    arr = np.load(npy)
                    sdf = pd.read_csv(csv_s)
                    assert len(arr) == len(sdf), (
                        f"{sname}: npy rows ({len(arr)}) != csv rows ({len(sdf)})"
                    )
                    for local_i, (_, row) in enumerate(sdf.iterrows()):
                        combined_lookup[row['image_path']] = offset + local_i
                    all_embs.append(arr)
                    offset += len(arr)
                    print(f"  {sname}: {len(arr)} embeddings")
            if all_embs:
                combined_emb = np.concatenate(all_embs, axis=0)
                print(f"  Combined: {combined_emb.shape}, {len(combined_lookup)} unique paths")
        else:
            # Legacy / non-kfold: single split .npy aligned positionally with csv_path
            mae_emb_path = os.path.join(args_cmd.mae_embedding_path, f'{args_cmd.split}.npy')
            if not os.path.exists(mae_emb_path):
                raise FileNotFoundError(f"MAE embedding file not found: {mae_emb_path}")
            print(f"Using embedding mode: {mae_emb_path}")

    # Build dataset
    print(f"Loading {args_cmd.split} dataset...")
    if use_embedding_mode:
        transform = None  # No transforms needed for embeddings
    elif use_2d:
        transform = get_opencell_2d_val_transforms()
    else:
        transform = get_opencell_val_transforms()

    csv_path = os.path.join(args.csv_path, f'{args_cmd.split}.csv')
    if is_fusion:
        # Fusion dataset needs SubCell embeddings
        embedding_path = os.path.join(args.embedding_path, f'{args_cmd.split}.npy')
        dataset = OpenCellLocalizationFusionDataset(
            csv_path=csv_path,
            localization_csv_path=args.localization_csv_path,
            embedding_path=embedding_path,
            split=args_cmd.split,
            transform=transform,
            cache_rate=0.0,
            use_max_projection=use_2d,
            grade_weights=getattr(args, 'grade_weights', None),
            z_slice_start=getattr(args, 'z_slice_start', None),
            z_slice_end=getattr(args, 'z_slice_end', None),
            mae_embedding_path=mae_emb_path,
        )
    else:
        dataset = OpenCellLocalizationDataset(
            csv_path=csv_path,
            localization_csv_path=args.localization_csv_path,
            split=args_cmd.split,
            transform=transform,
            cache_rate=0.0,
            use_max_projection=use_2d,
            grade_weights=getattr(args, 'grade_weights', None),
            z_slice_start=getattr(args, 'z_slice_start', None),
            z_slice_end=getattr(args, 'z_slice_end', None),
            mae_embedding_path=mae_emb_path,
            mae_embedding_array=combined_emb,
            mae_embedding_lookup_dict=combined_lookup,
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
    logits, targets = evaluate_model(model, dataloader, device, use_embedding_mode=use_embedding_mode, is_fusion=is_fusion)

    # Compute metrics
    print("\nComputing metrics...")
    metrics = compute_metrics(logits, targets, threshold=args_cmd.threshold)

    # Print results
    print("\n" + "="*80)
    print(f"RESULTS ON {args_cmd.split.upper()} SET")
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
        'metrics': metrics
    }

    # Remove .json or .csv extension if provided, we'll add them
    output_prefix = args_cmd.output.replace('.json', '').replace('.csv', '')

    json_path = f"{output_prefix}.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nJSON results saved to: {json_path}")

    # Save per-class results to CSV for easy viewing
    csv_path = f"{output_prefix}_per_class.csv"
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
    df.to_csv(csv_path, index=False)
    print(f"Per-class CSV saved to: {csv_path}")

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
    ]
    df_summary = pd.DataFrame(summary_data)
    df_summary.to_csv(summary_csv_path, index=False)
    print(f"Summary CSV saved to: {summary_csv_path}")

    print(f"\n✓ Evaluation complete! Results saved to:")
    print(f"  - {json_path} (complete data)")
    print(f"  - {csv_path} (per-class metrics)")
    print(f"  - {summary_csv_path} (summary metrics)")


if __name__ == '__main__':
    main()
