"""
Evaluate WTC-11 protein localization classifier.

Usage
-----
    python src/evaluate_localization_wtc.py \
        --config     configs/wtc/wtc_localization_emb_3d_fft_kfold.yaml \
        --checkpoint /path/to/checkpoint.pth.tar \
        --mae_embedding_path /path/to/fold0/mae3d_embeddings \
        --csv_path   /path/to/kfold5/fold0 \
        --output     /path/to/results/val_results \
        --split val
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score, f1_score,
    average_precision_score, roc_auc_score,
    classification_report,
)
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from utils.utils import get_conf
from lib.models import ViT3DCrossAttentionClassifier, ViT2DClassifier, ViT3DClassifier
from data.wtc.localization_dataset import (
    WTCLocalizationDataset,
    WTC_LOCALIZATION_LABELS,
)


# ── Evaluation helpers ────────────────────────────────────────────────────────

def evaluate_model(model, dataloader, device):
    model.eval()
    all_logits, all_targets = [], []

    with torch.no_grad():
        for data in tqdm(dataloader, desc='Evaluating'):
            targets = data['label'].to(device)
            with torch.cuda.amp.autocast(True):
                if 'mae_embedding' in data:
                    mae_emb = data['mae_embedding'].to(device)
                    underlying = model.module if hasattr(model, 'module') else model
                    logits = underlying.forward_from_embeddings(mae_emb)
                else:
                    images = data['image'].to(device)
                    logits = model(images)
            all_logits.append(logits.cpu())
            all_targets.append(targets.cpu())

    logits  = torch.cat(all_logits,  dim=0).numpy()
    targets = torch.cat(all_targets, dim=0).numpy()
    return logits, targets


def compute_metrics(logits, targets, threshold=0.5):
    probs = torch.sigmoid(torch.from_numpy(logits)).numpy()
    binary_targets = (targets > 0).astype(int)
    binary_preds   = (probs > threshold).astype(int)

    # Hard predictions via argmax (single-label classification)
    pred_class   = probs.argmax(axis=1)
    target_class = binary_targets.argmax(axis=1)

    metrics = {
        'accuracy': float(accuracy_score(target_class, pred_class)),
        'macro_F1': float(f1_score(target_class, pred_class,
                                   average='macro', zero_division=0)),
        'micro_F1': float(f1_score(target_class, pred_class,
                                   average='micro', zero_division=0)),
    }

    # Multi-label metrics for reference
    try:
        metrics['mAP']       = float(average_precision_score(binary_targets, probs, average='macro'))
        metrics['macro_AUC'] = float(roc_auc_score(binary_targets, probs, average='macro'))
    except Exception:
        metrics['mAP'] = metrics['macro_AUC'] = 0.0

    # Per-class
    per_class = {}
    for i, label in enumerate(WTC_LOCALIZATION_LABELS):
        n_pos = int(binary_targets[:, i].sum())
        cm = {}
        if n_pos > 0:
            try:
                cm['AP']  = float(average_precision_score(binary_targets[:, i], probs[:, i]))
                cm['AUC'] = float(roc_auc_score(binary_targets[:, i], probs[:, i]))
            except Exception:
                cm['AP'] = cm['AUC'] = 0.0
            cm['F1']      = float(f1_score(binary_targets[:, i], binary_preds[:, i],
                                           zero_division=0))
        else:
            cm = {'AP': 0.0, 'AUC': 0.0, 'F1': 0.0}
        cm['support'] = n_pos
        per_class[label] = cm

    metrics['per_class'] = per_class
    return metrics


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Evaluate WTC-11 Localization Model')
    parser.add_argument('--config',             type=str, required=True)
    parser.add_argument('--checkpoint',         type=str, required=True)
    parser.add_argument('--output',             type=str, default='results')
    parser.add_argument('--split',              type=str, default='val',
                        choices=['train', 'val'])
    parser.add_argument('--batch_size',         type=int, default=128)
    parser.add_argument('--threshold',          type=float, default=0.5)
    parser.add_argument('--mae_embedding_path', type=str, default=None)
    parser.add_argument('--csv_path',           type=str, default=None)
    args_cmd = parser.parse_args()

    args = get_conf(args_cmd.config)
    if args_cmd.csv_path:
        args.csv_path = args_cmd.csv_path

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # ── Build model ───────────────────────────────────────────────────────────
    arch = getattr(args, 'arch', 'ViT3DCrossAttentionClassifier')
    use_2d = getattr(args, 'use_2d', False)

    model_params = dict(
        input_size=tuple(args.input_size),
        patch_size=tuple(args.patch_size),
        in_chans=args.in_chans,
        num_classes=args.num_classes,
        embed_dim=args.encoder_embed_dim,
        depth=args.encoder_depth,
        num_heads=args.encoder_num_heads,
        drop_path_rate=getattr(args, 'drop_path', 0.0),
        pos_embed_type=getattr(args, 'pos_embed_type', 'sincos'),
        use_global_pool=getattr(args, 'use_global_pool', True),
    )

    if arch == 'ViT3DCrossAttentionClassifier':
        model_params['cross_attention_type'] = getattr(args, 'cross_attention_type', 'position_wise')
        model_params['pool_mode'] = getattr(args, 'pool_mode', 'concat')
        model = ViT3DCrossAttentionClassifier(**model_params)
    elif use_2d:
        model_params['input_size'] = model_params['input_size'][:2]
        model_params['patch_size'] = model_params['patch_size'][:2]
        model = ViT2DClassifier(**model_params)
    else:
        model = ViT3DClassifier(**model_params)

    ckpt = torch.load(args_cmd.checkpoint, map_location='cpu')
    model.load_state_dict(ckpt.get('state_dict', ckpt))
    model = model.to(device)
    model.eval()
    print(f'Loaded checkpoint (epoch {ckpt.get("epoch", "?")})')

    # ── Dataset ───────────────────────────────────────────────────────────────
    use_embedding_mode = args_cmd.mae_embedding_path is not None
    mae_emb_path = None
    if use_embedding_mode:
        mae_emb_path = os.path.join(args_cmd.mae_embedding_path,
                                    f'{args_cmd.split}.npy')
        if not os.path.exists(mae_emb_path):
            raise FileNotFoundError(f'MAE embedding not found: {mae_emb_path}')

    dataset = WTCLocalizationDataset(
        csv_path=os.path.join(args.csv_path, f'{args_cmd.split}.csv'),
        split=args_cmd.split,
        use_max_projection=use_2d,
        mae_embedding_path=mae_emb_path,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args_cmd.batch_size,
        shuffle=False, num_workers=4, pin_memory=True,
    )
    print(f'Dataset: {len(dataset)} samples')

    # ── Evaluate ──────────────────────────────────────────────────────────────
    logits, targets = evaluate_model(model, dataloader, device)
    metrics = compute_metrics(logits, targets, threshold=args_cmd.threshold)

    print('\n' + '=' * 70)
    print(f'RESULTS  [{args_cmd.split.upper()}]')
    print('=' * 70)
    print(f'Accuracy:  {metrics["accuracy"]:.4f}')
    print(f'Macro F1:  {metrics["macro_F1"]:.4f}')
    print(f'Micro F1:  {metrics["micro_F1"]:.4f}')
    print(f'mAP:       {metrics["mAP"]:.4f}')
    print(f'Macro AUC: {metrics["macro_AUC"]:.4f}')

    print('\nPer-class AP (sorted):')
    for label, cm in sorted(metrics['per_class'].items(),
                             key=lambda x: -x[1]['AP']):
        print(f'  {label:20s}  AP:{cm["AP"]:.3f}  F1:{cm["F1"]:.3f}  '
              f'support:{cm["support"]}')
    print('=' * 70)

    # ── Save ──────────────────────────────────────────────────────────────────
    prefix = args_cmd.output.rstrip('.json').rstrip('.csv')
    os.makedirs(os.path.dirname(os.path.abspath(prefix)), exist_ok=True)

    results = {
        'config': args_cmd.config,
        'checkpoint': args_cmd.checkpoint,
        'split': args_cmd.split,
        'num_samples': len(dataset),
        'metrics': metrics,
    }
    with open(f'{prefix}.json', 'w') as f:
        json.dump(results, f, indent=2)

    rows = [{'Localization': lb,
             'AP':  f"{cm['AP']:.4f}",
             'F1':  f"{cm['F1']:.4f}",
             'AUC': f"{cm['AUC']:.4f}",
             'Support': cm['support']}
            for lb, cm in sorted(metrics['per_class'].items(),
                                  key=lambda x: -x[1]['AP'])]
    pd.DataFrame(rows).to_csv(f'{prefix}_per_class.csv', index=False)

    summary = [
        {'Metric': 'Accuracy',  'Value': f"{metrics['accuracy']:.4f}"},
        {'Metric': 'Macro F1',  'Value': f"{metrics['macro_F1']:.4f}"},
        {'Metric': 'Micro F1',  'Value': f"{metrics['micro_F1']:.4f}"},
        {'Metric': 'mAP',       'Value': f"{metrics['mAP']:.4f}"},
        {'Metric': 'Macro AUC', 'Value': f"{metrics['macro_AUC']:.4f}"},
    ]
    pd.DataFrame(summary).to_csv(f'{prefix}_summary.csv', index=False)

    print(f'\nSaved: {prefix}.json, {prefix}_per_class.csv, {prefix}_summary.csv')


if __name__ == '__main__':
    main()
