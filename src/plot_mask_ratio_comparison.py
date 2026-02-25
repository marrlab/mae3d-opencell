"""
Plot localization metrics comparison across mask ratios for 2D and 3D models.

This script creates plots comparing 2D and 3D model performance
across different mask ratios (0.7, 0.75, 0.8, 0.85, 0.9).

Usage:
    python src/plot_mask_ratio_comparison.py --output_dir /path/to/save/plots
"""

import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def load_results(base_dir, model_type, mask_ratios):
    """
    Load test results for all mask ratios.

    Args:
        base_dir: Base directory containing results
        model_type: '2d' or '3d'
        mask_ratios: List of mask ratios to load

    Returns:
        Dictionary with mask_ratio -> metrics dict
    """
    results = {}

    for mask_ratio in mask_ratios:
        # Construct path based on directory structure
        dir_name = f"opencell_localization_{model_type}_vit_base_pretrain_mask{mask_ratio}"
        csv_path = os.path.join(base_dir, dir_name, dir_name, "test_results_summary.csv")

        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            # Convert to dict: Metric -> Value
            metrics = dict(zip(df['Metric'], df['Value']))
            results[mask_ratio] = metrics
            print(f"Loaded {model_type} mask {mask_ratio}: {csv_path}")
        else:
            print(f"Warning: File not found: {csv_path}")

    return results


def plot_metric_comparison(results_2d, results_3d, metric_name, mask_ratios, output_path):
    """
    Plot a single metric comparing 2D and 3D across mask ratios.

    Args:
        results_2d: Dict of 2D results
        results_3d: Dict of 3D results
        metric_name: Name of metric to plot
        mask_ratios: List of mask ratios
        output_path: Path to save the plot
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    # Extract values for each mask ratio
    values_2d = []
    values_3d = []
    valid_ratios = []

    for ratio in mask_ratios:
        if ratio in results_2d and ratio in results_3d:
            if metric_name in results_2d[ratio] and metric_name in results_3d[ratio]:
                values_2d.append(results_2d[ratio][metric_name])
                values_3d.append(results_3d[ratio][metric_name])
                valid_ratios.append(ratio)

    if not valid_ratios:
        print(f"Warning: No data for metric {metric_name}")
        return

    # Plot lines
    ax.plot(valid_ratios, values_2d, 'o-', color='#2196F3', linewidth=2,
            markersize=8, label='MAE2D', markeredgecolor='white', markeredgewidth=1)
    ax.plot(valid_ratios, values_3d, 's-', color='#FF5722', linewidth=2,
            markersize=8, label='MAE3D', markeredgecolor='white', markeredgewidth=1)

    # Add value annotations
    for i, ratio in enumerate(valid_ratios):
        ax.annotate(f'{values_2d[i]:.3f}', (ratio, values_2d[i]),
                   textcoords="offset points", xytext=(0, 10), ha='center', fontsize=9)
        ax.annotate(f'{values_3d[i]:.3f}', (ratio, values_3d[i]),
                   textcoords="offset points", xytext=(0, -15), ha='center', fontsize=9)

    ax.set_xlabel('Mask Ratio', fontsize=12)
    ax.set_ylabel(metric_name, fontsize=12)
    ax.set_title(f'{metric_name} vs Mask Ratio', fontsize=14)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(valid_ratios)

    # Set y-axis limits with some padding
    all_values = values_2d + values_3d
    y_min = min(all_values) - 0.05 * (max(all_values) - min(all_values) + 0.01)
    y_max = max(all_values) + 0.05 * (max(all_values) - min(all_values) + 0.01)
    ax.set_ylim(y_min, y_max)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_all_metrics_combined(results_2d, results_3d, metrics, mask_ratios, output_path):
    """
    Create a combined figure with all metrics as subplots.
    """
    n_metrics = len(metrics)
    n_cols = 3
    n_rows = (n_metrics + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    axes = axes.flatten() if n_metrics > 1 else [axes]

    for idx, metric_name in enumerate(metrics):
        ax = axes[idx]

        values_2d = []
        values_3d = []
        valid_ratios = []

        for ratio in mask_ratios:
            if ratio in results_2d and ratio in results_3d:
                if metric_name in results_2d[ratio] and metric_name in results_3d[ratio]:
                    values_2d.append(results_2d[ratio][metric_name])
                    values_3d.append(results_3d[ratio][metric_name])
                    valid_ratios.append(ratio)

        if not valid_ratios:
            ax.set_visible(False)
            continue

        ax.plot(valid_ratios, values_2d, 'o-', color='#2196F3', linewidth=2,
                markersize=8, label='MAE2D', markeredgecolor='white', markeredgewidth=1)
        ax.plot(valid_ratios, values_3d, 's-', color='#FF5722', linewidth=2,
                markersize=8, label='MAE3D', markeredgecolor='white', markeredgewidth=1)

        ax.set_xlabel('Mask Ratio', fontsize=11)
        ax.set_ylabel(metric_name, fontsize=11)
        ax.set_title(f'{metric_name}', fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(valid_ratios)

        # Set y-axis limits
        all_values = values_2d + values_3d
        y_min = min(all_values) - 0.05 * (max(all_values) - min(all_values) + 0.01)
        y_max = max(all_values) + 0.05 * (max(all_values) - min(all_values) + 0.01)
        ax.set_ylim(y_min, y_max)

    # Hide unused subplots
    for idx in range(len(metrics), len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle('MAE2D vs MAE3D: Localization Performance by Mask Ratio',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved combined plot: {output_path}")


def create_summary_table(results_2d, results_3d, metrics, mask_ratios, output_path):
    """Create a summary CSV table with all results."""
    rows = []

    for ratio in mask_ratios:
        for model_type, results in [('2D', results_2d), ('3D', results_3d)]:
            if ratio in results:
                row = {'Mask Ratio': ratio, 'Model': model_type}
                for metric in metrics:
                    row[metric] = results[ratio].get(metric, None)
                rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"Saved summary table: {output_path}")
    return df


def main():
    parser = argparse.ArgumentParser(description='Plot mask ratio comparison for 2D vs 3D')
    parser.add_argument('--results_dir', type=str,
                        default='/path/to/datasets/opencell/localization_results',
                        help='Base directory containing localization results')
    parser.add_argument('--output_dir', type=str,
                        default='/path/to/datasets/opencell/localization_results/comparison_plots',
                        help='Directory to save plots')
    parser.add_argument('--mask_ratios', type=float, nargs='+',
                        default=[0.7, 0.75, 0.8, 0.85, 0.9],
                        help='Mask ratios to compare')
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load results for both 2D and 3D
    print("Loading 2D results...")
    results_2d = load_results(args.results_dir, '2d', args.mask_ratios)

    print("\nLoading 3D results...")
    results_3d = load_results(args.results_dir, '3d', args.mask_ratios)

    # Define metrics to plot
    metrics = ['mAP', 'Macro AUC', 'Micro AUC', 'Macro F1', 'Micro F1']

    # Create individual plots for each metric
    print("\nCreating individual metric plots...")
    for metric in metrics:
        output_path = os.path.join(args.output_dir, f'{metric.replace(" ", "_").lower()}_comparison.png')
        plot_metric_comparison(results_2d, results_3d, metric, args.mask_ratios, output_path)

    # Create combined plot with all metrics
    print("\nCreating combined plot...")
    combined_path = os.path.join(args.output_dir, 'all_metrics_comparison.png')
    plot_all_metrics_combined(results_2d, results_3d, metrics, args.mask_ratios, combined_path)

    # Create summary table
    print("\nCreating summary table...")
    table_path = os.path.join(args.output_dir, 'metrics_summary.csv')
    df = create_summary_table(results_2d, results_3d, metrics, args.mask_ratios, table_path)

    # Print summary
    print("\n" + "="*60)
    print("Summary Table:")
    print("="*60)
    print(df.to_string(index=False))
    print("="*60)

    print(f"\nAll plots saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
