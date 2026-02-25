# Mask Ratio Experiments

This directory contains configuration files for running MAE experiments with different mask ratios (70%, 75%, 80%, 85%, 90%).

## Directory Structure

After training, each experiment will have its own output directory:

```
mae_opencell_3d/
├── mae3d_vit_base_opencell_mask0.7/
│   ├── ckpts/              # Model checkpoints
│   ├── wandb/              # WandB logs (saved locally)
│   ├── config.yaml         # Saved configuration
│   └── training_*.log      # Training logs (timestamped)
├── mae3d_vit_base_opencell_mask0.75/
│   └── ...
├── mae3d_vit_base_opencell_mask0.8/
│   └── ...
└── ...
```

## Available Configurations

### 3D MAE Experiments
- `opencell_3d_mask0.70.yaml` - 70% masking (30% visible)
- `opencell_3d_mask0.75.yaml` - 75% masking (25% visible) - Standard MAE
- `opencell_3d_mask0.80.yaml` - 80% masking (20% visible)
- `opencell_3d_mask0.85.yaml` - 85% masking (15% visible)
- `opencell_3d_mask0.90.yaml` - 90% masking (10% visible)

### 2D MAE Experiments
- `opencell_2d_mask0.70.yaml` - 70% masking (30% visible)
- `opencell_2d_mask0.75.yaml` - 75% masking (25% visible)
- `opencell_2d_mask0.80.yaml` - 80% masking (20% visible)
- `opencell_2d_mask0.85.yaml` - 85% masking (15% visible)
- `opencell_2d_mask0.90.yaml` - 90% masking (10% visible)

## Running Experiments

### Single GPU Training

#### 3D MAE
```bash
# 70% mask ratio
python src/train_with_trainer.py --config configs/opencell/mask_ratio_experiments/opencell_3d_mask0.70.yaml

# 75% mask ratio
python src/train_with_trainer.py --config configs/opencell/mask_ratio_experiments/opencell_3d_mask0.75.yaml

# 80% mask ratio
python src/train_with_trainer.py --config configs/opencell/mask_ratio_experiments/opencell_3d_mask0.80.yaml

# 85% mask ratio
python src/train_with_trainer.py --config configs/opencell/mask_ratio_experiments/opencell_3d_mask0.85.yaml

# 90% mask ratio
python src/train_with_trainer.py --config configs/opencell/mask_ratio_experiments/opencell_3d_mask0.90.yaml
```

#### 2D MAE
```bash
# 70% mask ratio
python src/train_with_trainer.py --config configs/opencell/mask_ratio_experiments/opencell_2d_mask0.70.yaml

# 75% mask ratio
python src/train_with_trainer.py --config configs/opencell/mask_ratio_experiments/opencell_2d_mask0.75.yaml

# ... and so on
```

### Multi-GPU Training (torchrun)

```bash
# 3D MAE with 4 GPUs
torchrun --nproc_per_node=4 src/train_with_trainer.py --config configs/opencell/mask_ratio_experiments/opencell_3d_mask0.75.yaml

# 2D MAE with 4 GPUs
torchrun --nproc_per_node=4 src/train_with_trainer.py --config configs/opencell/mask_ratio_experiments/opencell_2d_mask0.75.yaml
```

## Running All Experiments in Batch

You can run all experiments sequentially:

```bash
#!/bin/bash
# run_mask_ratio_experiments.sh

# 3D experiments
for mask in 0.70 0.75 0.80 0.85 0.90; do
    echo "Running 3D MAE with mask_ratio=${mask}"
    python src/train_with_trainer.py \
        --config configs/opencell/mask_ratio_experiments/opencell_3d_mask${mask}.yaml
done

# 2D experiments
for mask in 0.70 0.75 0.80 0.85 0.90; do
    echo "Running 2D MAE with mask_ratio=${mask}"
    python src/train_with_trainer.py \
        --config configs/opencell/mask_ratio_experiments/opencell_2d_mask${mask}.yaml
done
```

Or run them in parallel on different GPUs:

```bash
#!/bin/bash
# run_parallel_experiments.sh

# Run 5 experiments in parallel on 5 GPUs
CUDA_VISIBLE_DEVICES=0 python src/train_with_trainer.py --config configs/opencell/mask_ratio_experiments/opencell_3d_mask0.70.yaml &
CUDA_VISIBLE_DEVICES=1 python src/train_with_trainer.py --config configs/opencell/mask_ratio_experiments/opencell_3d_mask0.75.yaml &
CUDA_VISIBLE_DEVICES=2 python src/train_with_trainer.py --config configs/opencell/mask_ratio_experiments/opencell_3d_mask0.80.yaml &
CUDA_VISIBLE_DEVICES=3 python src/train_with_trainer.py --config configs/opencell/mask_ratio_experiments/opencell_3d_mask0.85.yaml &
CUDA_VISIBLE_DEVICES=4 python src/train_with_trainer.py --config configs/opencell/mask_ratio_experiments/opencell_3d_mask0.90.yaml &

wait
echo "All experiments completed!"
```

## Output Files

For each experiment, you'll find:

### Checkpoints (`ckpts/`)
- `checkpoint_0000.pth.tar` - Checkpoint after epoch 0
- `checkpoint_0001.pth.tar` - Checkpoint after epoch 1
- ... and so on (saved every `save_freq` epochs)

### Training Logs (`training_*.log`)
- Timestamped log file with all console output
- Includes training progress, loss values, learning rates, etc.
- Example: `training_output_20260111_152030.log`

### Configuration (`config.yaml`)
- Complete configuration used for this experiment
- Useful for reproducing results or comparing settings

### WandB Logs (`wandb/`)
- Complete WandB run data saved locally
- Includes metrics, visualizations, code snapshots
- Can be viewed offline or synced to WandB cloud later

## Monitoring Training

### Real-time Monitoring
```bash
# Watch training log in real-time
tail -f /path/to/output_dir/training_output_*.log

# Or use grep to filter specific information
tail -f /path/to/output_dir/training_output_*.log | grep "Loss:"
```

### WandB Dashboard
```bash
# View WandB logs locally (if wandb server is set up)
cd /path/to/output_dir/wandb
wandb local

# Or sync to WandB cloud
wandb sync ./run-*
```

## Comparing Results

After training all experiments, you can compare:

1. **Reconstruction Quality**: Check visualizations in WandB
2. **Training Loss**: Compare loss curves across different mask ratios
3. **Convergence Speed**: See which mask ratio converges faster
4. **Final Performance**: Evaluate on downstream tasks (localization)

## Expected Results

According to MAE paper:
- **Higher mask ratios (80-90%)**: More challenging task, better representations for downstream tasks
- **Lower mask ratios (70-75%)**: Easier reconstruction, potentially faster convergence
- **Optimal range**: Original MAE found 75% to be optimal for ViT models

## Downstream Evaluation

After pretraining with different mask ratios, evaluate on localization task:

```bash
# Update the pretrain path in localization config
# configs/opencell/opencell_localization_3d.yaml

# Then run localization fine-tuning
python src/train_localization.py --config configs/opencell/opencell_localization_3d.yaml
```

Update the `pretrain_mask_ratio` field in localization configs to track which pretrained model was used.

## Tips

1. **Start with 75%**: This is the standard MAE setting and good baseline
2. **Monitor GPU memory**: Higher mask ratios use less memory (fewer tokens to process)
3. **Check reconstructions**: Visualizations in WandB show if model is learning
4. **Save checkpoints**: Keep checkpoints for downstream evaluation
5. **Compare systematically**: Use same hyperparameters except mask_ratio

## Troubleshooting

**Problem**: Training crashes with OOM (Out of Memory)
- **Solution**: Reduce batch_size or use gradient accumulation

**Problem**: Logs not saving
- **Solution**: Check that output_dir exists and has write permissions

**Problem**: WandB not initializing
- **Solution**: Run `wandb login` or set `WANDB_MODE=offline`

**Problem**: Config file not found
- **Solution**: Use absolute paths or run from project root

## Notes

- Each experiment is completely independent (separate directories)
- No risk of overwriting previous experiments
- All experiments use identical settings except `mask_ratio`
- Run names include mask ratio: `mae3d_vit_base_opencell_mask0.7`
