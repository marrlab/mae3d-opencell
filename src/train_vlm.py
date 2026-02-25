"""
Train Vision-Language Model (VLM) on OpenCell Visual QA dataset.

Two-stage training:
1. Stage 1: Train projection layer only (freeze vision encoder + LLM)
2. Stage 2: Fine-tune projection + LLM with LoRA

Usage:
    # Stage 1: Projection training
    python src/train_vlm.py \
        --config configs/opencell/vlm.yaml \
        --stage 1

    # Stage 2: Fine-tuning with LoRA
    python src/train_vlm.py \
        --config configs/opencell/vlm.yaml \
        --stage 2 \
        --resume /path/to/stage1/checkpoint
"""

import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
from tqdm import tqdm
import wandb

sys.path.insert(0, str(Path(__file__).parent))

from omegaconf import OmegaConf
from transformers import get_cosine_schedule_with_warmup

from lib.models.vlm import CellVLM, VLMConfig, build_vlm
from data.opencell.vlm_dataset import OpenCellVLMDataset, OpenCellVLMCollator


# ============================================================================
# Dataset
# ============================================================================

class VLMTrainDataset(OpenCellVLMDataset):
    """
    VLM training dataset that returns tokenized inputs ready for training.
    """

    def __init__(
        self,
        jsonl_path: str,
        tokenizer,
        image_transform=None,
        max_length: int = 512,
        use_3d: bool = True,
        use_max_projection: bool = True,
        image_token: str = "<image>",
        **kwargs
    ):
        super().__init__(
            jsonl_path=jsonl_path,
            image_transform=image_transform,
            tokenizer=None,  # We handle tokenization ourselves
            use_3d=use_3d,
            use_max_projection=use_max_projection,
            return_raw_text=True,
            **kwargs
        )
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.image_token = image_token

    def __getitem__(self, idx):
        sample = super().__getitem__(idx)

        image = sample['image']
        question = sample['question']
        answer = sample['answer']

        # Build prompt with image token
        prompt = f"{self.image_token}\nQuestion: {question}\nAnswer: {answer}"

        # Tokenize
        encoded = self.tokenizer(
            prompt,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )

        input_ids = encoded['input_ids'].squeeze(0)
        attention_mask = encoded['attention_mask'].squeeze(0)

        # Create labels (shift by 1 for causal LM)
        labels = input_ids.clone()

        # Mask prompt tokens (only compute loss on answer)
        # Find "Answer:" position and mask everything before it
        answer_start = prompt.find("Answer:")
        if answer_start > 0:
            answer_tokens_start = len(self.tokenizer.encode(prompt[:answer_start + 7], add_special_tokens=False))
            labels[:answer_tokens_start] = -100

        # Mask padding
        labels[attention_mask == 0] = -100

        return {
            'image': image,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'metadata': sample['metadata'],
        }


class VLMCollator:
    """Collator for VLM training batches."""

    def __call__(self, batch):
        images = torch.stack([b['image'] for b in batch])
        input_ids = torch.stack([b['input_ids'] for b in batch])
        attention_mask = torch.stack([b['attention_mask'] for b in batch])
        labels = torch.stack([b['labels'] for b in batch])

        metadata = {
            'gene_name': [b['metadata']['gene_name'] for b in batch],
            'qa_type': [b['metadata']['qa_type'] for b in batch],
        }

        return {
            'image': images,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'metadata': metadata,
        }


# ============================================================================
# Training
# ============================================================================

def train_epoch(
    model: CellVLM,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    config,
    device: torch.device,
    save_callback=None,  # Callback function for mid-epoch saving
):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_batches = 0

    # Mid-epoch save interval (in batches)
    save_every_n_batches = config.get('save_every_n_batches', 500)
    total_batches = len(dataloader)

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")

    for batch_idx, batch in enumerate(pbar):
        images = batch['image'].to(device)
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()

        # Use autocast with bfloat16 (no GradScaler needed for bfloat16)
        with autocast(dtype=torch.bfloat16):
            outputs = model(
                images=images,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs['loss']

        # Backward
        loss.backward()

        # Gradient clipping
        if config.get('grad_clip', 0) > 0:
            torch.nn.utils.clip_grad_norm_(
                model.get_trainable_parameters(),
                config.grad_clip
            )

        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        num_batches += 1

        # Update progress bar
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'avg_loss': f"{total_loss / num_batches:.4f}",
            'lr': f"{scheduler.get_last_lr()[0]:.2e}",
            'batch': f"{batch_idx+1}/{total_batches}",
        })

        # Log to wandb
        if batch_idx % config.get('log_interval', 50) == 0 and wandb.run:
            wandb.log({
                'train/loss': loss.item(),
                'train/lr': scheduler.get_last_lr()[0],
                'train/epoch': epoch,
                'train/step': epoch * len(dataloader) + batch_idx,
            })

        # Mid-epoch checkpoint saving
        if save_callback and (batch_idx + 1) % save_every_n_batches == 0:
            progress = (batch_idx + 1) / total_batches
            print(f"\n  Saving mid-epoch checkpoint at batch {batch_idx + 1}/{total_batches} ({progress*100:.1f}%)")
            save_callback(epoch, batch_idx + 1, total_loss / num_batches)

    return total_loss / num_batches


@torch.no_grad()
def validate(
    model: CellVLM,
    dataloader: DataLoader,
    device: torch.device,
):
    """Validate model."""
    model.eval()
    total_loss = 0
    num_batches = 0

    for batch in tqdm(dataloader, desc="Validating"):
        images = batch['image'].to(device)
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        with autocast(dtype=torch.bfloat16):
            outputs = model(
                images=images,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

        total_loss += outputs['loss'].item()
        num_batches += 1

    return total_loss / num_batches


@torch.no_grad()
def generate_samples(
    model: CellVLM,
    dataloader: DataLoader,
    device: torch.device,
    num_samples: int = 5,
):
    """Generate sample answers for qualitative evaluation."""
    model.eval()

    samples = []
    batch = next(iter(dataloader))

    images = batch['image'][:num_samples].to(device)
    metadata = batch['metadata']

    # Get questions from tokenized inputs
    questions = []
    for i in range(num_samples):
        input_ids = batch['input_ids'][i]
        text = model.tokenizer.decode(input_ids, skip_special_tokens=False)

        # Extract question
        if "Question:" in text and "Answer:" in text:
            q_start = text.find("Question:") + 9
            q_end = text.find("Answer:")
            questions.append(text[q_start:q_end].strip())
        else:
            questions.append("What protein is shown?")

    # Generate answers
    generated = model.generate(
        images=images,
        questions=questions,
        max_new_tokens=128,
        temperature=0.7,
    )

    for i in range(num_samples):
        samples.append({
            'gene': metadata['gene_name'][i],
            'qa_type': metadata['qa_type'][i],
            'question': questions[i],
            'generated': generated[i],
        })

    return samples


def save_checkpoint(
    model: CellVLM,
    optimizer,
    scheduler,
    epoch: int,
    loss: float,
    save_path: str,
):
    """Save training checkpoint."""
    os.makedirs(save_path, exist_ok=True)

    # Save projection weights
    torch.save({
        'epoch': epoch,
        'loss': loss,
        'projection_state_dict': model.projection.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }, os.path.join(save_path, f'checkpoint_epoch{epoch}.pt'))

    # Save LoRA weights if applicable
    if model.config.use_lora:
        model.llm.save_pretrained(os.path.join(save_path, f'llm_lora_epoch{epoch}'))

    # Save latest
    model.save_pretrained(os.path.join(save_path, 'latest'))

    print(f"Saved checkpoint to {save_path}")


def load_checkpoint(
    model: CellVLM,
    optimizer,
    scheduler,
    checkpoint_path: str,
    load_optimizer: bool = True,
):
    """Load training checkpoint.

    Args:
        model: The VLM model
        optimizer: The optimizer
        scheduler: The learning rate scheduler
        checkpoint_path: Path to checkpoint file
        load_optimizer: Whether to load optimizer/scheduler state. Set to False
                       when transitioning between stages (e.g., Stage 1 -> Stage 2)
                       since the trainable parameters change.
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    model.projection.load_state_dict(checkpoint['projection_state_dict'])

    if load_optimizer:
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        except ValueError as e:
            print(f"Warning: Could not load optimizer/scheduler state: {e}")
            print("This is expected when transitioning between training stages.")
            print("Optimizer will start fresh with current learning rate.")

    return checkpoint['epoch'], checkpoint['loss']


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train VLM on OpenCell')

    # Config
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config YAML')
    parser.add_argument('--stage', type=int, default=1, choices=[1, 2],
                        help='Training stage (1: projection only, 2: fine-tune with LoRA)')

    # Data
    parser.add_argument('--train_jsonl', type=str, default=None,
                        help='Path to training JSONL (overrides config)')
    parser.add_argument('--val_jsonl', type=str, default=None,
                        help='Path to validation JSONL (overrides config)')

    # Training
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint')

    # Output
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--run_name', type=str, default=None)

    # Other
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--wandb', action='store_true', help='Enable wandb logging')

    args = parser.parse_args()

    # Load config
    config = OmegaConf.load(args.config)
    OmegaConf.resolve(config)

    # Override with command line args
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.epochs:
        config.epochs = args.epochs
    if args.lr:
        config.lr = args.lr
    if args.train_jsonl:
        config.train_jsonl = args.train_jsonl
    if args.val_jsonl:
        config.val_jsonl = args.val_jsonl
    if args.output_dir:
        config.output_dir = args.output_dir

    # Set seed
    torch.manual_seed(args.seed)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Stage-specific settings
    if args.stage == 1:
        freeze_vision = True
        freeze_llm = True
        use_lora = False
        stage_name = "stage1_projection"
    else:
        freeze_vision = True
        freeze_llm = False
        use_lora = True
        stage_name = "stage2_lora"

    # Run name
    run_name = args.run_name or f"vlm_{stage_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Output directory
    output_dir = os.path.join(config.get('output_dir', 'outputs/vlm'), run_name)
    os.makedirs(output_dir, exist_ok=True)

    # Initialize wandb
    if args.wandb:
        wandb.init(
            project=config.get('wandb_project', 'cell-vlm'),
            name=run_name,
            config=OmegaConf.to_container(config, resolve=True),
        )

    # Load MAE config for vision encoder
    mae_config = OmegaConf.load(config.mae_config)
    OmegaConf.resolve(mae_config)

    print("\n" + "="*60)
    print(f"Building VLM (Stage {args.stage})")
    print("="*60)

    # Determine model type (2D or 3D)
    model_type = config.get('model_type', '3d')
    use_3d = config.get('use_3d', True)

    # Build model
    model = build_vlm(
        mae_config=mae_config,
        mae_checkpoint=config.mae_checkpoint,
        llm_name=config.get('llm_name', 'mistralai/Mistral-7B-v0.1'),
        freeze_vision=freeze_vision,
        freeze_llm=freeze_llm,
        use_lora=use_lora,
        num_vision_tokens=config.get('num_vision_tokens', 64),
        device_map='auto',
        model_type=model_type,
    )

    # Move projection and vision encoder to device with correct dtype
    model.projection = model.projection.to(device).to(torch.bfloat16)
    model.vision_encoder = model.vision_encoder.to(device).to(torch.bfloat16)
    # For nn.Parameter, modify data in place
    model.encoder_pos_embed.data = model.encoder_pos_embed.data.to(device).to(torch.bfloat16)

    print(f"\nTrainable parameters:")
    total_params = sum(p.numel() for p in model.get_trainable_parameters())
    print(f"  Total: {total_params:,}")

    # Load image transforms (2D or 3D)
    if use_3d:
        from data.opencell.vlm_dataset import get_vlm_train_transforms, get_vlm_val_transforms
        train_transform = get_vlm_train_transforms()
        val_transform = get_vlm_val_transforms()
    else:
        from data.opencell.transforms import get_opencell_2d_train_transforms, get_opencell_2d_val_transforms
        train_transform = get_opencell_2d_train_transforms()
        val_transform = get_opencell_2d_val_transforms()

    print("\n" + "="*60)
    print("Loading datasets")
    print("="*60)

    # Create datasets
    use_max_projection = config.get('use_max_projection', True)

    train_dataset = VLMTrainDataset(
        jsonl_path=config.train_jsonl,
        tokenizer=model.tokenizer,
        image_transform=train_transform,
        max_length=config.get('max_length', 512),
        use_3d=use_3d,
        use_max_projection=use_max_projection,
        image_token=model.image_token,
        qa_sampling='random',
        expanded_format=config.get('expanded_format', True),
    )

    val_dataset = VLMTrainDataset(
        jsonl_path=config.val_jsonl,
        tokenizer=model.tokenizer,
        image_transform=val_transform,
        max_length=config.get('max_length', 512),
        use_3d=use_3d,
        use_max_projection=use_max_projection,
        image_token=model.image_token,
        qa_sampling='first',
        expanded_format=config.get('expanded_format', True),
    )

    print(f"Train dataset: {len(train_dataset)} samples")
    print(f"Val dataset: {len(val_dataset)} samples")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=VLMCollator(),
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=VLMCollator(),
        pin_memory=True,
    )

    # Optimizer
    trainable_params = model.get_trainable_parameters()
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=config.lr,
        weight_decay=config.get('weight_decay', 0.01),
    )

    # Scheduler
    num_training_steps = len(train_loader) * config.epochs
    num_warmup_steps = int(config.get('warmup_ratio', 0.1) * num_training_steps)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    # Resume if specified
    start_epoch = 0
    if args.resume:
        print(f"Resuming from {args.resume}")
        # Don't load optimizer state when transitioning from Stage 1 to Stage 2
        # (different trainable parameters, so optimizer state is incompatible)
        load_optimizer = (args.stage == 1)  # Only load optimizer for same-stage resume
        start_epoch, _ = load_checkpoint(model, optimizer, scheduler, args.resume, load_optimizer=load_optimizer)
        if load_optimizer:
            start_epoch += 1
        else:
            # Starting fresh for Stage 2, so reset epoch counter
            start_epoch = 0
            print("Stage transition detected - starting epoch count from 0")

    print("\n" + "="*60)
    print("Starting training")
    print("="*60)
    print(f"  Stage: {args.stage}")
    print(f"  Epochs: {config.epochs}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Learning rate: {config.lr}")
    print(f"  Output: {output_dir}")

    best_val_loss = float('inf')

    # Create mid-epoch save callback
    def mid_epoch_save_callback(epoch, batch_idx, current_loss):
        save_checkpoint(
            model, optimizer, scheduler,
            epoch + 1,  # Use epoch + 1 to be consistent
            current_loss,
            os.path.join(output_dir, f'checkpoint_epoch{epoch+1}_batch{batch_idx}')
        )

    for epoch in range(start_epoch, config.epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{config.epochs}")
        print(f"{'='*60}")

        # Train with mid-epoch saving
        train_loss = train_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            config=config,
            device=device,
            save_callback=mid_epoch_save_callback,
        )

        # Validate
        val_loss = validate(model, val_loader, device)

        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"  Train loss: {train_loss:.4f}")
        print(f"  Val loss: {val_loss:.4f}")

        # Log to wandb
        if wandb.run:
            wandb.log({
                'epoch': epoch + 1,
                'train/epoch_loss': train_loss,
                'val/loss': val_loss,
            })

            # Generate samples
            if (epoch + 1) % config.get('sample_interval', 5) == 0:
                samples = generate_samples(model, val_loader, device, num_samples=3)
                for i, s in enumerate(samples):
                    wandb.log({
                        f'samples/sample_{i}_gene': s['gene'],
                        f'samples/sample_{i}_question': s['question'],
                        f'samples/sample_{i}_generated': s['generated'],
                    })

        # Save checkpoint
        if (epoch + 1) % config.get('save_interval', 5) == 0:
            save_checkpoint(model, optimizer, scheduler, epoch + 1, val_loss, output_dir)

        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, scheduler, epoch + 1, val_loss,
                          os.path.join(output_dir, 'best'))
            print(f"  New best model saved (val_loss: {val_loss:.4f})")

    # Final save
    save_checkpoint(model, optimizer, scheduler, config.epochs, val_loss, output_dir)

    print("\n" + "="*60)
    print("Training complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Outputs saved to: {output_dir}")
    print("="*60)

    if wandb.run:
        wandb.finish()


if __name__ == '__main__':
    main()
