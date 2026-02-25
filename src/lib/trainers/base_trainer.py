import os
import math
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from pathlib import Path
from abc import ABC, abstractmethod


class BaseTrainer(ABC):
    """
    Base class for all trainers.
    Handles common training infrastructure:
    - Distributed training setup
    - Model wrapping (DDP)
    - Optimizer creation
    - Checkpoint management
    - Learning rate scheduling
    """

    def __init__(self, args):
        self.args = args
        self.model = None
        self.wrapped_model = None
        self.optimizer = None
        self.train_loader = None
        self.val_loader = None

        # Distributed training attributes
        self.rank = args.rank if hasattr(args, 'rank') else 0
        self.world_size = args.world_size if hasattr(args, 'world_size') else 1
        self.local_rank = args.gpu if hasattr(args, 'gpu') else 0
        self.is_distributed = self.world_size > 1

        # Learning rate (can be scaled by world_size in subclasses)
        self.lr = args.lr

        # Create checkpoint directory
        if self.rank == 0:
            Path(args.ckpt_dir).mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def build_model(self):
        """Build the model architecture. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement build_model()")

    @abstractmethod
    def build_optimizer(self):
        """Build the optimizer. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement build_optimizer()")

    @abstractmethod
    def build_dataloader(self):
        """Build train/val dataloaders. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement build_dataloader()")

    @abstractmethod
    def epoch_train(self, epoch):
        """Training logic for one epoch. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement epoch_train()")

    def wrap_model(self):
        """
        Wrap model with DistributedDataParallel if using distributed training.
        """
        assert self.model is not None, "Please build model before wrapping"

        if self.is_distributed:
            # Apply SyncBatchNorm for multi-GPU training
            self.model = nn.SyncBatchNorm.convert_sync_batchnorm(self.model)

            # Set device
            torch.cuda.set_device(self.local_rank)
            self.model = self.model.cuda(self.local_rank)

            # Wrap with DDP
            self.wrapped_model = DDP(
                self.model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=True
            )

            if self.rank == 0:
                print(f"Model wrapped with DDP on {self.world_size} GPUs")
        else:
            # Single GPU
            if self.local_rank is not None:
                torch.cuda.set_device(self.local_rank)
                self.model = self.model.cuda(self.local_rank)
            else:
                self.model = self.model.cuda()

            self.wrapped_model = self.model

            if self.rank == 0:
                print("Single GPU training")

    def group_params(self, model):
        """
        Separate parameters into weight decay and no weight decay groups.
        Weight decay is applied to Conv and Linear layer weights only.
        """
        all_params = set(model.parameters())
        wd_params = set()

        for m in model.modules():
            if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                if m.weight is not None:
                    wd_params.add(m.weight)

        no_wd = all_params - wd_params

        params_group = [
            {'params': list(wd_params)},
            {'params': list(no_wd), 'weight_decay': 0.}
        ]

        return params_group

    def adjust_learning_rate(self, epoch):
        """
        Cosine learning rate schedule with warmup.
        Note: Warmup is handled per-step in the training loop (epoch_train).
        This function only applies cosine decay after warmup epochs.
        """
        args = self.args
        init_lr = self.lr

        if epoch < args.warmup_epochs:
            # During warmup epochs, LR is adjusted per-step in epoch_train()
            # Return starting LR for this epoch for logging purposes
            cur_lr = init_lr * 0.01  # Starting LR for warmup (actual LR increases per step)
        else:
            # Cosine decay after warmup
            cur_lr = init_lr * 0.5 * (1. + math.cos(
                math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)
            ))

            # Apply the cosine LR for post-warmup epochs
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = cur_lr

        return cur_lr

    def save_checkpoint(self, epoch, **kwargs):
        """
        Save checkpoint to disk.

        Args:
            epoch: Current epoch number
            **kwargs: Additional items to save in checkpoint
        """
        if self.rank != 0:
            return  # Only rank 0 saves checkpoints

        # Get unwrapped model state dict
        model_state = self.model.state_dict() if not self.is_distributed else self.wrapped_model.module.state_dict()

        checkpoint = {
            'epoch': epoch + 1,
            'arch': self.args.arch if hasattr(self.args, 'arch') else 'unknown',
            'state_dict': model_state,
            'optimizer': self.optimizer.state_dict(),
        }

        # Add any additional kwargs
        checkpoint.update(kwargs)

        checkpoint_path = Path(self.args.ckpt_dir) / f"checkpoint_{epoch:04d}.pth.tar"
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")

    def resume(self):
        """
        Resume training from checkpoint.
        """
        args = self.args

        if args.resume is None or not os.path.isfile(args.resume):
            if args.resume is not None:
                print(f"=> no checkpoint found at '{args.resume}'")
            return

        print(f"=> loading checkpoint '{args.resume}'")

        # Load checkpoint
        if self.local_rank is not None:
            loc = f'cuda:{self.local_rank}'
            checkpoint = torch.load(args.resume, map_location=loc)
        else:
            checkpoint = torch.load(args.resume)

        # Update start epoch
        args.start_epoch = checkpoint['epoch']

        # Load model state
        state_dict = checkpoint['state_dict']
        if self.is_distributed:
            self.wrapped_model.module.load_state_dict(state_dict)
        else:
            self.model.load_state_dict(state_dict)

        # Load optimizer state
        self.optimizer.load_state_dict(checkpoint['optimizer'])

        print(f"=> loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})")

        return checkpoint

    def run(self):
        """
        Main training loop.
        """
        args = self.args

        for epoch in range(args.start_epoch, args.epochs):
            # Set epoch for distributed sampler
            if self.is_distributed and hasattr(self.train_loader, 'sampler'):
                if hasattr(self.train_loader.sampler, 'set_epoch'):
                    self.train_loader.sampler.set_epoch(epoch)

            # Adjust learning rate
            current_lr = self.adjust_learning_rate(epoch)

            if self.rank == 0:
                print(f"\n{'='*60}")
                print(f"Epoch {epoch}/{args.epochs} | LR: {current_lr:.6f}")
                print(f"{'='*60}")

            # Train for one epoch
            self.epoch_train(epoch)

            # Save checkpoint
            if (epoch + 1) % args.save_freq == 0:
                self.save_checkpoint(epoch)

        if self.rank == 0:
            print("\nTraining completed!")
