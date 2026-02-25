# Trainer Pattern Implementation

This directory now includes two different approaches for training MAE3D models on OpenCell:

## 📁 File Overview

### **Direct Script Approach** (Original)
- **File**: `train_mae3d_opencell.py`
- **Style**: Functional/procedural
- **All logic in one file**: ~330 lines

### **Trainer Pattern Approach** (New)
- **File**: `train_with_trainer.py` (main script)
- **Trainers**: `lib/trainers/base_trainer.py`, `lib/trainers/mae3d_trainer.py`
- **Style**: Object-oriented with inheritance
- **Modular structure**: Separated concerns

---

## 🔄 Two Approaches Comparison

### Direct Script Approach (`train_mae3d_opencell.py`)

**Advantages:**
- ✅ **Simple and straightforward** - all code in one place
- ✅ **Easy to understand** - linear flow from top to bottom
- ✅ **Quick to modify** - no need to navigate class hierarchies
- ✅ **Good for prototyping** - fast iteration

**Disadvantages:**
- ❌ **Code duplication** if you add more training tasks
- ❌ **Hard to extend** to other tasks (fine-tuning, segmentation, etc.)
- ❌ **Less standardized** - each script could be different

**When to use:**
- Single task experiments
- Quick prototyping
- Learning/exploration
- You prefer simplicity over extensibility

---

### Trainer Pattern Approach (`train_with_trainer.py`)

**Advantages:**
- ✅ **Code reuse** - BaseTrainer handles 80% of boilerplate
- ✅ **Easy to extend** - add new trainers by inheriting from BaseTrainer
- ✅ **Standardized interface** - all trainers work the same way
- ✅ **Advanced features** - easier to add layer-wise LR decay, parameter grouping
- ✅ **Matches reference implementation** - same pattern as `temp/lib/trainers/`

**Disadvantages:**
- ❌ **More complex** - need to understand inheritance
- ❌ **Indirection** - logic spread across multiple files
- ❌ **Slightly more boilerplate** for simple tasks

**When to use:**
- Multiple training tasks (pretraining + fine-tuning)
- Production/library code
- Need to reuse infrastructure
- Team collaboration (standardized patterns)
- Following best practices from `temp/` reference implementation

---

## 🏗️ Trainer Pattern Architecture

```
BaseTrainer (abstract)
├── Handles common infrastructure
├── Distributed training setup
├── Model wrapping (DDP)
├── Checkpoint management
├── Learning rate scheduling
└── Abstract methods:
    ├── build_model()
    ├── build_optimizer()
    ├── build_dataloader()
    └── epoch_train()

MAE3DTrainer (concrete)
├── Inherits from BaseTrainer
├── Implements OpenCell-specific logic
├── MAE3D model creation
├── AdamW optimizer setup
├── OpenCell dataset loading
├── Mixed precision training
└── Reconstruction visualization
```

---

## 🚀 Usage Examples

### Direct Script

```bash
# Single GPU
python src/train_mae3d_opencell.py

# Multi-GPU with torchrun
torchrun --nproc_per_node=4 src/train_mae3d_opencell.py
```

### Trainer Pattern

```bash
# Single GPU
python src/train_with_trainer.py

# Multi-GPU with torchrun
torchrun --nproc_per_node=4 src/train_with_trainer.py
```

**Both approaches:**
- Read the same config: `configs/opencell_3d.yaml`
- Support distributed training via `torchrun`
- Log to Weights & Biases
- Save checkpoints to the same location
- **Produce identical training results**

---

## 🔧 Extending the Trainer Pattern

### Adding a New Trainer (e.g., ViT Fine-tuning)

1. **Create new trainer class** in `lib/trainers/vit_trainer.py`:

```python
from .base_trainer import BaseTrainer

class VitTrainer(BaseTrainer):
    def __init__(self, args):
        super().__init__(args)
        self.model_name = 'ViT'

    def build_model(self):
        # Your model creation logic
        pass

    def build_optimizer(self):
        # Your optimizer setup
        pass

    def build_dataloader(self):
        # Your dataloader creation
        pass

    def epoch_train(self, epoch):
        # Your training loop
        pass
```

2. **Register in `__init__.py`**:

```python
from .vit_trainer import VitTrainer
__all__ = ['BaseTrainer', 'MAE3DTrainer', 'VitTrainer']
```

3. **Create config** `configs/vit_finetune.yaml`:

```yaml
trainer_name: VitTrainer
arch: vit_base
# ... other settings
```

4. **Run**:

```bash
python src/train_with_trainer.py --config configs/vit_finetune.yaml
```

---

## 📊 Feature Comparison Matrix

| Feature | Direct Script | Trainer Pattern |
|---------|--------------|----------------|
| Lines of code (main) | ~330 | ~110 |
| Setup complexity | Low | Medium |
| Extensibility | Low | High |
| Code reuse | None | High |
| Add new task | Copy/modify entire script | Inherit + implement 4 methods |
| Matches `temp/` reference | ❌ | ✅ |
| Learning curve | Easy | Moderate |
| Production ready | Good for single task | Excellent for multiple tasks |

---

## 🎯 Recommendation

**Start with the Direct Script** (`train_mae3d_opencell.py`) if:
- You're just doing MAE3D pretraining on OpenCell
- You prefer simplicity and transparency
- You're still experimenting and iterating

**Switch to Trainer Pattern** (`train_with_trainer.py`) when:
- You need to add fine-tuning tasks (ViT, segmentation, etc.)
- You're building a library or production system
- You want to match the reference implementation in `temp/`
- You're working in a team and need standardization

---

## 📝 Implementation Notes

### BaseTrainer Features
- Distributed training (single/multi-GPU)
- DDP model wrapping with SyncBatchNorm
- Parameter grouping (weight decay for conv/linear only)
- Cosine learning rate schedule with warmup
- Checkpoint save/resume with all state
- Abstract interface for extensibility

### MAE3DTrainer Features
- MAE3D model with configurable encoder/decoder
- AdamW optimizer with scaled learning rate
- OpenCell dataset with caching support
- Mixed precision training (AMP + GradScaler)
- Beautiful reconstruction visualizations (6×3 grid)
- Step-based visualization logging

---

## 🔗 Related Files

- **Reference Implementation**: `temp/lib/trainers/` (SelfMedMAE)
- **Models**: `src/lib/models/mae3d.py`
- **Networks**: `src/lib/networks/mae_vit.py`
- **Datasets**: `src/data/opencell_dataset.py`
- **Configs**: `configs/opencell_3d.yaml`

---

## 💡 Tips

1. **Both scripts are maintained** - use whichever fits your workflow
2. **Same config file** - `configs/opencell_3d.yaml` works for both
3. **Identical results** - both produce the same training outcomes
4. **Easy migration** - you can switch between approaches at any time
5. **Checkpoints compatible** - saved checkpoints work with both scripts

---

## 🤝 Contributing

When adding new trainers:
1. Inherit from `BaseTrainer`
2. Implement the 4 abstract methods
3. Add tests to verify it works
4. Document in this README
5. Follow the pattern from `MAE3DTrainer`
