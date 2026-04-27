# VIPlanner Architecture Documentation

## Overview

This document describes the modular architecture of the refactored VIPlanner training framework.

## Design Principles

1. **Separation of Concerns**: Each module has a single, well-defined responsibility
2. **Modularity**: Components can be swapped or extended independently
3. **Reusability**: Code can be reused across different experiments
4. **Testability**: Individual components can be tested in isolation
5. **Maintainability**: Clear structure makes the codebase easier to understand and modify

## Module Structure

### 1. Models (`viplanner/models/`)

**Purpose**: Define neural network architectures

**Files**:
- `plannernet.py`: Base encoder network (ResNet-like architecture)
- `autoencoder.py`: Encoder-decoder models (single and dual stream)

**Key Classes**:
- `PlannerNet`: Convolutional encoder
- `AutoEncoder`: Single-stream depth encoder-decoder
- `DualAutoEncoder`: Dual-stream (depth + semantic/RGB) encoder-decoder
- `Decoder`: Standard decoder
- `DecoderSmall`: Lightweight decoder variant

**Usage**:
```python
from viplanner.models import DualAutoEncoder

model = DualAutoEncoder(
    in_channel=16,
    knodes=5,
    decoder_small=False,
)
```

### 2. Losses (`viplanner/losses/`)

**Purpose**: Define loss functions for training

**Files**:
- `trajectory_loss.py`: Trajectory-based loss computation

**Key Classes**:
- `TrajectoryLoss`: Combines obstacle, height, motion, and goal costs

**Usage**:
```python
from viplanner.losses import TrajectoryLoss

loss_fn = TrajectoryLoss(
    w_obs=0.25,
    w_height=1.0,
    w_motion=1.5,
    w_goal=4.0,
)
```

### 3. Trainers (`viplanner/trainers/`)

**Purpose**: Implement training loops and model management

**Files**:
- `viplanner_trainer.py`: Main training orchestration

**Key Classes**:
- `ViPlannerTrainer`: Handles training, validation, checkpointing, and logging

**Responsibilities**:
- Model initialization and loading
- Optimizer and scheduler setup
- Training and validation loops
- Checkpoint management
- Logging integration

**Usage**:
```python
from viplanner.trainers import ViPlannerTrainer

trainer = ViPlannerTrainer(config)
trainer.setup()
trainer.train(train_loader, val_loader)
trainer.test(test_loader)
trainer.save_config()
```

### 4. Datasets (`viplanner/datasets/`)

**Purpose**: Handle data loading, preprocessing, and augmentation

**Files**:
- `planner_dataset.py`: PyTorch Dataset implementation
- `preprocessing.py`: Image augmentation utilities

**Key Classes**:
- `PlannerData`: PyTorch Dataset for training data
- `PlannerDataGenerator`: Data generation and train/val splitting
- `ImageAugmentation`: Image preprocessing and augmentation

**Features**:
- Lazy loading or RAM caching
- Depth and semantic/RGB image loading
- Data augmentation (noise, polygons, flipping)
- Train/validation splitting

**Usage**:
```python
from viplanner.datasets import PlannerData

dataset = PlannerData(
    transform=transforms,
    semantics=True,
    max_depth=15.0,
)
```

### 5. Configs (`viplanner/configs/`)

**Purpose**: Define configuration dataclasses

**Files**:
- `train_config.py`: Training and data configuration

**Key Classes**:
- `TrainConfig`: Model, training, and system settings
- `DataConfig`: Data loading and preprocessing settings

**Benefits**:
- Type-safe configuration
- Default values
- Easy serialization to YAML
- IDE autocomplete support

**Usage**:
```python
from viplanner.configs import TrainConfig, DataConfig

cfg = TrainConfig(
    epochs=100,
    batch_size=64,
    lr=2e-3,
    data_cfg=DataConfig(max_depth=15.0),
)
```

### 6. Utils (`viplanner/utils/`)

**Purpose**: Provide utility functions and helpers

**Files**:
- `optimizer.py`: Optimizer utilities (early stopping, parameter counting)
- `logging.py`: Logging wrappers (Weights & Biases)

**Key Classes**:
- `EarlyStopScheduler`: Learning rate scheduling with early stopping
- `WandbLogger`: Weights & Biases logging wrapper

## Data Flow

```
1. Configuration
   TrainConfig + DataConfig
   ↓
2. Data Loading
   PlannerDataGenerator → PlannerData → DataLoader
   ↓
3. Model Setup
   DualAutoEncoder (or AutoEncoder)
   ↓
4. Training Loop
   ViPlannerTrainer.train()
   ├─ Forward pass
   ├─ Loss computation (TrajectoryLoss)
   ├─ Backward pass
   ├─ Optimizer step
   └─ Logging (WandbLogger)
   ↓
5. Validation
   ViPlannerTrainer.validate_epoch()
   ├─ Forward pass
   ├─ Loss computation
   └─ Checkpoint saving
   ↓
6. Testing
   ViPlannerTrainer.test()
```

## Extension Points

### Adding a New Model

1. Create new file in `viplanner/models/`
2. Inherit from `nn.Module`
3. Export in `viplanner/models/__init__.py`
4. Use in trainer by modifying `_setup_model()`

### Adding a New Loss

1. Create new file in `viplanner/losses/`
2. Inherit from `nn.Module`
3. Implement `forward()` method
4. Export in `viplanner/losses/__init__.py`
5. Use in trainer by modifying `_setup_loss()`

### Adding a New Trainer

1. Create new file in `viplanner/trainers/`
2. Implement training logic
3. Export in `viplanner/trainers/__init__.py`

### Adding Data Augmentation

1. Add methods to `ImageAugmentation` class
2. Configure in `DataConfig`
3. Apply in `PlannerData._load_depth_img()` or `_load_sem_rgb_img()`

## Comparison with Original Implementation

| Aspect | Original | Refactored |
|--------|----------|------------|
| File count | ~5 large files | ~15 focused files |
| Lines per file | 500-1300 | 100-300 |
| Coupling | High (monolithic) | Low (modular) |
| Testability | Difficult | Easy |
| Extensibility | Requires modification | Plug-and-play |
| Reusability | Limited | High |

## Best Practices

1. **Keep modules focused**: Each file should have one clear purpose
2. **Use type hints**: Helps with IDE support and documentation
3. **Document public APIs**: Add docstrings to classes and methods
4. **Separate configuration from code**: Use config classes instead of hardcoded values
5. **Make components independent**: Minimize dependencies between modules
6. **Test in isolation**: Write unit tests for individual components

## Future Improvements

1. Complete data loading pipeline implementation
2. Add trajectory cost optimization integration
3. Implement visualization utilities
4. Add comprehensive unit tests
5. Create example notebooks
6. Add multi-GPU training support
7. Implement distributed training
8. Add model export utilities (ONNX, TorchScript)
