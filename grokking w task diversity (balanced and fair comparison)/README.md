# Understanding Grokking: From Memorization to Generalization

This repository provides an implementation to reproduce and investigate the **grokking** phenomenon, where neural networks suddenly generalize long after overfitting the training set. The code trains a simple transformer on modular arithmetic (specifically modular division), replicating the abrupt generalization transition and double-descent loss curves described in [Power et al. (2022)](https://arxiv.org/pdf/2201.02177). It supports the configuration of different parameters to investigate their influence on the timing and emergence of grokking. This provides a reproducible environment for studying the transition from memorization to generalization in neural networks.

## Overview

The model learns to perform modular division: given `a / b mod p`, predict the result. With proper regularization (weight decay), the model first memorizes the training set, then after many more optimization steps, suddenly achieves perfect generalization on the validation set.

**Single-task mode:** Given `a / b mod p`, predict the result.

**Multi-task mode:** Learn multiple operations simultaneously (division, addition, subtraction mod p).

<img src="./grokking_plot.png" alt="Grokking Visualization" style="width: 500px; height: auto;">

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### Basic Training (Division Only)
```bash
python run.py
```

This will train for 250,000 steps with learning rate 1e-3 and weight decay 1e-3, and generate a plot showing the grokking phenomenon.

### Multi-Task Training
```bash
python run.py --multi_task --task_division 0.5 --task_addition 0.5 \
  --lr 1e-3 --weight_decay 1e-3
```

This trains on a 50-50 mix of division and addition tasks.

### Resume After Interruption
```bash
python run.py --resume latest
```

If your training gets interrupted (crash, kernel death, etc.), use this to resume exactly where you left off!

## Key Features

### Automatic Checkpointing

Training progress is automatically saved every 1000 steps (configurable). If your code crashes or kernel dies, simply resume:

```bash
# Training interrupted at step 67,000
python run.py --resume latest
# Continues from step 67,000 with all history preserved
```

**What gets saved:**
- Model weights
- Optimizer state (momentum, learning rate)
- Complete training history (all curves)
- Configuration (hyperparameters, task mix)

**Checkpoints are saved to:**
- `checkpoints/checkpoint_latest.pt` - Most recent checkpoint (updated every 1000 steps)
- `checkpoints/checkpoint_step_10000.pt` - Milestone checkpoints (every 10k steps)
- `checkpoints/checkpoint_final.pt` - Final checkpoint when training completes

### Multi-Task Learning

Train on multiple modular arithmetic operations simultaneously:

```bash
# 50% division + 50% addition
python run.py --multi_task --task_division 0.5 --task_addition 0.5

# 70% division + 30% addition
python run.py --multi_task --task_division 0.7 --task_addition 0.3

# Equal mix of all three operations
python run.py --multi_task \
  --task_division 0.33 --task_addition 0.33 --task_subtraction 0.34
```

**Hypothesis to test:** Does task diversity accelerate grokking?

### Balanced Batch Sampling

When using multi-task mode, batches are precisely balanced. For example, with `--task_division 0.5 --task_addition 0.5` and batch size 512:
- Every batch contains exactly 256 division + 256 addition examples
- Ensures consistent gradient updates across both tasks
- More controlled than random mixing

## Usage

### Basic Usage

```bash
python run.py --lr 1e-3 --weight_decay 1e-3
```

### Custom Configuration

```bash
python run.py \
  --lr 5e-4 \
  --weight_decay 5e-3 \
  --num_steps 500000 \
  --batch_size 256 \
  --d_model 256 \
  --nhead 8
```

### Multi-Task with Custom Checkpoint Directory

```bash
python run.py --multi_task --task_division 0.5 --task_addition 0.5 \
  --lr 1e-3 --weight_decay 1e-3 \
  --checkpoint_dir my_experiment \
  --save_path my_experiment.png
```

### Resume from Specific Checkpoint

```bash
# Resume from latest checkpoint
python run.py --resume latest

# Resume from specific checkpoint file
python run.py --resume checkpoints/checkpoint_step_50000.pt

# Resume from custom directory
python run.py --resume my_experiment/checkpoint_latest.pt
```

## Command Line Arguments

### Core Training Parameters
- `--lr`: Learning rate (default: 1e-3)
- `--weight_decay`: Weight decay for AdamW optimizer (default: 1e-3)
- `--num_steps`: Total training steps (default: 250000)
- `--batch_size`: Batch size (default: 512)
- `--seed`: Random seed for reproducibility (default: 42)

### Multi-Task Parameters (New!)
- `--multi_task`: Enable multi-task training (flag)
- `--task_division`: Proportion of division tasks (default: 0.5)
- `--task_addition`: Proportion of addition tasks (default: 0.5)
- `--task_subtraction`: Proportion of subtraction tasks (default: 0.0)

**Note:** Task proportions are automatically normalized to sum to 1.0.

### Checkpoint Parameters (New!)
- `--checkpoint_dir`: Directory to save checkpoints (default: checkpoints)
- `--checkpoint_interval`: Steps between checkpoint saves (default: 1000)
- `--resume`: Resume from checkpoint: "latest" or path to checkpoint file

### Data Parameters
- `--p`: Prime modulus for modular arithmetic (default: 97)
- `--train_fraction`: Fraction of data for training (default: 0.5)

### Model Parameters
- `--d_model`: Dimension of model embeddings (default: 128)
- `--nhead`: Number of attention heads (default: 4)
- `--num_layers`: Number of transformer layers (default: 2)
- `--dropout`: Dropout rate (default: 0.0)

### Output Parameters
- `--save_path`: Path to save the plot (default: grokking_result.png)
- `--log_interval`: Steps between logging (default: 50)

## Examples

### Compare Single-Task vs Multi-Task

```bash
# Baseline: Division only
python run.py --lr 1e-3 --weight_decay 1e-3 \
  --checkpoint_dir baseline \
  --save_path baseline.png

# Multi-task: Division + Addition
python run.py --multi_task --task_division 0.5 --task_addition 0.5 \
  --lr 1e-3 --weight_decay 1e-3 \
  --checkpoint_dir multitask \
  --save_path multitask.png
```

Compare the "Grokking delay" (gap between train and val reaching 95%) to see which approach groks faster!

### Experiment with Different Task Mixtures

```bash
# 90% division, 10% addition (minimal diversity)
python run.py --multi_task --task_division 0.9 --task_addition 0.1 \
  --save_path mix_90_10.png

# 70% division, 30% addition (medium diversity)
python run.py --multi_task --task_division 0.7 --task_addition 0.3 \
  --save_path mix_70_30.png

# 50% division, 50% addition (high diversity)
python run.py --multi_task --task_division 0.5 --task_addition 0.5 \
  --save_path mix_50_50.png
```

### Experiment with Different Hyperparameters

```bash
# Different learning rates
python run.py --lr 1e-3 --weight_decay 1e-2
python run.py --lr 5e-4 --weight_decay 1e-2
python run.py --lr 1e-4 --weight_decay 1e-2

# Different weight decay values
python run.py --lr 5e-4 --weight_decay 1e-1
python run.py --lr 5e-4 --weight_decay 1e-2
python run.py --lr 5e-4 --weight_decay 1e-3

# Larger model
python run.py --d_model 256 --nhead 8 --num_layers 4
```

### Training with Frequent Checkpoints

If your kernel often dies, save more frequently:

```bash
python run.py --lr 1e-3 --weight_decay 1e-3 \
  --checkpoint_interval 500 \
  --checkpoint_dir frequent_saves
```

### Multiple Random Seeds

For robust comparisons, run with different seeds:

```bash
python run.py --seed 42 --save_path result_seed42.png
python run.py --seed 43 --save_path result_seed43.png
python run.py --seed 44 --save_path result_seed44.png
```

## Checkpoint Management Utilities

### List All Checkpoints

```bash
python checkpoint_utils.py list
```

Shows all checkpoints with details (step, accuracy, grokking status, file size).

### Inspect Specific Checkpoint

```bash
python checkpoint_utils.py inspect checkpoints/checkpoint_latest.pt
```

Shows detailed information about a checkpoint:
- Training progress (current step)
- Performance (train/val accuracy and loss)
- Grokking status (detected or not)
- Configuration (hyperparameters, task mix)

### Compare Multiple Checkpoints

```bash
python checkpoint_utils.py compare \
  baseline/checkpoint_final.pt \
  multitask/checkpoint_final.pt
```

Compare experiments side-by-side to see which approach groks faster.

## Project Structure

```
.
├── README.md                      # This file
├── requirements.txt               # Python dependencies
├── run.py                         # Main training script
├── data.py                        # Dataset generation (single & multi-task)
├── model.py                       # Transformer model definition
├── train.py                       # Training loop with checkpointing
├── utils.py                       # Plotting utilities
├── balanced_batch_sampling.py     # Balanced multi-task batch sampler
├── checkpoint_utils.py            # Checkpoint management utilities
└── verify_balanced_batches.py    # Verification script for balanced batching
```

## Expected Output

The script will:
1. Generate a modular arithmetic dataset (single-task or multi-task)
2. Train a transformer model
3. Save checkpoints automatically during training
4. Detect when grokking occurs (if it does)
5. Save a plot showing:
   - Training and validation accuracy over time (demonstrating grokking)
   - Training and validation loss curves (showing double descent)
   - Configuration details and grokking metrics

The plot will be saved to `grokking_result.png` (or your specified path).

### Example Output

```
Using device: cuda

Generating dataset (p=97, train_fraction=0.5)...
Train size: 4656, Val size: 4656
Vocabulary size: 99

Creating model (d_model=128, nhead=4, num_layers=2)...
Model parameters: 210,867

Optimizer: AdamW (lr=0.001, weight_decay=0.001)

Training for 250000 steps...
Checkpoints will be saved to: checkpoints/
Checkpoint interval: every 1000 steps

Training...
 18%|████      | 45000/250000 [12:34<56:78, train_acc=99.5%, val_acc=45.2%]
 
============================================================
- GROKKING DETECTED! 
Training accuracy reached 95% at step: 35,000
Validation accuracy reached 95% at step: 145,000
Grokking delay: 110,000 steps
============================================================

100%|██████████| 250000/250000 [1:23:45<00:00, ...]

Training complete.
  Grokking occurred at step: 145,000
  Training threshold (95%) reached at: 35,000
  Grokking delay: 110,000 steps

Plot saved to grokking_result.png
```

## Key Findings

### Original Grokking Phenomenon
- **Weight decay is crucial**: Higher weight decay (e.g., 1e-3) promotes grokking
- **Training time**: Grokking typically occurs after 10,000-100,000 steps, well after the model has memorized the training set
- **Generalization**: The model eventually achieves ~100% accuracy on both training and validation sets

### Multi-Task Learning Observations
- **Task diversity effects**: Experiment to see whether mixing tasks accelerates or delays grokking
- **Task difficulty**: Addition is simpler than division; different operations may have different grokking timings
- **Regularization**: Multi-task models may require different hyperparameters (learning rate, weight decay)
- **Balanced batching**: Ensures consistent exposure to all tasks during training

## References

This implementation is based on the paper:
- [Power et al. (2022). "Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets"](https://arxiv.org/pdf/2201.02177)
