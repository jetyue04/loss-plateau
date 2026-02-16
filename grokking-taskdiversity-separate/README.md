# Understanding Grokking: From Memorization to Generalization

This repository provides an implementation to reproduce and investigate the **grokking** phenomenon, where neural networks suddenly generalize long after overfitting the training set. The code trains a simple transformer on modular arithmetic tasks (division, addition, subtraction, and multiplication), replicating the abrupt generalization transition and double-descent loss curves described in [Power et al. (2022)](https://arxiv.org/pdf/2201.02177). It supports multi-task training, per-task grokking detection, and comprehensive visualization of the transition from memorization to generalization.

## Overview

The model learns to perform modular arithmetic operations: given `a ⊕ b mod p`, predict the result, where ⊕ can be division, addition, subtraction, or multiplication. With proper regularization (weight decay), the model first memorizes the training set, then after many more optimization steps, suddenly achieves perfect generalization on the validation set.

The implementation supports:
- **Multi-task training**: Train on multiple arithmetic operations simultaneously
- **Per-task validation**: Track generalization performance separately for each task
- **Grokking detection**: Automatically identify when each task reaches 95% validation accuracy
- **Checkpoint system**: Resume training from saved checkpoints
- **Comprehensive visualization**: Generate plots showing accuracy and loss curves for all tasks

<img src="./grokking_plot.png" alt="Grokking Visualization" style="width: 500px; height: auto;">

## Installation

To install the dependencies, run the following command from the root directory of the project:
```bash
pip install -r requirements.txt
```

## Quick Start

Run with default parameters (single task - division (baseline)):
```bash
python run.py
```

Train on multiple tasks:
```bash
python run.py --tasks div add sub mult
```

This will train for 400,000 steps with learning rate 1e-3 and weight decay 1e-3, and generate a plot showing the grokking phenomenon for each task.

## Usage

### Single Task Training

```bash
python run.py --tasks div --lr 1e-3 --weight_decay 1e-3
```

### Multi-Task Training

```bash
python run.py --tasks div mult --lr 1e-3 --weight_decay 1e-3
```

### Custom Configuration

```bash
python run.py \
  --tasks div add sub mult \
  --lr 5e-4 \
  --weight_decay 5e-3 \
  --num_steps 200000 \
  --batch_size 256 \
  --d_model 256 \
  --nhead 8
```

## Command Line Arguments

### Task Configuration
- `--tasks`: List of tasks to train on (choices: `div`, `add`, `sub`, `mult`)
  - Default: `['div']`
  - Example: `--tasks div mult add`

### Optimizer Parameters (Key Parameters)
- `--lr`: Learning rate (default: 1e-3)
- `--weight_decay`: Weight decay for AdamW optimizer (default: 1e-3)

### Data Parameters
- `--p`: Prime modulus for modular arithmetic (default: 97)
- `--train_fraction`: Fraction of data for training (default: 0.5)
- `--seed`: Random seed for reproducibility (default: 42)

### Model Parameters
- `--d_model`: Dimension of model embeddings (default: 128)
- `--nhead`: Number of attention heads (default: 4)
- `--num_layers`: Number of transformer layers (default: 2)
- `--dropout`: Dropout rate (default: 0.0)

### Training Parameters
- `--batch_size`: Batch size (default: 512)
- `--num_steps`: Total training steps (default: 400000)

### Output Parameters
- `--checkpoint_dir`: Directory to save checkpoints (default: `checkpoints`)
- `--save_path`: Path to save the plot (default: `grokking_result.png`)

## Examples

### Compare Grokking Across Different Tasks

```bash
# Train on all four operations
python run.py --tasks div add sub mult --num_steps 150000

# Compare division vs multiplication
python run.py --tasks div mult --num_steps 100000
```

### Experiment with Different Learning Rates

```bash
python run.py --tasks div --lr 1e-3 --weight_decay 1e-2
python run.py --tasks div --lr 5e-4 --weight_decay 1e-2
python run.py --tasks div --lr 1e-4 --weight_decay 1e-2
```

### Experiment with Different Weight Decay Values

```bash
python run.py --tasks div add --lr 5e-4 --weight_decay 1e-1
python run.py --tasks div add --lr 5e-4 --weight_decay 1e-2
python run.py --tasks div add --lr 5e-4 --weight_decay 1e-3
python run.py --tasks div add --lr 5e-4 --weight_decay 0
```

### Larger Model for Multi-Task Learning

```bash
python run.py --tasks div add sub mult --d_model 256 --nhead 8 --num_layers 4
```

## Project Structure

```
.
├── README.md           # This file
├── requirements.txt    # Python dependencies
├── run.py             # Main script to run experiments
├── data.py            # Data generation and preprocessing
├── model.py           # Transformer model definition
├── train.py           # Training loop with checkpointing and grokking detection
└── utils.py           # Plotting utilities
```

## Key Features

### Multi-Task Learning
The code supports training on multiple modular arithmetic operations simultaneously:
- **Division** (`div`): `a / b mod p` where `b ≠ 0`
- **Addition** (`add`): `a + b mod p`
- **Subtraction** (`sub`): `a - b mod p`
- **Multiplication** (`mult`): `a * b mod p`

Each task uses a special token (`<DIV>`, `<ADD>`, `<SUB>`, `<MULT>`) to identify the operation.

### Grokking Detection
The training loop automatically detects when each task "groks" by monitoring when validation accuracy first exceeds 95%. The grokking step is:
- Logged to the console during training
- Marked with a vertical line on the accuracy plot
- Stored in the training history for analysis

### Checkpoint System
Training progress is automatically saved every 1,000 steps, allowing you to:
- Resume interrupted training runs
- Inspect intermediate model states
- Recover from crashes without losing progress

Checkpoints include:
- Model weights
- Optimizer state
- Full training history
- Current step count

### Per-Task Visualization
The generated plot shows:
- **Left panel**: Accuracy curves for each task with color coding
  - Training accuracy (combined across all tasks)
  - Validation accuracy (separate curve for each task)
  - Vertical lines marking grokking points for each task
- **Right panel**: Loss curves (log-log scale)
  - Training loss (combined)
  - Validation loss (per task)

Task colors: Division (red), Addition (blue), Subtraction (green), Multiplication (purple)

## Expected Output

The script will:
1. Generate a multi-task modular arithmetic dataset
2. Train a transformer model with periodic checkpointing
3. Detect and log grokking events for each task
4. Save a plot showing:
   - Training and validation accuracy over time (demonstrating per-task grokking)
   - Training and validation loss curves (showing double descent)

The plot will be saved to `grokking_result.png` (or your specified path).

## Key Findings

- **Weight decay is crucial**: Higher weight decay (e.g., 1e-3) promotes grokking
- **Training time**: Grokking typically occurs after 10,000-100,000 steps, well after the model has memorized the training set
- **Task-dependent grokking**: Different arithmetic operations may grok at different times
- **Generalization**: The model eventually achieves ~100% accuracy on both training and validation sets for all tasks
- **Multi-task interference**: Training on multiple tasks simultaneously can affect grokking dynamics

## Resuming Training

If training is interrupted, simply run the same command again. The script will automatically:
1. Detect the latest checkpoint
2. Load model and optimizer state
3. Resume from the saved step

```bash
# Initial run (interrupted at step 50,000)
python run.py --tasks div mult --num_steps 100000

# Resume (will continue from step 50,000)
python run.py --tasks div mult --num_steps 100000
```

## References

This implementation is based on the paper:
- [Power et al. (2022). "Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets"](https://arxiv.org/pdf/2201.02177)