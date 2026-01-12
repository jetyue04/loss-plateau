# Understanding Grokking: From Memorization to Generalization

This repository provides an implementation to reproduce and investigate the **grokking** phenomenon, where neural networks suddenly generalize long after overfitting the training set. The code trains a simple transformer on modular arithmetic (specifically modular division), replicating the abrupt generalization transition and double-descent loss curves described in [Power et al. (2022)](https://arxiv.org/pdf/2201.02177). It supports the configuration of different parameters to investigate their influence on the timing and emergence of grokking. This provides a reproducible environment for studying the transition from memorization to generalization in neural networks.

## Overview

The model learns to perform modular division: given `a / b mod p`, predict the result. With proper regularization (weight decay), the model first memorizes the training set, then after many more optimization steps, suddenly achieves perfect generalization on the validation set.

<img src="./grokking_plot.png" alt="Grokking Visualization" style="width: 500px; height: auto;">

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

Run with default parameters:
```bash
python run.py
```

This will train for 250,000 steps with learning rate 1e-3 and weight decay 1e-3, and generate a plot showing the grokking phenomenon.

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

## Command Line Arguments

### Optimizer Parameters (Key Parameters)
- `--lr`: Learning rate (default: 1e-3)
- `--weight_decay`: Weight decay for AdamW optimizer (default: 1e-3)

### Data Parameters
- `--p`: Prime modulus for modular division (default: 97)
- `--train_fraction`: Fraction of data for training (default: 0.5)
- `--seed`: Random seed for reproducibility (default: 42)

### Model Parameters
- `--d_model`: Dimension of model embeddings (default: 128)
- `--nhead`: Number of attention heads (default: 4)
- `--num_layers`: Number of transformer layers (default: 2)
- `--dropout`: Dropout rate (default: 0.0)

### Training Parameters
- `--batch_size`: Batch size (default: 512)
- `--num_steps`: Total training steps (default: 250000)
- `--log_interval`: Steps between logging (default: 50)

### Output Parameters
- `--save_path`: Path to save the plot (default: grokking_result.png)

## Examples

### Experiment with different learning rates:
```bash
python run.py --lr 1e-3 --weight_decay 1e-2
python run.py --lr 5e-4 --weight_decay 1e-2
python run.py --lr 1e-4 --weight_decay 1e-2
```

### Experiment with different weight decay values:
```bash
python run.py --lr 5e-4 --weight_decay 1e-1
python run.py --lr 5e-4 --weight_decay 1e-2
python run.py --lr 5e-4 --weight_decay 1e-3
python run.py --lr 5e-4 --weight_decay 0
```

### Larger model:
```bash
python run.py --d_model 256 --nhead 8 --num_layers 4
```

## Project Structure

```
.
├── README.md           # This file
├── requirements.txt    # Python dependencies
├── run.py             # Main script to run experiments
├── data.py            # Data generation and preprocessing
├── model.py           # Transformer model definition
├── train.py           # Training loop and evaluation
└── utils.py           # Plotting utilities
```

## Expected Output

The script will:
1. Generate a modular division dataset
2. Train a transformer model
3. Save a plot showing:
   - Training and validation accuracy over time (demonstrating grokking)
   - Training and validation loss curves (showing double descent)

The plot will be saved to `grokking_result.png` (or your specified path).

## Key Findings

- **Weight decay is crucial**: Higher weight decay (e.g., 1e-3) promotes grokking.
- **Training time**: Grokking typically occurs after 10,000-100,000 steps, well after the model has memorized the training set
- **Generalization**: The model eventually achieves ~100% accuracy on both training and validation sets

## References

This implementation is based on the paper:
- [Power et al. (2022). "Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets"](https://arxiv.org/pdf/2201.02177)
