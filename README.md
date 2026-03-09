# loss-plateau

This project explores the phenomenon of **training-loss plateaus** in deep learning models, particularly in transformer architectures. It is organized into **two separate modules**, each focusing on different aspects of training dynamics and reproducibility:

1. **Transformer Loss Plateau (TF Loss Plateau)**
   This module investigates how transformer models experience extended periods of nearly constant training loss (plateaus), how these plateaus relate to model architecture and learning dynamics, and strategies to analyze and visualize them.

2. **Grokking**
   This module focuses on the **grokking phenomenon**, where models suddenly generalize after a long period of overfitting. It provides experiments, visualizations, and analysis to reproduce and study this effect.

---

## Getting Started

Each module has its own set of instructions and reproducibility steps. Please navigate to the respective module directories for detailed guidance:

* [TF Loss Plateau Instructions](./tf-loss-plateau)
* [Grokking Instructions](./grokking)

### Running Experiments

Each module contains example scripts and notebooks to reproduce the results:

* **TF Loss Plateau:**

```bash
cd tf-loss-plateau
python train.py --config configs/default.yaml
```

* **Grokking:**

```bash
cd grokking
python run_grokking.py --config configs/grokking.yaml
```

## Project Website

For our front-facing website, visit Here[https://jetyue04.github.io/loss-plateau/](https://jetyue04.github.io/loss-plateau/)
