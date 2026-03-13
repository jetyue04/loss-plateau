# loss-plateau

## Problem Description
Transformer models, which power modern AI systems such as GPT and BERT, often spend significant compute in inefficient training regimes where learning appears to stall. Two commonly observed phenomena are:

- Training-loss plateaus – extended periods during which training loss remains nearly constant before suddenly decreasing.

- Grokking (generalization plateau) – a phenomenon where a model initially memorizes the training data and only generalizes after a long delay.

Despite being studied independently, these behaviors may arise from similar optimization dynamics.

In this project, we investigate whether these phenomena share common underlying causes and whether the same interventions can shorten both forms of stalled learning. To isolate these dynamics, we focus on controlled modular arithmetic tasks, allowing us to systematically analyze transformer training behavior.

The project is divided into two modules:

- Transformer Loss Plateau (TF Loss Plateau) – studies prolonged plateaus in training loss during optimization.

- Grokking – studies delayed generalization after extended overfitting.
---

## Getting Started

Each module has its own set of instructions and reproducibility steps. Please navigate to the respective module directories for detailed guidance:

* [TF Loss Plateau Instructions](./tf-loss-plateau)
* [Grokking Instructions](./grokking)

### Directory Structure
```
loss-plateau/
│
├── tf-loss-plateau/        # Transformer loss plateau experiments
│   ├── train.py
│   ├── configs/
│   └── utils/
│
├── grokking/               # Grokking experiments
│   ├── run_grokking.py
│   ├── configs/
│   └── analysis/
│
├── docs/                   # Project website files
│
├── requirements.txt
└── README.md
```

## Installation
Clone the repository:
```bash
git clone https://github.com/jetyue04/loss-plateau.git
cd loss-plateau
```

Install dependencies:
```bash
pip install -r requirements.txt
```

Addtionally, navigate to each submodule for the installation/dependencies.

## Running Experiments

Each module contains example scripts and notebooks to reproduce the results:

* **TF Loss Plateau:**

```bash
cd tf-loss-plateau
python train.py --config configs/mws.yaml
```

* **Grokking:**

```bash
cd grokking
python run.py
```
## Dataset
All datasets used in this project are synthetically generated and do not require external downloads.

## Expected Outputs 
Running the experiments will generate:
- Training Logs

- Saved Model Checkpoints

Plots and Visualizations

- Generated plots include:

Additoinally, the tf-loss-plateau module is generated to automatically log into weights and biases for visualization.

## Project Website

For our front-facing website, visit Here[https://jetyue04.github.io/loss-plateau/](https://jetyue04.github.io/loss-plateau/)
