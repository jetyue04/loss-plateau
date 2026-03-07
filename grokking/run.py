'''
run.py is the main entry point for training transformers on grokking experiments.

This script orchestrates the complete training pipeline:
1. Parse command-line arguments for experiment configuration
2. Generate multi-task modular arithmetic datasets
3. Create vocabulary and prepare data loaders
4. Initialize transformer model and optimizer
5. Run training with automatic grokking detection and checkpointing
6. Generate visualization plots of training dynamics

Targets (pass as positional arguments):
  all   - Run the full pipeline (data + train + plot). Default if no target given.
  test  - Run the full pipeline on a small unit-test configuration (fast smoke test).
  clean - Delete all generated checkpoints and output plots.

Examples:
  python run.py                          # full run with defaults
  python run.py all                      # same as above, explicit
  python run.py test                     # smoke-test with tiny config
  python run.py clean                    # remove checkpoints + plots
  python run.py --tasks div add --lr 5e-4
'''

import argparse
import os
import shutil
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from data import generate_multitask_dataset, create_vocab, prepare_data
from model import SimpleTransformer
from train import train_model
from utils import plot_grokking


# Unit-test
TEST_CONFIG = {
    'tasks':          ['div', 'add'],
    'p':              11,
    'train_fraction': 0.5,
    'seed':           0,
    'd_model':        32,
    'nhead':          2,
    'num_layers':     1,
    'dropout':        0.0,
    'lr':             1e-3,
    'weight_decay':   1e-3,
    'batch_size':     64,
    'num_steps':      200,
    'checkpoint_dir': 'checkpoints_test',
    'save_path':      'grokking_test.png',
}


def build_pipeline(tasks, p, train_fraction, seed,
                   d_model, nhead, num_layers, dropout,
                   lr, weight_decay, batch_size, num_steps,
                   checkpoint_dir, save_path):
    """
    Execute the full training pipeline for a grokking experiment.

    Generates the dataset, initialises the model, runs training with
    automatic checkpointing and grokking detection, then saves a plot.

    :param tasks:          List of arithmetic task names, e.g. ['div', 'add']
    :param p:              Prime modulus for modular arithmetic
    :param train_fraction: Fraction of data used for training
    :param seed:           Random seed for reproducibility
    :param d_model:        Transformer embedding dimension
    :param nhead:          Number of attention heads
    :param num_layers:     Number of transformer encoder layers
    :param dropout:        Dropout probability in transformer layers
    :param lr:             AdamW learning rate
    :param weight_decay:   AdamW weight-decay (L2 regularisation)
    :param batch_size:     Mini-batch size
    :param num_steps:      Total number of optimization steps
    :param checkpoint_dir: Directory for saving / loading checkpoints
    :param save_path:      File path for the output accuracy plot
    :return:               Training history dictionary
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Training on tasks: {tasks}")

    # Generate Dataset
    print("\n--- Generating Dataset ---")
    train_data, val_data_dict = generate_multitask_dataset(
        tasks=tasks,
        p=p,
        train_fraction=train_fraction,
        seed=seed,
    )

    # Create Vocabulary and Prepare Data
    vocab = create_vocab(p=p)
    X_train, y_train = prepare_data(train_data, vocab)
    train_dataset = TensorDataset(X_train, y_train)

    bs = min(batch_size, len(train_dataset))
    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)

    print(f"Total Train Size: {len(train_data)}")
    print(f"Vocabulary Size:  {len(vocab)}")

    val_loaders = {}
    for task_name, data_list in val_data_dict.items():
        if len(data_list) > 0:
            X_val, y_val = prepare_data(data_list, vocab)
            val_loaders[task_name] = DataLoader(
                TensorDataset(X_val, y_val),
                batch_size=1024,
            )
            print(f"Val Size ({task_name}): {len(data_list)}")

    print("\n--- Initializing Model ---")
    model = SimpleTransformer(
        vocab_size=len(vocab),
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dropout=dropout,
    )
    print(f"Model parameters: {sum(param.numel() for param in model.parameters()):,}")

    # AdamW with betas from the original grokking paper
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        betas=(0.9, 0.98),
    )
    criterion = nn.CrossEntropyLoss()

    # Bundle all hyper-parameters so they are stored inside the checkpoint
    config = dict(
        tasks=tasks, p=p, train_fraction=train_fraction, seed=seed,
        d_model=d_model, nhead=nhead, num_layers=num_layers, dropout=dropout,
        lr=lr, weight_decay=weight_decay, batch_size=batch_size,
        num_steps=num_steps,
    )

    # ===== Step 4: Train Model =====
    print("\n--- Starting Training ---")
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loaders=val_loaders,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        num_steps=num_steps,
        config=config,
        save_dir=checkpoint_dir,
    )

    # ===== Step 5: Generate Visualisation =====
    print("\n--- Generating Visualization ---")
    plot_grokking(history, save_path=save_path)

    # Print grokking summary
    print("\n--- Grokking Summary ---")
    for task, step in history['grok_steps'].items():
        if step is not None:
            print(f"  {task.upper()}: Grokked at step {step:,}")
        else:
            print(f"  {task.upper()}: Did not grok within {num_steps:,} steps")

    print(f"\nExperiment complete! Results saved to {save_path}")
    return history


def run_clean(checkpoint_dir, save_path):
    """
    Remove all generated artefacts (checkpoints directory and output plot).

    :param checkpoint_dir: Checkpoint directory to delete
    :param save_path:      Plot file to delete
    """
    removed = []
    if os.path.isdir(checkpoint_dir):
        shutil.rmtree(checkpoint_dir)
        removed.append(checkpoint_dir + '/')
    if os.path.isfile(save_path):
        os.remove(save_path)
        removed.append(save_path)
    if removed:
        print(f"Removed: {', '.join(removed)}")
    else:
        print("Nothing to clean.")


def main():
    """
    Parse command-line arguments and dispatch to the appropriate target.

    Positional argument ``target`` controls which action to run:
      all   - full training pipeline (default)
      test  - quick smoke-test with a tiny dataset and model
      clean - delete checkpoints and output plots
    """
    parser = argparse.ArgumentParser(
        description='Train a transformer on modular arithmetic tasks to study grokking.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Targets:
  all    Full pipeline with the given flags (default).
  test   Smoke-test: tiny config, finishes in seconds.
  clean  Delete generated checkpoints and plots.

Examples:
  python run.py
  python run.py all --tasks div add --lr 5e-4 --weight_decay 1e-2
  python run.py test
  python run.py clean
  python run.py --tasks div add sub mult --num_steps 150000
  python run.py --tasks div mult --d_model 256 --nhead 8 --num_layers 4
        ''',
    )

    # Optional positional target
    parser.add_argument('target', nargs='?', default='all',
                        choices=['all', 'test', 'clean'],
                        help='Pipeline target to run (default: all)')

    # Task configuration
    parser.add_argument('--tasks', nargs='+', default=['div'],
                        choices=['div', 'add', 'sub', 'mult'],
                        help='Modular arithmetic tasks to train on (default: div)')

    # Data parameters
    parser.add_argument('--p', type=int, default=97,
                        help='Prime modulus for modular arithmetic (default: 97)')
    parser.add_argument('--train_fraction', type=float, default=0.5,
                        help='Fraction of data used for training (default: 0.5)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')

    # Model architecture parameters
    parser.add_argument('--d_model', type=int, default=128,
                        help='Transformer embedding dimension (default: 128)')
    parser.add_argument('--nhead', type=int, default=4,
                        help='Number of attention heads (default: 4)')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='Number of transformer encoder layers (default: 2)')
    parser.add_argument('--dropout', type=float, default=0.0,
                        help='Dropout probability in transformer layers (default: 0.0)')

    # Optimization parameters
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='AdamW learning rate (default: 1e-3)')
    parser.add_argument('--weight_decay', type=float, default=1e-3,
                        help='AdamW weight decay - crucial for grokking (default: 1e-3)')

    # Training parameters
    parser.add_argument('--batch_size', type=int, default=512,
                        help='Mini-batch size (default: 512)')
    parser.add_argument('--num_steps', type=int, default=400000,
                        help='Total optimization steps (default: 400000)')

    # Output parameters
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help='Directory for saving/loading checkpoints (default: checkpoints)')
    parser.add_argument('--save_path', type=str, default='grokking_result.png',
                        help='Output plot file path (default: grokking_result.png)')

    args = parser.parse_args()

    if args.target == 'clean':
        run_clean(args.checkpoint_dir, args.save_path)
        return

    if args.target == 'test':
        print("=== Running smoke test (test target) ===")
        cfg = TEST_CONFIG
    else:
        # 'all' or default -- use CLI arguments
        cfg = dict(
            tasks=args.tasks,
            p=args.p,
            train_fraction=args.train_fraction,
            seed=args.seed,
            d_model=args.d_model,
            nhead=args.nhead,
            num_layers=args.num_layers,
            dropout=args.dropout,
            lr=args.lr,
            weight_decay=args.weight_decay,
            batch_size=args.batch_size,
            num_steps=args.num_steps,
            checkpoint_dir=args.checkpoint_dir,
            save_path=args.save_path,
        )

    build_pipeline(**cfg)


if __name__ == '__main__':
    main()