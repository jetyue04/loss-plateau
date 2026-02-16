'''
run.py is the main entry point for training transformers on grokking experiments.

This script orchestrates the complete training pipeline:
1. Parse command-line arguments for experiment configuration
2. Generate multi-task modular arithmetic datasets
3. Create vocabulary and prepare data loaders
4. Initialize transformer model and optimizer
5. Run training with automatic grokking detection and checkpointing
6. Generate visualization plots of training dynamics
'''

import argparse
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from data import generate_multitask_dataset, create_vocab, prepare_data
from model import SimpleTransformer
from train import train_model
from utils import plot_grokking


def main():
    """
    Main function to run grokking experiments.
    
    Parses command-line arguments, sets up the dataset and model, trains the
    transformer on modular arithmetic tasks, and generates visualization plots.
    
    The training process includes:
    - Multi-task support for division, addition, subtraction, and multiplication
    - Automatic checkpointing for resumable training
    - Per-task validation tracking
    - Grokking detection (when validation accuracy exceeds 95%)
    - Comprehensive plotting of training dynamics
    
    Command-line arguments control all aspects of the experiment including
    data generation, model architecture, optimization, and training duration.
    """
    parser = argparse.ArgumentParser(
        description='Train transformer on modular arithmetic tasks to study grokking',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Single task training
  python run.py --tasks div --num_steps 100000
  
  # Multi-task training
  python run.py --tasks div add sub mult --num_steps 150000
  
  # Experiment with hyperparameters
  python run.py --tasks div --lr 5e-4 --weight_decay 1e-2
  
  # Larger model
  python run.py --tasks div mult --d_model 256 --nhead 8 --num_layers 4
        '''
    )
    
    # Task configuration
    parser.add_argument('--tasks', nargs='+', default=['div'], 
                        choices=['div', 'add', 'sub', 'mult'],
                        help='List of modular arithmetic tasks to train on')
    
    # Data parameters
    parser.add_argument('--p', type=int, default=97,
                        help='Prime modulus for modular arithmetic')
    parser.add_argument('--train_fraction', type=float, default=0.5,
                        help='Fraction of data to use for training')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    # Model architecture parameters
    parser.add_argument('--d_model', type=int, default=128,
                        help='Dimension of model embeddings')
    parser.add_argument('--nhead', type=int, default=4,
                        help='Number of attention heads')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='Number of transformer layers')
    
    # Optimization parameters
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate for AdamW optimizer')
    parser.add_argument('--weight_decay', type=float, default=1e-3,
                        help='Weight decay (L2 regularization) - crucial for grokking')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=512,
                        help='Batch size for training')
    parser.add_argument('--num_steps', type=int, default=400000,
                        help='Total number of training steps')
    
    # Output parameters
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help='Directory to save training checkpoints')
    parser.add_argument('--save_path', type=str, default='grokking_result.png',
                        help='Path to save the visualization plot')
    
    args = parser.parse_args()
    
    # Determine device (GPU if available, otherwise CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Training on tasks: {args.tasks}")
    
    # ===== Step 1: Generate Dataset =====
    print("\n--- Generating Dataset ---")
    train_data, val_data_dict = generate_multitask_dataset(
        tasks=args.tasks, 
        p=args.p, 
        train_fraction=args.train_fraction, 
        seed=args.seed
    )
    
    # ===== Step 2: Create Vocabulary and Prepare Data =====
    vocab = create_vocab(p=args.p)
    X_train, y_train = prepare_data(train_data, vocab)
    train_dataset = TensorDataset(X_train, y_train)
    
    # Adjust batch size if dataset is smaller
    bs = min(args.batch_size, len(train_dataset))
    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
    
    print(f"Total Train Size: {len(train_data)}")
    print(f"Vocabulary Size: {len(vocab)}")
    
    # Create separate validation loaders for each task
    val_loaders = {}
    for task_name, data_list in val_data_dict.items():
        if len(data_list) > 0:
            X_val, y_val = prepare_data(data_list, vocab)
            val_loaders[task_name] = DataLoader(
                TensorDataset(X_val, y_val), 
                batch_size=1024
            )
            print(f"Val Size ({task_name}): {len(data_list)}")

    # ===== Step 3: Initialize Model and Optimizer =====
    print("\n--- Initializing Model ---")
    model = SimpleTransformer(
        vocab_size=len(vocab), 
        d_model=args.d_model, 
        nhead=args.nhead, 
        num_layers=args.num_layers
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # AdamW optimizer with specific beta values as in original grokking paper
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=args.lr, 
        weight_decay=args.weight_decay, 
        betas=(0.9, 0.98)
    )
    
    criterion = nn.CrossEntropyLoss()
    
    # ===== Step 4: Train Model =====
    print("\n--- Starting Training ---")
    history = train_model(
        model=model, 
        train_loader=train_loader, 
        val_loaders=val_loaders,
        optimizer=optimizer, 
        criterion=criterion, 
        device=device,
        num_steps=args.num_steps, 
        save_dir=args.checkpoint_dir
    )
    
    # ===== Step 5: Generate Visualization =====
    print("\n--- Generating Visualization ---")
    plot_grokking(history, save_path=args.save_path)
    
    # Print summary of grokking events
    print("\n--- Grokking Summary ---")
    for task, step in history['grok_steps'].items():
        if step is not None:
            print(f"{task.upper()}: Grokked at step {step:,}")
        else:
            print(f"{task.upper()}: Did not grok yet")
    
    print(f"\nExperiment complete! Results saved to {args.save_path}")


if __name__ == '__main__':
    main()
