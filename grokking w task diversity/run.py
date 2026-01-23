import argparse
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from data import (generate_modular_division_dataset, generate_mixed_task_dataset,
                  create_vocab, prepare_data)
from model import SimpleTransformer
from train import train_model
from utils import plot_grokking


def main():
    parser = argparse.ArgumentParser(description='Train a transformer to demonstrate grokking')
    
    # Data parameters
    parser.add_argument('--p', type=int, default=97, 
                        help='Prime modulus for modular division (default: 97)')
    parser.add_argument('--train_fraction', type=float, default=0.5,
                        help='Fraction of data for training (default: 0.5)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    
    # Multi-task parameters
    parser.add_argument('--multi_task', action='store_true',
                        help='Enable multi-task training with mixed operations')
    parser.add_argument('--task_division', type=float, default=0.5,
                        help='Proportion of division tasks (default: 0.5)')
    parser.add_argument('--task_addition', type=float, default=0.5,
                        help='Proportion of addition tasks (default: 0.5)')
    parser.add_argument('--task_subtraction', type=float, default=0.0,
                        help='Proportion of subtraction tasks (default: 0.0)')
    
    # Model parameters
    parser.add_argument('--d_model', type=int, default=128,
                        help='Dimension of model embeddings (default: 128)')
    parser.add_argument('--nhead', type=int, default=4,
                        help='Number of attention heads (default: 4)')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='Number of transformer layers (default: 2)')
    parser.add_argument('--dropout', type=float, default=0.0,
                        help='Dropout rate (default: 0.0)')
    
    # Training parameters
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate (default: 1e-3)')
    parser.add_argument('--weight_decay', type=float, default=1e-3,
                        help='Weight decay for AdamW (default: 1e-3)')
    parser.add_argument('--batch_size', type=int, default=512,
                        help='Batch size (default: 512)')
    parser.add_argument('--num_steps', type=int, default=250000,
                        help='Total training steps (default: 250000)')
    parser.add_argument('--log_interval', type=int, default=50,
                        help='Steps between logging (default: 50)')
    
    # Checkpoint parameters
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints (default: checkpoints)')
    parser.add_argument('--checkpoint_interval', type=int, default=1000,
                        help='Steps between checkpoint saves (default: 1000)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint: "latest" or path to checkpoint file')
    
    # Output parameters
    parser.add_argument('--save_path', type=str, default='grokking_result.png',
                        help='Path to save the plot (default: grokking_result.png)')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Generate dataset
    if args.multi_task:
        # Build task mix dictionary
        task_mix = {}
        if args.task_division > 0:
            task_mix['division'] = args.task_division
        if args.task_addition > 0:
            task_mix['addition'] = args.task_addition
        if args.task_subtraction > 0:
            task_mix['subtraction'] = args.task_subtraction
        
        # Normalize to sum to 1.0
        total = sum(task_mix.values())
        task_mix = {k: v/total for k, v in task_mix.items()}
        
        print(f"\nGenerating MULTI-TASK dataset (p={args.p}, train_fraction={args.train_fraction})...")
        print(f"Task mix: {task_mix}")
        
        train_data, val_data = generate_mixed_task_dataset(
            p=args.p,
            train_fraction=args.train_fraction,
            seed=args.seed,
            task_mix=task_mix
        )
    else:
        print(f"\nGenerating dataset (p={args.p}, train_fraction={args.train_fraction})...")
        train_data, val_data = generate_modular_division_dataset(
            p=args.p, 
            train_fraction=args.train_fraction, 
            seed=args.seed
        )
    
    print(f"Train size: {len(train_data)}, Val size: {len(val_data)}")
    
    # Create vocabulary
    vocab = create_vocab(p=args.p, multi_task=args.multi_task)
    vocab_size = len(vocab)
    print(f"Vocabulary size: {vocab_size}")
    
    # Prepare data
    X_train, y_train = prepare_data(train_data, vocab, multi_task=args.multi_task)
    X_val, y_val = prepare_data(val_data, vocab, multi_task=args.multi_task)
    
    # Create data loaders
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    
    batch_size = min(args.batch_size, len(train_dataset) // 2)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Create model
    print(f"\nCreating model (d_model={args.d_model}, nhead={args.nhead}, num_layers={args.num_layers})...")
    model = SimpleTransformer(
        vocab_size=vocab_size,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dropout=args.dropout
    )
    model_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {model_params:,}")
    
    # Create optimizer
    print(f"\nOptimizer: AdamW (lr={args.lr}, weight_decay={args.weight_decay})")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.98)
    )
    
    criterion = nn.CrossEntropyLoss()
    
    # Prepare config dictionary for visualization
    config = {
        'model_params': model_params,
        'd_model': args.d_model,
        'nhead': args.nhead,
        'num_layers': args.num_layers,
        'dropout': args.dropout,
        'optimizer': 'AdamW',
        'lr': args.lr,
        'weight_decay': args.weight_decay,
        'betas': (0.9, 0.98),
        'batch_size': batch_size,
        'num_steps': args.num_steps,
        'p': args.p,
        'train_fraction': args.train_fraction,
        'train_size': len(train_data),
        'val_size': len(val_data),
        'seed': args.seed,
        'multi_task': args.multi_task,
    }
    
    if args.multi_task:
        config['task_mix'] = task_mix
    
    # Train model
    print(f"\nTraining for {args.num_steps} steps...")
    if args.resume:
        print(f"Note: Will attempt to resume from checkpoint")
    print(f"Checkpoints will be saved to: {args.checkpoint_dir}/")
    print(f"Checkpoint interval: every {args.checkpoint_interval} steps")
    print(f"\n  If training is interrupted, resume with: --resume latest\n")
    
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        num_steps=args.num_steps,
        log_interval=args.log_interval,
        config=config,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_interval=args.checkpoint_interval,
        resume_from=args.resume
    )
    
    # Plot results
    print("\nPlotting results...")
    plot_grokking(history, save_path=args.save_path)


if __name__ == '__main__':
    main()
