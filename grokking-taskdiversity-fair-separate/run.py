import argparse
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from data import generate_multitask_dataset, create_vocab, prepare_data
from model import SimpleTransformer
from train import train_model
from utils import plot_grokking

def main():
    parser = argparse.ArgumentParser(description='Train transformer on mixed modular tasks')
    
    # Task configuration
    parser.add_argument('--tasks', nargs='+', default=['div'], choices=['div', 'add', 'sub'],
                        help='List of tasks to train on (e.g. --tasks div add)')
    
    # Standard parameters
    parser.add_argument('--p', type=int, default=97, help='Modulus')
    parser.add_argument('--train_fraction', type=float, default=0.5, help='Fraction of data for training')
    parser.add_argument('--seed', type=int, default=42)
    
    # Model & Train params
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--nhead', type=int, default=4)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-3) # Critical for grokking
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--num_steps', type=int, default=350000)
    
    # Checkpointing
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--save_path', type=str, default='grokking_mixed.png')
    
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Tasks: {args.tasks}")
    
    # 1. Data Generation
    # Note: generate_multitask_dataset ensures EQUAL training examples per task
    train_data, val_data_dict = generate_multitask_dataset(
        tasks=args.tasks, p=args.p, train_fraction=args.train_fraction, seed=args.seed
    )
    
    vocab = create_vocab(p=args.p)
    
    # 2. Prepare Training Data
    X_train, y_train = prepare_data(train_data, vocab)
    train_dataset = TensorDataset(X_train, y_train)
    
    # Ensure batch size isn't larger than dataset
    bs = min(args.batch_size, len(train_dataset))
    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
    
    print(f"Total Train Size: {len(train_data)} ({len(train_data)//len(args.tasks)} per task)")
    
    # 3. Prepare Validation Data (Per Task)
    val_loaders = {}
    for task_name, data_list in val_data_dict.items():
        if len(data_list) > 0:
            X_val, y_val = prepare_data(data_list, vocab)
            # Use full batch for validation to speed it up
            val_loaders[task_name] = DataLoader(TensorDataset(X_val, y_val), batch_size=1024)
            print(f"Val Size ({task_name}): {len(data_list)}")

    # 4. Model Setup
    model = SimpleTransformer(
        vocab_size=len(vocab),
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers
    )
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.98)
    )
    criterion = nn.CrossEntropyLoss()
    
    # 5. Training
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
    
    # 6. Plotting
    plot_grokking(history, save_path=args.save_path)

if __name__ == '__main__':
    main()