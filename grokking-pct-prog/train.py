'''
train.py contains the training loop, evaluation functions, and checkpoint management
for grokking experiments.

This module handles:
- Training transformer models on modular arithmetic tasks
- Per-task validation and metric tracking
- Automatic grokking detection (when validation accuracy exceeds 95%)
- Checkpoint saving and loading for resumable training
- Progress tracking and logging
'''

import torch
import torch.nn as nn
from tqdm import tqdm
import os
import glob


def save_checkpoint(state, save_dir='checkpoints', filename='checkpoint.pt'):
    """
    Save a training checkpoint to disk.
    
    Creates the checkpoint directory if it doesn't exist and saves the complete
    training state including model weights, optimizer state, and training history.
    
    :param state: Dictionary containing checkpoint data (model_state_dict, 
                  optimizer_state_dict, step, history)
    :param save_dir: Directory to save checkpoints
    :param filename: Name of the checkpoint file
    """
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, filename)
    torch.save(state, path)


def load_checkpoint(save_dir='checkpoints', filename='checkpoint.pt', device='cpu'):
    """
    Load a training checkpoint from disk if it exists.
    
    :param save_dir: Directory containing checkpoints
    :param filename: Name of the checkpoint file to load
    :param device: Device to map the checkpoint to ('cpu' or 'cuda')
    :return: Checkpoint dictionary if found, None otherwise
    """
    path = os.path.join(save_dir, filename)
    if os.path.exists(path):
        print(f"Resuming from checkpoint: {path}")
        return torch.load(path, map_location=device)
    return None


def evaluate(model, loader, criterion, device):
    """
    Evaluate model performance on a dataset.
    
    Computes average loss and accuracy over all batches in the data loader.
    Sets model to eval mode and disables gradient computation for efficiency.
    
    :param model: The transformer model to evaluate
    :param loader: DataLoader containing evaluation data
    :param criterion: Loss function (e.g., CrossEntropyLoss)
    :param device: Device to run evaluation on ('cpu' or 'cuda')
    :return: Tuple of (average_loss, accuracy_percentage)
    """
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += (predicted == y_batch).sum().item()
            total += y_batch.size(0)
            
    return total_loss / len(loader), 100 * correct / total


def train_model(model, train_loader, val_loaders, optimizer, criterion, device, 
                num_steps=400000, log_interval=50, checkpoint_interval=1000,
                config=None, save_dir='checkpoints'):
    """
    Train a transformer model on modular arithmetic tasks with multi-task validation.
    
    This function implements the main training loop with the following features:
    - Step-based training (not epoch-based) for precise control
    - Per-task validation tracking
    - Automatic grokking detection (when validation accuracy first exceeds 95%)
    - Periodic checkpointing for resumable training
    - Progress bar with real-time metrics
    - Graceful interrupt handling (Ctrl+C saves checkpoint)
    
    :param model: Transformer model to train
    :param train_loader: DataLoader for training data (combined across tasks)
    :param val_loaders: Dictionary mapping task names to validation DataLoaders
    :param optimizer: PyTorch optimizer (typically AdamW)
    :param criterion: Loss function (typically CrossEntropyLoss)
    :param device: Device to train on ('cpu' or 'cuda')
    :param num_steps: Total number of training steps
    :param log_interval: Steps between evaluation and logging
    :param checkpoint_interval: Steps between checkpoint saves
    :param config: Optional configuration dict to store in history
    :param save_dir: Directory to save checkpoints
    :return: Dictionary containing complete training history with keys:
        - 'steps': List of step numbers where metrics were logged
        - 'train_loss': Training loss at each logged step
        - 'train_acc': Training accuracy at each logged step
        - 'val_stats': Dict mapping task names to their loss and accuracy lists
        - 'grok_steps': Dict mapping task names to step where they first grokked
        - 'config': Configuration parameters
    """
    
    model = model.to(device)
    
    # Initialize history to track all metrics
    history = {
        'steps': [],
        'progress_pct': [],  # Percentage of total training completed
        'train_loss': [], 
        'train_acc': [],
        'val_stats': {task: {'loss': [], 'acc': []} for task in val_loaders.keys()},
        'grok_steps': {task: None for task in val_loaders.keys()},
        'grok_pcts': {task: None for task in val_loaders.keys()},  # Grok point as percentage
        'num_steps': num_steps,
        'config': config if config else {}
    }
    
    start_step = 0
    
    # Attempt to load checkpoint for resuming training
    checkpoint = load_checkpoint(save_dir, device=device)
    if checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_step = checkpoint['step']
        history = checkpoint['history']
        # Ensure new keys exist for older checkpoints
        if 'progress_pct' not in history:
            history['progress_pct'] = [s / num_steps * 100 for s in history['steps']]
        if 'grok_pcts' not in history:
            history['grok_pcts'] = {
                task: (gs / num_steps * 100 if gs is not None else None)
                for task, gs in history['grok_steps'].items()
            }
        if 'num_steps' not in history:
            history['num_steps'] = num_steps
    
    if start_step >= num_steps:
        print("Training already completed based on checkpoint.")
        return history

    print(f"Training from step {start_step} to {num_steps}...")
    
    def infinite_iter(loader):
        """
        Create an infinite iterator over the dataloader.
        
        This allows step-based training without worrying about epoch boundaries.
        """
        while True:
            for batch in loader:
                yield batch
    
    train_iter = infinite_iter(train_loader)
    
    model.train()
    pbar = tqdm(total=num_steps, initial=start_step)
    
    try:
        for step in range(start_step + 1, num_steps + 1):
            # Get next training batch
            X_batch, y_batch = next(train_iter)
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            # Standard training step
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            pbar.update(1)
            
            # Periodic evaluation and logging
            if step % log_interval == 0:
                train_loss, train_acc = evaluate(model, train_loader, criterion, device)
                
                pct = step / num_steps * 100
                history['steps'].append(step)
                history['progress_pct'].append(pct)
                history['train_loss'].append(train_loss)
                history['train_acc'].append(train_acc)
                
                postfix_stats = {'train_acc': f"{train_acc:.1f}%"}
                
                # Evaluate on each task separately
                for task_name, loader in val_loaders.items():
                    v_loss, v_acc = evaluate(model, loader, criterion, device)
                    history['val_stats'][task_name]['loss'].append(v_loss)
                    history['val_stats'][task_name]['acc'].append(v_acc)
                    postfix_stats[f'val_{task_name}'] = f"{v_acc:.1f}%"
                    
                    # Detect grokking: first time validation accuracy exceeds 95%
                    if history['grok_steps'][task_name] is None and v_acc >= 95.0:
                        history['grok_steps'][task_name] = step
                        history['grok_pcts'][task_name] = pct
                        tqdm.write(f"✓ Grokking detected for {task_name.upper()} at step {step} ({pct:.1f}%)")
                
                pbar.set_postfix(postfix_stats)

            # Periodic checkpoint saving
            if step % checkpoint_interval == 0:
                save_checkpoint({
                    'step': step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'history': history
                }, save_dir)
                
    except KeyboardInterrupt:
        # Handle Ctrl+C gracefully by saving checkpoint
        print("\nTraining interrupted! Saving checkpoint...")
        save_checkpoint({
            'step': step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'history': history
        }, save_dir)
        return history
        
    pbar.close()
    
    # Save final checkpoint
    save_checkpoint({
        'step': num_steps,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'history': history
    }, save_dir)
    
    return history
