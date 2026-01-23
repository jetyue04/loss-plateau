import torch
import torch.nn as nn
from tqdm import tqdm
import os
import json


def save_checkpoint(checkpoint_path, model, optimizer, history, step, config):
    """
    Save training checkpoint.
    
    Args:
        checkpoint_path: Path to save checkpoint
        model: PyTorch model
        optimizer: Optimizer
        history: Training history dictionary
        step: Current training step
        config: Configuration dictionary
    """
    checkpoint = {
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'history': history,
        'config': config,
    }
    
    # Save to temporary file first, then rename (atomic operation)
    temp_path = checkpoint_path + '.tmp'
    torch.save(checkpoint, temp_path)
    os.replace(temp_path, checkpoint_path)
    

def load_checkpoint(checkpoint_path, model, optimizer, device):
    """
    Load training checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        model: PyTorch model
        optimizer: Optimizer
        device: Device to load tensors to
    
    Returns:
        step: Training step to resume from
        history: Training history
        config: Configuration dictionary
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint['step'], checkpoint['history'], checkpoint['config']


def train_epoch(model, loader, optimizer, criterion, device):
    """
    Train for one epoch.
    
    Args:
        model: PyTorch model
        loader: DataLoader for training data
        optimizer: Optimizer
        criterion: Loss function
        device: Device to train on
    
    Returns:
        avg_loss: Average loss for the epoch
        accuracy: Accuracy percentage
    """
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += (predicted == y_batch).sum().item()
        total += y_batch.size(0)
    
    return total_loss / len(loader), 100 * correct / total


def evaluate(model, loader, criterion, device):
    """
    Evaluate model on validation/test set.
    
    Args:
        model: PyTorch model
        loader: DataLoader for evaluation data
        criterion: Loss function
        device: Device to evaluate on
    
    Returns:
        avg_loss: Average loss
        accuracy: Accuracy percentage
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


def detect_grokking(history, train_threshold=95.0, val_threshold=95.0, min_gap=1000):
    """
    Detect when grokking occurs based on training and validation accuracy.
    
    Grokking is defined as: validation accuracy reaching the threshold after
    training accuracy has been above threshold for at least min_gap steps.
    
    Args:
        history: Dictionary containing training history
        train_threshold: Training accuracy threshold (default: 95%)
        val_threshold: Validation accuracy threshold (default: 95%)
        min_gap: Minimum steps between train and val threshold crossing
    
    Returns:
        grokking_step: Step when grokking occurred (None if not detected)
        train_step: Step when training accuracy crossed threshold
    """
    train_step = None
    grokking_step = None
    
    # Find when training accuracy first crosses threshold
    for i, (step, train_acc) in enumerate(zip(history['steps'], history['train_acc'])):
        if train_acc >= train_threshold:
            train_step = step
            break
    
    # Find when validation accuracy crosses threshold (must be after train + min_gap)
    if train_step is not None:
        for i, (step, val_acc) in enumerate(zip(history['steps'], history['val_acc'])):
            if step > train_step + min_gap and val_acc >= val_threshold:
                grokking_step = step
                break
    
    return grokking_step, train_step


def train_model(model, train_loader, val_loader, optimizer, criterion, device, 
                num_steps=350000, log_interval=50, train_threshold=95.0, val_threshold=95.0,
                config=None, checkpoint_dir='checkpoints', checkpoint_interval=1000,
                resume_from=None):
    """
    Main training loop with grokking detection and checkpointing.
    
    Args:
        model: PyTorch model
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        optimizer: Optimizer
        criterion: Loss function
        device: Device to train on
        num_steps: Total number of training steps
        log_interval: Steps between logging
        train_threshold: Training accuracy threshold for grokking detection
        val_threshold: Validation accuracy threshold for grokking detection
        config: Dictionary with training configuration info for plotting
        checkpoint_dir: Directory to save checkpoints
        checkpoint_interval: Steps between checkpoint saves
        resume_from: Path to checkpoint file to resume from (or 'latest')
    
    Returns:
        history: Dictionary containing training history and grokking info
    """
    model = model.to(device)
    
    # Create checkpoint directory if it doesn't exist
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, 'checkpoint_latest.pt')
    
    # Initialize or resume training
    start_step = 0
    
    if resume_from is not None:
        if resume_from == 'latest':
            resume_path = checkpoint_path
        else:
            resume_path = resume_from
            
        if os.path.exists(resume_path):
            print(f"Resuming from checkpoint: {resume_path}")
            start_step, history, saved_config = load_checkpoint(resume_path, model, optimizer, device)
            print(f"Resumed from step {start_step}")
            
            # Update config with any new values
            if config is not None:
                history['config'].update(config)
        else:
            print(f"Checkpoint not found: {resume_path}")
            print("Starting training from scratch...")
            history = {
                'train_loss': [], 'train_acc': [],
                'val_loss': [], 'val_acc': [],
                'steps': [],
                'grokking_detected': False,
                'grokking_step': None,
                'train_threshold_step': None,
                'train_threshold': train_threshold,
                'val_threshold': val_threshold,
                'config': config if config is not None else {}
            }
    else:
        history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [],
            'steps': [],
            'grokking_detected': False,
            'grokking_step': None,
            'train_threshold_step': None,
            'train_threshold': train_threshold,
            'val_threshold': val_threshold,
            'config': config if config is not None else {}
        }
    
    step = start_step
    grokking_announced = False
    
    # Check if grokking was already detected
    if history['grokking_detected']:
        grokking_announced = True
    
    print("Training...")
    pbar = tqdm(initial=start_step, total=num_steps)
    
    try:
        while step < num_steps:
            for X_batch, y_batch in train_loader:
                if step >= num_steps:
                    break
                    
                model.train()
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                
                step += 1
                pbar.update(1)
                
                # Logging
                if step % log_interval == 0:
                    train_loss, train_acc = evaluate(model, train_loader, criterion, device)
                    val_loss, val_acc = evaluate(model, val_loader, criterion, device)
                    
                    history['steps'].append(step)
                    history['train_loss'].append(train_loss)
                    history['train_acc'].append(train_acc)
                    history['val_loss'].append(val_loss)
                    history['val_acc'].append(val_acc)
                    
                    # Check for grokking in real-time
                    if not grokking_announced:
                        grok_step, train_step = detect_grokking(history)
                        if grok_step is not None:
                            grokking_announced = True
                            history['grokking_detected'] = True
                            history['grokking_step'] = grok_step
                            history['train_threshold_step'] = train_step
                            pbar.write(f"\n{'='*60}")
                            pbar.write(f"🎉 GROKKING DETECTED! 🎉")
                            pbar.write(f"Training accuracy reached 95% at step: {train_step:,}")
                            pbar.write(f"Validation accuracy reached 95% at step: {grok_step:,}")
                            pbar.write(f"Grokking delay: {grok_step - train_step:,} steps")
                            pbar.write(f"{'='*60}\n")
                    
                    pbar.set_postfix({
                        'train_acc': f'{train_acc:.1f}%',
                        'val_acc': f'{val_acc:.1f}%'
                    })
                
                # Save checkpoint
                if step % checkpoint_interval == 0:
                    save_checkpoint(checkpoint_path, model, optimizer, history, step, 
                                  history['config'])
                    
                    # Also save a numbered checkpoint periodically (every 10k steps)
                    if step % (checkpoint_interval * 10) == 0:
                        numbered_path = os.path.join(checkpoint_dir, f'checkpoint_step_{step}.pt')
                        save_checkpoint(numbered_path, model, optimizer, history, step,
                                      history['config'])
    
    except KeyboardInterrupt:
        pbar.write("\n\n   Training interrupted by user!")
        pbar.write("Saving checkpoint before exit...")
        save_checkpoint(checkpoint_path, model, optimizer, history, step, history['config'])
        pbar.write(f"  Checkpoint saved to {checkpoint_path}")
        pbar.write(f"Resume training with: --resume latest\n")
        pbar.close()
        raise
    
    except Exception as e:
        pbar.write(f"\n\n  Training crashed with error: {e}")
        pbar.write("Saving checkpoint before exit...")
        save_checkpoint(checkpoint_path, model, optimizer, history, step, history['config'])
        pbar.write(f"  Checkpoint saved to {checkpoint_path}")
        pbar.write(f"Resume training with: --resume latest\n")
        pbar.close()
        raise
    
    pbar.close()
    
    # Save final checkpoint
    save_checkpoint(checkpoint_path, model, optimizer, history, step, history['config'])
    final_path = os.path.join(checkpoint_dir, 'checkpoint_final.pt')
    save_checkpoint(final_path, model, optimizer, history, step, history['config'])
    
    # Final grokking detection
    grokking_step, train_threshold_step = detect_grokking(history)
    history['grokking_detected'] = grokking_step is not None
    history['grokking_step'] = grokking_step
    history['train_threshold_step'] = train_threshold_step
    
    print("\n" + "="*60)
    print("Training complete.")
    print("="*60)
    
    if history['grokking_detected']:
        print(f"  Grokking occurred at step: {grokking_step:,}")
        print(f"  Training threshold (95%) reached at: {train_threshold_step:,}")
        print(f"  Grokking delay: {grokking_step - train_threshold_step:,} steps")
    else:
        print("✗ Grokking not detected within training period")
        if train_threshold_step:
            print(f"  (Training threshold reached at step {train_threshold_step:,}, but validation did not follow)")
    
    print("="*60 + "\n")
    
    return history