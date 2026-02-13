import torch
import torch.nn as nn
from tqdm import tqdm
import os
import glob

def save_checkpoint(state, save_dir='checkpoints', filename='checkpoint.pt'):
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, filename)
    torch.save(state, path)

def load_checkpoint(save_dir='checkpoints', filename='checkpoint.pt', device='cpu'):
    path = os.path.join(save_dir, filename)
    if os.path.exists(path):
        print(f"Resuming from checkpoint: {path}")
        return torch.load(path, map_location=device)
    return None

def evaluate(model, loader, criterion, device):
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
                num_steps=250000, log_interval=50, checkpoint_interval=1000,
                config=None, save_dir='checkpoints'):
    
    model = model.to(device)
    
    # Initialize history
    history = {
        'steps': [],
        'train_loss': [], 'train_acc': [],
        'val_stats': {task: {'loss': [], 'acc': []} for task in val_loaders.keys()},
        # NEW: Dictionary to store the step where each task first "groks" (>95% acc)
        'grok_steps': {task: None for task in val_loaders.keys()},
        'config': config if config else {}
    }
    
    start_step = 0
    
    # Attempt to load checkpoint
    checkpoint = load_checkpoint(save_dir, device=device)
    if checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_step = checkpoint['step']
        history = checkpoint['history']
    
    if start_step >= num_steps:
        print("Training already completed based on checkpoint.")
        return history

    print(f"Training from step {start_step} to {num_steps}...")
    
    def infinite_iter(loader):
        while True:
            for batch in loader:
                yield batch
    
    train_iter = infinite_iter(train_loader)
    
    model.train()
    pbar = tqdm(total=num_steps, initial=start_step)
    
    try:
        for step in range(start_step + 1, num_steps + 1):
            X_batch, y_batch = next(train_iter)
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            pbar.update(1)
            
            if step % log_interval == 0:
                train_loss, train_acc = evaluate(model, train_loader, criterion, device)
                
                history['steps'].append(step)
                history['train_loss'].append(train_loss)
                history['train_acc'].append(train_acc)
                
                postfix_stats = {'train_acc': f"{train_acc:.1f}%"}
                
                # Check validation for each task
                for task_name, loader in val_loaders.items():
                    v_loss, v_acc = evaluate(model, loader, criterion, device)
                    history['val_stats'][task_name]['loss'].append(v_loss)
                    history['val_stats'][task_name]['acc'].append(v_acc)
                    postfix_stats[f'val_{task_name}'] = f"{v_acc:.1f}%"
                    
                    # NEW: Detect Grokking (First time crossing 95%)
                    if history['grok_steps'][task_name] is None and v_acc >= 95.0:
                        history['grok_steps'][task_name] = step
                        tqdm.write(f" Grokking detected for {task_name.upper()} at step {step}!")
                
                pbar.set_postfix(postfix_stats)

            if step % checkpoint_interval == 0:
                save_checkpoint({
                    'step': step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'history': history
                }, save_dir)
                
    except KeyboardInterrupt:
        print("\nTraining interrupted! Saving checkpoint...")
        save_checkpoint({
            'step': step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'history': history
        }, save_dir)
        return history
        
    pbar.close()
    
    save_checkpoint({
        'step': num_steps,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'history': history
    }, save_dir)
    
    return history