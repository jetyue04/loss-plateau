import torch
import torch.nn as nn
from tqdm import tqdm


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


def train_model(model, train_loader, val_loader, optimizer, criterion, device, 
                num_steps=350000, log_interval=50):
    """
    Main training loop.
    
    Args:
        model: PyTorch model
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        optimizer: Optimizer
        criterion: Loss function
        device: Device to train on
        num_steps: Total number of training steps
        log_interval: Steps between logging
    
    Returns:
        history: Dictionary containing training history
    """
    model = model.to(device)
    
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'steps': []
    }
    
    step = 0
    print("Training...")
    pbar = tqdm(total=num_steps)
    
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
            
            if step % log_interval == 0:
                train_loss, train_acc = evaluate(model, train_loader, criterion, device)
                val_loss, val_acc = evaluate(model, val_loader, criterion, device)
                
                history['steps'].append(step)
                history['train_loss'].append(train_loss)
                history['train_acc'].append(train_acc)
                history['val_loss'].append(val_loss)
                history['val_acc'].append(val_acc)
                
                pbar.set_postfix({
                    'train_acc': f'{train_acc:.1f}%',
                    'val_acc': f'{val_acc:.1f}%'
                })
    
    pbar.close()
    print("Training complete.")
    
    return history