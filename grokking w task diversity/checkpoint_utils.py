#!/usr/bin/env python3
"""
Utility script for managing training checkpoints.
"""

import torch
import os
import argparse
from datetime import datetime
import json


def list_checkpoints(checkpoint_dir):
    """List all checkpoints in directory with details."""
    if not os.path.exists(checkpoint_dir):
        print(f"Directory not found: {checkpoint_dir}")
        return
    
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pt')]
    
    if not checkpoint_files:
        print(f"No checkpoints found in {checkpoint_dir}")
        return
    
    print(f"\n{'='*80}")
    print(f"Checkpoints in: {checkpoint_dir}")
    print(f"{'='*80}\n")
    
    checkpoints_info = []
    
    for filename in checkpoint_files:
        filepath = os.path.join(checkpoint_dir, filename)
        
        try:
            checkpoint = torch.load(filepath, map_location='cpu')
            
            step = checkpoint.get('step', 'Unknown')
            history = checkpoint.get('history', {})
            config = checkpoint.get('config', {})
            
            train_acc = history.get('train_acc', [])
            val_acc = history.get('val_acc', [])
            
            file_size = os.path.getsize(filepath) / (1024 * 1024)  # MB
            mod_time = datetime.fromtimestamp(os.path.getmtime(filepath))
            
            info = {
                'filename': filename,
                'step': step,
                'train_acc': train_acc[-1] if train_acc else 0,
                'val_acc': val_acc[-1] if val_acc else 0,
                'grokking': history.get('grokking_detected', False),
                'grokking_step': history.get('grokking_step', None),
                'size_mb': file_size,
                'modified': mod_time,
                'multi_task': config.get('multi_task', False),
            }
            
            checkpoints_info.append(info)
            
        except Exception as e:
            print(f"Error loading {filename}: {e}")
    
    # Sort by step number
    checkpoints_info.sort(key=lambda x: x['step'] if isinstance(x['step'], int) else 0)
    
    # Print table
    print(f"{'Filename':<35} {'Step':>10} {'Train Acc':>10} {'Val Acc':>10} {'Size':>10} {'Grokking':>10}")
    print(f"{'-'*35} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    
    for info in checkpoints_info:
        grokking_str = f"✓ @{info['grokking_step']}" if info['grokking'] else "✗"
        multi_task_marker = " [MT]" if info['multi_task'] else ""
        
        print(f"{info['filename']:<35} {info['step']:>10,} "
              f"{info['train_acc']:>9.1f}% {info['val_acc']:>9.1f}% "
              f"{info['size_mb']:>8.1f}MB {grokking_str:>10}{multi_task_marker}")
    
    print(f"\n{'='*80}\n")


def inspect_checkpoint(checkpoint_path):
    """Show detailed information about a checkpoint."""
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        return
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    step = checkpoint.get('step', 'Unknown')
    history = checkpoint.get('history', {})
    config = checkpoint.get('config', {})
    
    print(f"\n{'='*80}")
    print(f"Checkpoint Details: {os.path.basename(checkpoint_path)}")
    print(f"{'='*80}\n")
    
    # Training progress
    print("TRAINING PROGRESS:")
    print(f"  Current Step: {step:,}")
    if history.get('steps'):
        print(f"  Total Logged Steps: {len(history['steps']):,}")
    
    # Performance
    print("\nPERFORMANCE:")
    if history.get('train_acc'):
        print(f"  Training Accuracy: {history['train_acc'][-1]:.2f}%")
    if history.get('val_acc'):
        print(f"  Validation Accuracy: {history['val_acc'][-1]:.2f}%")
    if history.get('train_loss'):
        print(f"  Training Loss: {history['train_loss'][-1]:.4f}")
    if history.get('val_loss'):
        print(f"  Validation Loss: {history['val_loss'][-1]:.4f}")
    
    # Grokking status
    print("\nGROKKING STATUS:")
    if history.get('grokking_detected'):
        print(f"    Grokking detected at step: {history['grokking_step']:,}")
        print(f"  Train threshold reached at: {history['train_threshold_step']:,}")
        if history['grokking_step'] and history['train_threshold_step']:
            delay = history['grokking_step'] - history['train_threshold_step']
            print(f"  Grokking delay: {delay:,} steps")
    else:
        print(f"  ✗ Grokking not yet detected")
        if history.get('train_threshold_step'):
            print(f"  (Training threshold reached at step {history['train_threshold_step']:,})")
    
    # Configuration
    print("\nCONFIGURATION:")
    if config.get('multi_task'):
        print(f"  Multi-Task: Enabled")
        if config.get('task_mix'):
            print(f"  Task Mix: {config['task_mix']}")
    else:
        print(f"  Multi-Task: Disabled (division only)")
    
    print(f"  Learning Rate: {config.get('lr', 'Unknown')}")
    print(f"  Weight Decay: {config.get('weight_decay', 'Unknown')}")
    print(f"  Batch Size: {config.get('batch_size', 'Unknown')}")
    print(f"  Model Params: {config.get('model_params', 'Unknown'):,}" if config.get('model_params') else "")
    
    # File info
    print("\nFILE INFO:")
    file_size = os.path.getsize(checkpoint_path) / (1024 * 1024)
    mod_time = datetime.fromtimestamp(os.path.getmtime(checkpoint_path))
    print(f"  File Size: {file_size:.2f} MB")
    print(f"  Last Modified: {mod_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    print(f"\n{'='*80}\n")


def compare_checkpoints(checkpoint_paths):
    """Compare multiple checkpoints side by side."""
    if len(checkpoint_paths) < 2:
        print("Please provide at least 2 checkpoints to compare")
        return
    
    print(f"\n{'='*80}")
    print(f"Checkpoint Comparison")
    print(f"{'='*80}\n")
    
    checkpoints = []
    for path in checkpoint_paths:
        if not os.path.exists(path):
            print(f"Warning: {path} not found, skipping...")
            continue
        
        try:
            cp = torch.load(path, map_location='cpu')
            checkpoints.append({
                'path': os.path.basename(path),
                'checkpoint': cp
            })
        except Exception as e:
            print(f"Error loading {path}: {e}")
    
    if len(checkpoints) < 2:
        print("Not enough valid checkpoints to compare")
        return
    
    # Print comparison table
    print(f"{'Metric':<30}", end='')
    for cp in checkpoints:
        print(f"{cp['path']:<25}", end='')
    print()
    print('-' * (30 + 25 * len(checkpoints)))
    
    # Step
    print(f"{'Step':<30}", end='')
    for cp in checkpoints:
        step = cp['checkpoint'].get('step', 'N/A')
        print(f"{step:>20,}     ", end='')
    print()
    
    # Train Accuracy
    print(f"{'Train Accuracy':<30}", end='')
    for cp in checkpoints:
        acc = cp['checkpoint'].get('history', {}).get('train_acc', [])
        acc_str = f"{acc[-1]:.2f}%" if acc else "N/A"
        print(f"{acc_str:>20}     ", end='')
    print()
    
    # Val Accuracy
    print(f"{'Val Accuracy':<30}", end='')
    for cp in checkpoints:
        acc = cp['checkpoint'].get('history', {}).get('val_acc', [])
        acc_str = f"{acc[-1]:.2f}%" if acc else "N/A"
        print(f"{acc_str:>20}     ", end='')
    print()
    
    # Grokking
    print(f"{'Grokking Detected':<30}", end='')
    for cp in checkpoints:
        grokking = cp['checkpoint'].get('history', {}).get('grokking_detected', False)
        grok_str = "✓ Yes" if grokking else "✗ No"
        print(f"{grok_str:>20}     ", end='')
    print()
    
    # Grokking Step
    print(f"{'Grokking Step':<30}", end='')
    for cp in checkpoints:
        grok_step = cp['checkpoint'].get('history', {}).get('grokking_step')
        step_str = f"{grok_step:,}" if grok_step else "N/A"
        print(f"{step_str:>20}     ", end='')
    print()
    
    # Multi-task
    print(f"{'Multi-Task':<30}", end='')
    for cp in checkpoints:
        mt = cp['checkpoint'].get('config', {}).get('multi_task', False)
        mt_str = "Yes" if mt else "No"
        print(f"{mt_str:>20}     ", end='')
    print()
    
    print(f"\n{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description='Manage training checkpoints')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # List command
    list_parser = subparsers.add_parser('list', help='List all checkpoints')
    list_parser.add_argument('--dir', type=str, default='checkpoints',
                           help='Checkpoint directory (default: checkpoints)')
    
    # Inspect command
    inspect_parser = subparsers.add_parser('inspect', help='Inspect a checkpoint')
    inspect_parser.add_argument('checkpoint', type=str,
                              help='Path to checkpoint file')
    
    # Compare command
    compare_parser = subparsers.add_parser('compare', help='Compare checkpoints')
    compare_parser.add_argument('checkpoints', type=str, nargs='+',
                              help='Paths to checkpoint files to compare')
    
    args = parser.parse_args()
    
    if args.command == 'list':
        list_checkpoints(args.dir)
    elif args.command == 'inspect':
        inspect_checkpoint(args.checkpoint)
    elif args.command == 'compare':
        compare_checkpoints(args.checkpoints)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
