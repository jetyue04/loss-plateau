'''
utils.py contains visualization utilities for grokking experiments.

This module provides functions to create plots showing the grokking 
phenomenon, including per-task accuracy and loss curves with
automatic grokking detection markers.
'''

import matplotlib.pyplot as plt


def plot_grokking(history, save_path='grokking_result.png'):
    """
    Create a comprehensive visualization of the grokking phenomenon.
    
    Generates a two-panel figure showing:
    - Left panel: Accuracy curves over training steps (log scale)
      - Combined training accuracy (dashed black line)
      - Per-task validation accuracy (colored solid lines)
      - Vertical lines marking grokking points (when validation exceeds 95%)
    - Right panel: Loss curves over training steps (log-log scale)
      - Combined training loss (dashed black line)
      - Per-task validation loss (colored solid lines)
    
    Each task is assigned a distinct color:
    - Division: red
    - Addition: blue
    - Subtraction: green
    - Multiplication: purple
    
    :param history: Dictionary containing training history with keys:
        - 'steps': List of step numbers
        - 'train_acc': Training accuracy at each step
        - 'train_loss': Training loss at each step
        - 'val_stats': Dict mapping task names to {'acc': [...], 'loss': [...]}
        - 'grok_steps': Dict mapping task names to step where they grokked
    :param save_path: Path where the plot image will be saved
    """
    # Create figure with two side-by-side subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    steps = history['steps']
    grok_steps = history.get('grok_steps', {})
    
    # ===== Left Panel: Accuracy Curves =====
    # Plot combined training accuracy
    ax1.plot(steps, history['train_acc'], 'k--', label='Train (Combined)', alpha=0.3)
    
    val_stats = history['val_stats']
    colors = {'div': 'red', 'add': 'blue', 'sub': 'green', 'mult': 'purple'}
    
    # Plot validation accuracy for each task
    for task, stats in val_stats.items():
        color = colors.get(task, 'orange')
        acc = stats['acc']
        ax1.plot(steps, acc, label=f'Val ({task.upper()})', color=color, linewidth=2)
        
        # Mark grokking point with vertical line
        grok_step = grok_steps.get(task)
        if grok_step is not None:
            ax1.axvline(x=grok_step, color=color, linestyle=':', alpha=0.8)
            # Add text label showing grokking step in thousands
            ax1.text(grok_step, 50, f" {grok_step//1000}k", 
                     rotation=90, verticalalignment='center', color=color, fontweight='bold')

    ax1.set_xlabel('Steps')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('Grokking: Accuracy per Task')
    ax1.set_xscale('log')  # Log scale emphasizes the sudden transition
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # ===== Right Panel: Loss Curves =====
    # Plot combined training loss
    ax2.plot(steps, history['train_loss'], 'k--', label='Train Loss', alpha=0.3)
    
    # Plot validation loss for each task
    for task, stats in val_stats.items():
        color = colors.get(task, 'orange')
        loss = stats['loss']
        ax2.plot(steps, loss, label=f'Val Loss ({task.upper()})', color=color, linewidth=1.5)
        
    ax2.set_xlabel('Steps')
    ax2.set_ylabel('Loss')
    ax2.set_title('Loss Curves')
    ax2.set_xscale('log')   # Log scale for steps
    ax2.set_yscale('log')   # Log scale for loss (shows double descent)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Finalize and save
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Plot saved to {save_path}")
