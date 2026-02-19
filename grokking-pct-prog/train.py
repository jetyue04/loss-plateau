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
    # Use progress_pct if available (new format), otherwise compute from steps
    num_steps = history.get('num_steps', steps[-1] if steps else 1)
    x_vals = history.get('progress_pct', [s / num_steps * 100 for s in steps])
    
    grok_steps = history.get('grok_steps', {})
    grok_pcts = history.get('grok_pcts', {})
    
    # ===== Left Panel: Accuracy Curves =====
    # Plot combined training accuracy
    ax1.plot(x_vals, history['train_acc'], 'k--', label='Train (Combined)', alpha=0.3)
    
    val_stats = history['val_stats']
    colors = {'div': 'red', 'add': 'blue', 'sub': 'green', 'mult': 'purple'}
    
    # Plot validation accuracy for each task
    for task, stats in val_stats.items():
        color = colors.get(task, 'orange')
        acc = stats['acc']
        ax1.plot(x_vals, acc, label=f'Val ({task.upper()})', color=color, linewidth=2)
        
        # Mark grokking point with vertical line
        grok_pct = grok_pcts.get(task)
        if grok_pct is None:
            # Fall back to computing from step if pct not stored
            grok_step = grok_steps.get(task)
            if grok_step is not None:
                grok_pct = grok_step / num_steps * 100
        
        if grok_pct is not None:
            ax1.axvline(x=grok_pct, color=color, linestyle=':', alpha=0.8)
            ax1.text(grok_pct, 50, f" {grok_pct:.1f}%", 
                     rotation=90, verticalalignment='center', color=color, fontweight='bold')

    ax1.set_xlabel('Training Progress (%)')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('Grokking: Accuracy per Task')
    ax1.set_xlim(left=max(x_vals[0] if x_vals else 0, 0.001))
    ax1.set_xscale('log')  # Log scale emphasizes the sudden transition
    ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda val, _: f'{val:.4g}%'))
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # ===== Right Panel: Loss Curves =====
    # Plot combined training loss
    ax2.plot(x_vals, history['train_loss'], 'k--', label='Train Loss', alpha=0.3)
    
    # Plot validation loss for each task
    for task, stats in val_stats.items():
        color = colors.get(task, 'orange')
        loss = stats['loss']
        ax2.plot(x_vals, loss, label=f'Val Loss ({task.upper()})', color=color, linewidth=1.5)
        
    ax2.set_xlabel('Training Progress (%)')
    ax2.set_ylabel('Loss')
    ax2.set_title('Loss Curves')
    ax2.set_xlim(left=max(x_vals[0] if x_vals else 0, 0.001))
    ax2.set_xscale('log')   # Log scale for x
    ax2.set_yscale('log')   # Log scale for loss (shows double descent)
    ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda val, _: f'{val:.4g}%'))
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Finalize and save
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Plot saved to {save_path}")
