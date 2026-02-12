import matplotlib.pyplot as plt

def plot_grokking(history, save_path='grokking_result.png'):
    """
    Plot training curves showing grokking for multiple tasks.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    steps = history['steps']
    
    # --- Accuracy Plot ---
    ax1.plot(steps, history['train_acc'], 'k--', label='Train (Combined)', alpha=0.7)
    
    # Plot validation for each task
    val_stats = history['val_stats']
    colors = {'div': 'red', 'add': 'blue', 'sub': 'green'}
    
    for task, stats in val_stats.items():
        color = colors.get(task, None) # Auto-color if task not in map
        acc = stats['acc']
        ax1.plot(steps, acc, label=f'Val ({task.upper()})', color=color, linewidth=2)

    ax1.set_xlabel('Steps')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('Grokking: Accuracy per Task')
    ax1.set_xscale('log')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # --- Loss Plot ---
    ax2.plot(steps, history['train_loss'], 'k--', label='Train Loss', alpha=0.7)
    
    for task, stats in val_stats.items():
        color = colors.get(task, None)
        loss = stats['loss']
        ax2.plot(steps, loss, label=f'Val Loss ({task.upper()})', color=color, linewidth=1.5)
        
    ax2.set_xlabel('Steps')
    ax2.set_ylabel('Loss')
    ax2.set_title('Loss Curves')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Plot saved to {save_path}")