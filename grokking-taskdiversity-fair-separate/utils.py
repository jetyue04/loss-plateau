import matplotlib.pyplot as plt

def plot_grokking(history, save_path='grokking_result.png'):
    """
    Plot training curves showing grokking for multiple tasks.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    steps = history['steps']
    grok_steps = history.get('grok_steps', {})
    
    # --- Accuracy Plot ---
    ax1.plot(steps, history['train_acc'], 'k--', label='Train (Combined)', alpha=0.3)
    
    val_stats = history['val_stats']
    colors = {'div': 'red', 'add': 'blue', 'sub': 'green'}
    
    # Plot curves and add vertical lines
    for task, stats in val_stats.items():
        color = colors.get(task, 'orange') # default color
        acc = stats['acc']
        
        # Plot the curve
        ax1.plot(steps, acc, label=f'Val ({task.upper()})', color=color, linewidth=2)
        
        # Plot the Grokking Line
        grok_step = grok_steps.get(task)
        if grok_step is not None:
            ax1.axvline(x=grok_step, color=color, linestyle=':', alpha=0.8)
            # Add text label slightly offset
            ax1.text(grok_step, 50, f" {grok_step//1000}k", 
                     rotation=90, verticalalignment='center', color=color, fontweight='bold')

    ax1.set_xlabel('Steps')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('Grokking: Accuracy per Task')
    ax1.set_xscale('log')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # --- Loss Plot ---
    ax2.plot(steps, history['train_loss'], 'k--', label='Train Loss', alpha=0.3)
    
    for task, stats in val_stats.items():
        color = colors.get(task, 'orange')
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