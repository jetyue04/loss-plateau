import matplotlib.pyplot as plt


def plot_grokking(history, save_path='grokking_result.png'):
    """
    Plot training curves showing the grokking phenomenon with annotations.
    
    Args:
        history: Dictionary containing training history and grokking info
        save_path: Path to save the figure
    """
    # Check if we have per-task data
    has_per_task = 'task_val_acc' in history and len(history.get('task_val_acc', {})) > 0
    
    if has_per_task:
        # Create larger figure with per-task plots
        fig = plt.figure(figsize=(20, 10))
        gs = fig.add_gridspec(3, 2, height_ratios=[4, 4, 1], width_ratios=[1, 1], 
                              hspace=0.3, wspace=0.3)
        ax1 = fig.add_subplot(gs[0, 0])  # Overall accuracy
        ax2 = fig.add_subplot(gs[0, 1])  # Loss curves
        ax3 = fig.add_subplot(gs[1, :])  # Per-task validation accuracy
        ax4 = fig.add_subplot(gs[2, :])  # Config text
    else:
        # Original layout
        fig = plt.figure(figsize=(16, 6))
        gs = fig.add_gridspec(2, 2, height_ratios=[4, 1], width_ratios=[1, 1], 
                              hspace=0.3, wspace=0.3)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax4 = fig.add_subplot(gs[1, :])
    
    # Plot overall accuracy curves
    ax1.plot(history['steps'], history['train_acc'], 'r-', label='Train Accuracy', linewidth=2)
    ax1.plot(history['steps'], history['val_acc'], 'g-', label='Validation Accuracy (Overall)', linewidth=2)
    
    # Add grokking annotations if detected
    if history.get('grokking_detected', False):
        grok_step = history['grokking_step']
        train_step = history['train_threshold_step']
        
        # Add vertical line at grokking point
        ax1.axvline(x=grok_step, color='blue', linestyle='--', linewidth=2, alpha=0.7, 
                   label=f'Grokking at step {grok_step:,}')
        
        # Add vertical line at training threshold
        ax1.axvline(x=train_step, color='orange', linestyle=':', linewidth=1.5, alpha=0.6,
                   label=f'Train 95% at step {train_step:,}')
        
        # Add gap annotation
        gap = grok_step - train_step
        ax1.text(0.05, 0.95, f'Grokking Delay: {gap:,} steps', 
                transform=ax1.transAxes, fontsize=11,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    ax1.set_xlabel('Optimization Steps', fontsize=12)
    ax1.set_ylabel('Accuracy (%)', fontsize=12)
    ax1.set_xscale('log')
    ax1.set_title('Grokking: Generalization Long After Overfitting', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=9, loc='lower right')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 105])
    
    # Plot loss curves
    ax2.plot(history['steps'], history['train_loss'], 'r-', label='Train Loss', linewidth=2)
    ax2.plot(history['steps'], history['val_loss'], 'g-', label='Validation Loss', linewidth=2)
    
    # Add grokking line to loss plot as well
    if history.get('grokking_detected', False):
        grok_step = history['grokking_step']
        ax2.axvline(x=grok_step, color='blue', linestyle='--', linewidth=2, alpha=0.7,
                   label=f'Grokking at step {grok_step:,}')
    
    ax2.set_xlabel('Optimization Steps', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_title('Loss Curves (Double Descent)', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # Plot per-task validation accuracy if available
    if has_per_task:
        colors = {'division': 'blue', 'addition': 'green', 'subtraction': 'purple'}
        
        for task_name, accuracies in history['task_val_acc'].items():
            color = colors.get(task_name, 'gray')
            ax3.plot(history['steps'], accuracies, '-', 
                    color=color, label=f'{task_name.capitalize()} Val Acc', linewidth=2)
        
        # Also plot overall for comparison
        ax3.plot(history['steps'], history['val_acc'], 'k--', 
                label='Overall Val Acc', linewidth=2, alpha=0.5)
        
        # Add grokking line
        if history.get('grokking_detected', False):
            ax3.axvline(x=history['grokking_step'], color='red', 
                       linestyle='--', linewidth=2, alpha=0.7,
                       label=f'Grokking at step {history["grokking_step"]:,}')
        
        ax3.set_xlabel('Optimization Steps', fontsize=12)
        ax3.set_ylabel('Validation Accuracy (%)', fontsize=12)
        ax3.set_xscale('log')
        ax3.set_title('Per-Task Validation Accuracy', fontsize=14, fontweight='bold')
        ax3.legend(fontsize=10, loc='lower right')
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim([0, 105])
    
    # Add configuration information panel
    ax4.axis('off')
    
    config = history.get('config', {})
    
    # Build info text
    info_lines = []
    info_lines.append("EXPERIMENT CONFIGURATION & RESULTS")
    info_lines.append("=" * 80)
    
    # Model info
    if 'model_params' in config:
        info_lines.append(f"Model Parameters: {config['model_params']:,}")
    if 'd_model' in config:
        info_lines.append(f"Model Dimensions: d_model={config['d_model']}, n_heads={config.get('nhead', 'N/A')}, n_layers={config.get('num_layers', 'N/A')}")
    
    # Optimizer info
    if 'optimizer' in config:
        opt_str = f"Optimizer: {config['optimizer']}"
        if 'betas' in config:
            opt_str += f" (betas={config['betas']})"
        info_lines.append(opt_str)
    if 'lr' in config:
        info_lines.append(f"Learning Rate: {config['lr']:.0e}")
    if 'weight_decay' in config:
        info_lines.append(f"Weight Decay: {config['weight_decay']:.0e}")
    if 'batch_size' in config:
        info_lines.append(f"Batch Size: {config['batch_size']}")
    if 'num_steps' in config:
        info_lines.append(f"Total Training Steps: {config['num_steps']:,}")
    
    # Data info
    if 'p' in config:
        info_lines.append(f"Modulus (p): {config['p']}")
    if 'train_fraction' in config:
        info_lines.append(f"Train Fraction: {config['train_fraction']:.1%}")
    if 'train_size' in config and 'val_size' in config:
        info_lines.append(f"Dataset: {config['train_size']:,} train / {config['val_size']:,} val examples")
    
    # Multi-task info
    if config.get('multi_task', False):
        info_lines.append("Multi-Task Training: ENABLED")
        if 'task_mix' in config:
            task_mix_str = ", ".join([f"{k}: {v:.1%}" for k, v in config['task_mix'].items()])
            info_lines.append(f"  Task Mix: {task_mix_str}")
        if 'fair_comparison' in config:
            info_lines.append(f"  Fair Comparison: {'Yes' if config['fair_comparison'] else 'No'}")
    else:
        info_lines.append("Multi-Task Training: DISABLED (division only)")
    
    # Initialization info
    if 'init_type' in config and config['init_type'] != 'default':
        info_lines.append(f"Initialization: {config['init_type']}")
        if config['init_type'] == 'low_rank' and 'rank_ratio' in config:
            info_lines.append(f"  Rank Ratio: {config['rank_ratio']}")
        elif config['init_type'] == 'sparse' and 'sparsity' in config:
            info_lines.append(f"  Sparsity: {config['sparsity']*100:.0f}%")
        elif config['init_type'] == 'small' and 'init_scale' in config:
            info_lines.append(f"  Init Scale: {config['init_scale']}")
    
    # Results
    info_lines.append("")
    info_lines.append("FINAL RESULTS:")
    info_lines.append(f"  Final Train Accuracy: {history['train_acc'][-1]:.2f}%")
    info_lines.append(f"  Final Val Accuracy: {history['val_acc'][-1]:.2f}%")
    
    # Per-task final accuracies
    if has_per_task:
        info_lines.append("  Per-Task Final Val Accuracies:")
        for task_name in history['task_val_acc']:
            final_acc = history['task_val_acc'][task_name][-1]
            info_lines.append(f"    {task_name.capitalize()}: {final_acc:.2f}%")
    
    if history.get('grokking_detected', False):
        info_lines.append(f"    GROKKING DETECTED:")
        info_lines.append(f"    - Train threshold ({history.get('train_threshold', 95):.0f}%) reached: step {history['train_threshold_step']:,}")
        info_lines.append(f"    - Val threshold ({history.get('val_threshold', 95):.0f}%) reached: step {history['grokking_step']:,}")
        info_lines.append(f"    - Grokking delay: {history['grokking_step'] - history['train_threshold_step']:,} steps")
    else:
        info_lines.append(f"    Grokking not detected within training period")
    
    # Display text
    info_text = '\n'.join(info_lines)
    ax4.text(0.05, 0.95, info_text, transform=ax4.transAxes, 
            fontsize=9, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    # Print final results to console
    print(f"\n{'='*60}")
    print(f"FINAL RESULTS:")
    print(f"{'='*60}")
    print(f"Train Accuracy: {history['train_acc'][-1]:.2f}%")
    print(f"Val Accuracy: {history['val_acc'][-1]:.2f}%")
    
    if has_per_task:
        print(f"\nPer-Task Final Val Accuracies:")
        for task_name in history['task_val_acc']:
            final_acc = history['task_val_acc'][task_name][-1]
            print(f"  {task_name.capitalize()}: {final_acc:.2f}%")
    
    if history.get('grokking_detected', False):
        print(f"\n  Grokking detected:")
        print(f"  - Training reached {history.get('train_threshold', 95):.0f}% at step: {history['train_threshold_step']:,}")
        print(f"  - Validation reached {history.get('val_threshold', 95):.0f}% at step: {history['grokking_step']:,}")
        print(f"  - Grokking delay: {history['grokking_step'] - history['train_threshold_step']:,} steps")
    else:
        print(f"\n  Grokking not detected within training period")
    
    print(f"\nPlot saved to {save_path}")
    print(f"{'='*60}")
    """
    Plot training curves showing the grokking phenomenon with annotations.
    
    Args:
        history: Dictionary containing training history and grokking info
        save_path: Path to save the figure
    """
    fig = plt.figure(figsize=(16, 6))
    
    # Create grid with space for text box
    gs = fig.add_gridspec(2, 2, height_ratios=[4, 1], width_ratios=[1, 1], 
                          hspace=0.3, wspace=0.3)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, :])
    
    # Plot accuracy curves
    ax1.plot(history['steps'], history['train_acc'], 'r-', label='Train Accuracy', linewidth=2)
    ax1.plot(history['steps'], history['val_acc'], 'g-', label='Validation Accuracy', linewidth=2)
    
    # Add grokking annotations if detected
    if history.get('grokking_detected', False):
        grok_step = history['grokking_step']
        train_step = history['train_threshold_step']
        
        # Add vertical line at grokking point
        ax1.axvline(x=grok_step, color='blue', linestyle='--', linewidth=2, alpha=0.7, 
                   label=f'Grokking at step {grok_step:,}')
        
        # Add vertical line at training threshold
        ax1.axvline(x=train_step, color='orange', linestyle=':', linewidth=1.5, alpha=0.6,
                   label=f'Train 95% at step {train_step:,}')
        
        # Add gap annotation
        gap = grok_step - train_step
        ax1.text(0.05, 0.95, f'Grokking Delay: {gap:,} steps', 
                transform=ax1.transAxes, fontsize=11,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    ax1.set_xlabel('Optimization Steps', fontsize=12)
    ax1.set_ylabel('Accuracy (%)', fontsize=12)
    ax1.set_xscale('log')
    ax1.set_title('Grokking: Generalization Long After Overfitting', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=9, loc='lower right')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 105])
    
    # Plot loss curves
    ax2.plot(history['steps'], history['train_loss'], 'r-', label='Train Loss', linewidth=2)
    ax2.plot(history['steps'], history['val_loss'], 'g-', label='Validation Loss', linewidth=2)
    
    # Add grokking line to loss plot as well
    if history.get('grokking_detected', False):
        grok_step = history['grokking_step']
        ax2.axvline(x=grok_step, color='blue', linestyle='--', linewidth=2, alpha=0.7,
                   label=f'Grokking at step {grok_step:,}')
    
    ax2.set_xlabel('Optimization Steps', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_title('Loss Curves (Double Descent)', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # Add configuration information panel
    ax3.axis('off')
    
    config = history.get('config', {})
    
    # Build info text
    info_lines = []
    info_lines.append("EXPERIMENT CONFIGURATION & RESULTS")
    info_lines.append("=" * 80)
    
    # Model info
    if 'model_params' in config:
        info_lines.append(f"Model Parameters: {config['model_params']:,}")
    if 'd_model' in config:
        info_lines.append(f"Model Dimensions: d_model={config['d_model']}, n_heads={config.get('nhead', 'N/A')}, n_layers={config.get('num_layers', 'N/A')}")
    
    # Optimizer info
    if 'optimizer' in config:
        opt_str = f"Optimizer: {config['optimizer']}"
        if 'betas' in config:
            opt_str += f" (betas={config['betas']})"
        info_lines.append(opt_str)
    if 'lr' in config:
        info_lines.append(f"Learning Rate: {config['lr']:.0e}")
    if 'weight_decay' in config:
        info_lines.append(f"Weight Decay: {config['weight_decay']:.0e}")
    if 'batch_size' in config:
        info_lines.append(f"Batch Size: {config['batch_size']}")
    if 'num_steps' in config:
        info_lines.append(f"Total Training Steps: {config['num_steps']:,}")
    
    # Data info
    if 'p' in config:
        info_lines.append(f"Modulus (p): {config['p']}")
    if 'train_fraction' in config:
        info_lines.append(f"Train Fraction: {config['train_fraction']:.1%}")
    if 'train_size' in config and 'val_size' in config:
        info_lines.append(f"Dataset: {config['train_size']:,} train / {config['val_size']:,} val examples")
    
    # Multi-task info
    if config.get('multi_task', False):
        info_lines.append("Multi-Task Training: ENABLED")
        if 'task_mix' in config:
            task_mix_str = ", ".join([f"{k}: {v:.1%}" for k, v in config['task_mix'].items()])
            info_lines.append(f"  Task Mix: {task_mix_str}")
    else:
        info_lines.append("Multi-Task Training: DISABLED (division only)")
    
    # Results
    info_lines.append("")
    info_lines.append("FINAL RESULTS:")
    info_lines.append(f"  Final Train Accuracy: {history['train_acc'][-1]:.2f}%")
    info_lines.append(f"  Final Val Accuracy: {history['val_acc'][-1]:.2f}%")
    
    if history.get('grokking_detected', False):
        info_lines.append(f"    GROKKING DETECTED:")
        info_lines.append(f"    - Train threshold ({history.get('train_threshold', 95):.0f}%) reached: step {history['train_threshold_step']:,}")
        info_lines.append(f"    - Val threshold ({history.get('val_threshold', 95):.0f}%) reached: step {history['grokking_step']:,}")
        info_lines.append(f"    - Grokking delay: {history['grokking_step'] - history['train_threshold_step']:,} steps")
    else:
        info_lines.append(f"    Grokking not detected within training period")
    
    # Display text
    info_text = '\n'.join(info_lines)
    ax3.text(0.05, 0.95, info_text, transform=ax3.transAxes, 
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    # Print final results to console
    print(f"\n{'='*60}")
    print(f"FINAL RESULTS:")
    print(f"{'='*60}")
    print(f"Train Accuracy: {history['train_acc'][-1]:.2f}%")
    print(f"Val Accuracy: {history['val_acc'][-1]:.2f}%")
    
    if history.get('grokking_detected', False):
        print(f"\n  Grokking detected:")
        print(f"  - Training reached {history.get('train_threshold', 95):.0f}% at step: {history['train_threshold_step']:,}")
        print(f"  - Validation reached {history.get('val_threshold', 95):.0f}% at step: {history['grokking_step']:,}")
        print(f"  - Grokking delay: {history['grokking_step'] - history['train_threshold_step']:,} steps")
    else:
        print(f"\n  Grokking not detected within training period")
    
    print(f"\nPlot saved to {save_path}")
    print(f"{'='*60}")