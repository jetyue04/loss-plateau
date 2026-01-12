import matplotlib.pyplot as plt


def plot_grokking(history, save_path='grokking_result.png'):
    """
    Plot training curves showing the grokking phenomenon.
    
    Args:
        history: Dictionary containing training history
        save_path: Path to save the figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    ax1.plot(history['steps'], history['train_acc'], 'r-', label='Train Accuracy', linewidth=2)
    ax1.plot(history['steps'], history['val_acc'], 'g-', label='Validation Accuracy', linewidth=2)
    ax1.set_xlabel('Optimization Steps', fontsize=12)
    ax1.set_ylabel('Accuracy (%)', fontsize=12)
    ax1.set_xscale('log')
    ax1.set_title('Grokking: Generalization Long After Overfitting', fontsize=14)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 105])
    
    ax2.plot(history['steps'], history['train_loss'], 'r-', label='Train Loss', linewidth=2)
    ax2.plot(history['steps'], history['val_loss'], 'g-', label='Validation Loss', linewidth=2)
    ax2.set_xlabel('Optimization Steps', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_title('Loss Curves (Double Descent)', fontsize=14)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"\nFinal Results:")
    print(f"Train Accuracy: {history['train_acc'][-1]:.2f}%")
    print(f"Val Accuracy: {history['val_acc'][-1]:.2f}%")
    print(f"Plot saved to {save_path}")