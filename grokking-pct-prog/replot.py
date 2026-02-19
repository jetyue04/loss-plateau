'''
replot.py loads an existing checkpoint and regenerates the grokking plot
with training progress (%) on the x-axis instead of raw step counts.

Usage:
    python replot.py --checkpoint_dir checkpoints/baseline --save_path replot.png
    python replot.py --checkpoint_dir checkpoints/baseline --save_path replot.png --num_steps 400000
    python replot.py --checkpoint_dir checkpoints/baseline --save_path replot.png --num_steps 400000 --grok_steps div=334000
    python replot.py --checkpoint_dir checkpoints/baseline --save_path replot.png --num_steps 400000 --grok_steps div=334000 add=120000
'''

import argparse
import torch
import matplotlib.pyplot as plt
import os


def replot_from_checkpoint(checkpoint_dir, save_path, num_steps_override=None, grok_steps_override=None):
    checkpoint_path = os.path.join(checkpoint_dir, 'checkpoint.pt')

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"No checkpoint found at: {checkpoint_path}")

    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    history = checkpoint['history']
    steps = history['steps']

    if not steps:
        raise ValueError("Checkpoint history contains no recorded steps.")

    # Determine total steps for percentage calculation.
    # Priority: command-line override > stored in history > last recorded step
    if num_steps_override is not None:
        num_steps = num_steps_override
        print(f"Using num_steps override: {num_steps:,}")
    elif 'num_steps' in history:
        num_steps = history['num_steps']
        print(f"Using num_steps from checkpoint: {num_steps:,}")
    else:
        num_steps = checkpoint.get('step', steps[-1])
        print(f"Inferring num_steps from checkpoint step: {num_steps:,}")

    # Compute percentage for each recorded step
    x_vals = [s / num_steps * 100 for s in steps]

    # Recover grokking steps from checkpoint
    grok_steps = history.get('grok_steps', {})
    grok_pcts = history.get('grok_pcts', {})

    # Apply any manual overrides for tasks where grokking wasn't auto-detected
    if grok_steps_override:
        for task, gs in grok_steps_override.items():
            if grok_steps.get(task) is None:
                print(f"Applying manual grok step override for {task.upper()}: step {gs:,}")
            else:
                print(f"Overriding stored grok step for {task.upper()}: {grok_steps[task]:,} -> {gs:,}")
            grok_steps[task] = gs
            grok_pcts[task] = gs / num_steps * 100

    # Compute any remaining missing grok_pcts from grok_steps
    for task, gs in grok_steps.items():
        if grok_pcts.get(task) is None and gs is not None:
            grok_pcts[task] = gs / num_steps * 100

    val_stats = history['val_stats']
    colors = {'div': 'red', 'add': 'blue', 'sub': 'green', 'mult': 'purple'}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # ===== Left Panel: Accuracy =====
    ax1.plot(x_vals, history['train_acc'], 'k--', label='Train (Combined)', alpha=0.3)

    for task, stats in val_stats.items():
        color = colors.get(task, 'orange')
        ax1.plot(x_vals, stats['acc'], label=f'Val ({task.upper()})', color=color, linewidth=2)

        grok_pct = grok_pcts.get(task)
        if grok_pct is not None:
            ax1.axvline(x=grok_pct, color=color, linestyle=':', alpha=0.8)
            ax1.text(grok_pct, 50, f" {grok_pct:.1f}% (step {grok_steps.get(task):,})",
                     rotation=90, verticalalignment='center', color=color, fontweight='bold')

    ax1.set_xlabel('Training Progress (%)')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('Grokking: Accuracy per Task')
    ax1.set_xlim(0, 100)
    ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda val, _: f'{val:.4g}%'))
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)

    # ===== Right Panel: Loss =====
    ax2.plot(x_vals, history['train_loss'], 'k--', label='Train Loss', alpha=0.3)

    for task, stats in val_stats.items():
        color = colors.get(task, 'orange')
        ax2.plot(x_vals, stats['loss'], label=f'Val Loss ({task.upper()})', color=color, linewidth=1.5)

    ax2.set_xlabel('Training Progress (%)')
    ax2.set_ylabel('Loss')
    ax2.set_title('Loss Curves')
    ax2.set_xlim(0, 100)
    ax2.set_yscale('log')
    ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda val, _: f'{val:.4g}%'))
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)

    plt.suptitle(f'Total Training Steps: {num_steps:,}', y=1.02, fontsize=12, color='gray')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {save_path}")

    # Print grokking summary
    print("\n--- Grokking Summary ---")
    for task, gs in grok_steps.items():
        if gs is not None:
            pct = grok_pcts.get(task, gs / num_steps * 100)
            print(f"  {task.upper()}: Grokked at step {gs:,} ({pct:.1f}% of training)")
        else:
            print(f"  {task.upper()}: Did not grok")


def parse_grok_steps(values):
    """Parse 'task=step' pairs from command line, e.g. div=334000 add=120000."""
    result = {}
    for v in values:
        try:
            task, step = v.split('=')
            result[task.strip()] = int(step.strip())
        except ValueError:
            raise argparse.ArgumentTypeError(
                f"Invalid format '{v}'. Expected task=step, e.g. div=334000"
            )
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Replot grokking results from a checkpoint using training progress % on x-axis'
    )
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help='Directory containing checkpoint.pt (default: checkpoints)')
    parser.add_argument('--save_path', type=str, default='replot.png',
                        help='Path to save the output plot (default: replot.png)')
    parser.add_argument('--num_steps', type=int, default=None,
                        help='Total training steps used (optional -- inferred from checkpoint if not provided)')
    parser.add_argument('--grok_steps', nargs='+', default=None, metavar='TASK=STEP',
                        help='Manually specify grokking steps for tasks not auto-detected, '
                             'e.g. --grok_steps div=334000 add=120000')
    args = parser.parse_args()

    grok_steps_override = parse_grok_steps(args.grok_steps) if args.grok_steps else None

    replot_from_checkpoint(
        checkpoint_dir=args.checkpoint_dir,
        save_path=args.save_path,
        num_steps_override=args.num_steps,
        grok_steps_override=grok_steps_override,
    )
