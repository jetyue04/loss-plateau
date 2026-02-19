'''
replot_sub.py - Replot script compatible with the original checkpoint format
(no progress_pct, no grok_pcts, no num_steps stored in history).

Usage:
    python replot_sub.py --checkpoint_path checkpoints/sub/checkpoint.pt --save_path sub_replot.png --num_steps 400000
    python replot_sub.py --checkpoint_path checkpoints/sub/checkpoint.pt --save_path sub_replot.png --num_steps 100000
'''

import argparse
import torch
import matplotlib.pyplot as plt
import os


def replot_from_checkpoint(checkpoint_path, save_path, num_steps_override=None, grok_steps_override=None):
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

    print(f"Steps recorded: {steps[0]:,} to {steps[-1]:,} ({len(steps)} entries)")

    # Compute x-axis as percentage of total training
    x_vals = [s / num_steps * 100 for s in steps]

    # Recover grokking steps from history (original format stores these)
    grok_steps = dict(history.get('grok_steps', {}))

    # Apply any manual overrides
    if grok_steps_override:
        for task, gs in grok_steps_override.items():
            action = "Overriding" if grok_steps.get(task) is not None else "Applying manual"
            print(f"{action} grok step for {task.upper()}: step {gs:,}")
            grok_steps[task] = gs

    # Compute grok percentages from steps
    grok_pcts = {
        task: (gs / num_steps * 100 if gs is not None else None)
        for task, gs in grok_steps.items()
    }

    # Log what we found
    for task, gs in grok_steps.items():
        if gs is not None:
            print(f"  {task.upper()} grokked at step {gs:,} ({grok_pcts[task]:.1f}%)")
        else:
            print(f"  {task.upper()} did not grok")

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
            ax1.text(grok_pct, 50, f" {grok_pct:.1f}% (step {grok_steps[task]:,})",
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
    print(f"\nPlot saved to: {save_path}")


def parse_grok_steps(values):
    result = {}
    for v in values:
        try:
            task, step = v.split('=')
            result[task.strip()] = int(step.strip())
        except ValueError:
            raise argparse.ArgumentTypeError(
                f"Invalid format '{v}'. Expected task=step, e.g. sub=75050"
            )
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Replot grokking checkpoint (original format) with training progress % on x-axis'
    )
    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help='Full path to checkpoint.pt file')
    parser.add_argument('--save_path', type=str, default='replot.png',
                        help='Path to save the output plot (default: replot.png)')
    parser.add_argument('--num_steps', type=int, default=None,
                        help='Total training steps the run was configured for. '
                             'If omitted, inferred from the last checkpoint step.')
    parser.add_argument('--grok_steps', nargs='+', default=None, metavar='TASK=STEP',
                        help='Manually specify or override grokking steps, '
                             'e.g. --grok_steps sub=75050')
    args = parser.parse_args()

    grok_steps_override = parse_grok_steps(args.grok_steps) if args.grok_steps else None

    replot_from_checkpoint(
        checkpoint_path=args.checkpoint_path,
        save_path=args.save_path,
        num_steps_override=args.num_steps,
        grok_steps_override=grok_steps_override,
    )
