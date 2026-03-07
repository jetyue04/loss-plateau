'''
utils.py contains visualization utilities for grokking experiments.

This module provides functions to create plots showing the grokking 
phenomenon, including per-task accuracy curves with automatic grokking
detection markers plotted against training progress percentage.
'''

import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter


def _setup_log_xaxis(ax, x_min=0.01, x_max=100.0):
    """
    Configure the X-axis to use a log scale with percentage formatting.

    Sets major and minor tick positions appropriate for a 0.01%–100% range
    and formats major ticks as percentage strings (e.g. '1%', '10%').

    :param ax: Matplotlib Axes object to configure
    :param x_min: Minimum x-axis value (default: 0.01)
    :param x_max: Maximum x-axis value (default: 100.0)
    """
    ax.set_xscale('log')
    ax.set_xlim(x_min, x_max)

    major_ticks = [0.01, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100]
    minor_ticks = [0.02, 0.03, 0.04, 0.05, 0.07,
                   0.3, 0.4, 0.6, 0.7, 0.8, 0.9,
                   3, 4, 6, 7, 8, 9, 15, 25, 30, 40, 60, 70, 80, 90]

    ax.set_xticks(major_ticks)
    ax.set_xticks(minor_ticks, minor=True)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda val, _: f'{val:g}%'))
    ax.xaxis.set_minor_formatter(NullFormatter())


def plot_grokking(history, save_path='grokking_result.png'):
    """
    Create a visualization of the grokking phenomenon using training progress percentage.

    Generates a single-panel figure showing:
    - Training accuracy (orange dashed line)
    - Per-task validation accuracy (colored solid lines)
    - Vertical dashed lines marking grokking points (when validation exceeds 95%),
      with staggered labels showing the exact percentage and step number

    X-axis is log-scale training progress as a percentage of total steps,
    making the sudden generalisation transition visually prominent regardless
    of the absolute number of training steps.

    Each task is assigned a distinct color:
    - Division:       red
    - Addition:       blue
    - Subtraction:    green
    - Multiplication: purple

    :param history: Dictionary containing training history with keys:
        - 'steps':      List of step numbers where metrics were logged
        - 'train_acc':  Training accuracy (%) at each logged step
        - 'train_loss': Training loss at each logged step (unused in plot but kept for API compat)
        - 'val_stats':  Dict mapping task names to {'acc': [...], 'loss': [...]}
        - 'grok_steps': Dict mapping task names to the step where they first grokked (or None)
        - 'config':     Optional config dict; may contain 'num_steps' for x-axis scaling
    :param save_path: File path where the plot image will be saved
    """
    steps = history['steps']
    grok_steps = history.get('grok_steps', {})
    val_stats = history['val_stats']
    colors = {'div': 'red', 'add': 'blue', 'sub': 'green', 'mult': 'purple'}

    # Determine total training steps for percentage x-axis conversion.
    # Prefer an explicit value stored in config, then fall back to the last
    # recorded step (which may under-estimate if training was cut short).
    num_steps = history.get('config', {}).get('num_steps') or steps[-1]

    # Convert absolute step counts to training-progress percentages
    x_vals = [s / num_steps * 100 for s in steps]
    grok_pcts = {
        task: (gs / num_steps * 100 if gs is not None else None)
        for task, gs in grok_steps.items()
    }

    # Build Figure
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))

    # Training accuracy — orange dashed line
    ax.plot(x_vals, history['train_acc'],
            color='#E08040', linestyle='--', label='Train Acc', alpha=0.7, linewidth=1.5)

    # Styling helpers shared by all grokking annotations
    text_bbox = dict(boxstyle="round,pad=0.05", facecolor="white", edgecolor="none", alpha=0.8)
    # Use axis-transform so Y positions are in relative (0–1) coordinates,
    # keeping labels visible even when accuracy values vary widely
    trans = ax.get_xaxis_transform()

    # Stagger label heights to prevent overlapping vertical-line annotations
    grok_y_positions = [0.55, 0.35, 0.70, 0.20]
    grok_y_iter = iter(grok_y_positions)

    # Per-task validation accuracy curves and grokking markers
    for task, stats in val_stats.items():
        color = colors.get(task, 'orange')
        ax.plot(x_vals, stats['acc'],
                label=f'Val Acc ({task.upper()})', color=color, linewidth=2)

        grok_pct = grok_pcts.get(task)
        if grok_pct is not None:
            ax.axvline(x=grok_pct, color=color, linestyle='--', alpha=1.0, linewidth=1.5)
            y_pos = next(grok_y_iter, 0.5)
            ax.text(
                grok_pct, y_pos,
                f" {grok_pct:.1f}% (step {grok_steps[task]:,})",
                rotation=90, verticalalignment='center',
                color=color, fontweight='bold', fontsize=9,
                bbox=text_bbox, transform=trans
            )

    # Axes Labels, Title, and Formatting
    ax.set_xlabel(f'Training Progress (%) (Total Steps: {num_steps:,})')
    ax.set_ylabel('Accuracy (%)')

    task_names = [t.upper() for t in val_stats.keys()]
    if len(task_names) == 1:
        title = f'Grokking: Accuracy Per Task — {task_names[0]}'
    elif len(task_names) == 2:
        title = f'Grokking: Accuracy Per Task — {task_names[0]} & {task_names[1]}'
    else:
        title = f'Grokking: Accuracy Per Task — {", ".join(task_names[:-1])} & {task_names[-1]}'
    ax.set_title(title)

    _setup_log_xaxis(ax)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3, which='both')

    # Save Figure
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to {save_path}")