import matplotlib.pyplot as plt

def plot_progress(data, xtick_labels, title, save_path=None):
    """Plot progress of best output over rounds."""
    fig, ax = plt.subplots(figsize=(8, 5))
    x_vals = list(range(len(data)))
    ax.plot(x_vals, data, marker='o', linestyle='-', linewidth=2)
    ax.set_title(title)
    ax.set_xlabel('Round')
    ax.set_ylabel('Best Output')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(x_vals)
    ax.set_xticklabels(xtick_labels, rotation=45, ha='right')
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
