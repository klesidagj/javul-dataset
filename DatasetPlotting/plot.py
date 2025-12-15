# --- FILE: plot.py ---
#!/usr/bin/env python3
"""
Plotting helpers for embeddings grid.
"""
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np

def scatter(ax, arr2d: np.ndarray, labels: List[str], title: str):
    unique = sorted(list(set(labels)))
    cmap = plt.cm.get_cmap('tab20', max(2, len(unique)))
    color_map = {u: i for i, u in enumerate(unique)}
    colors = [color_map[l] for l in labels]
    ax.scatter(arr2d[:, 0], arr2d[:, 1], c=colors, s=18, alpha=0.85)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])

def plot_grid(outputs: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, dict]],
              labels: List[str],
              out_path: str):
    # outputs: dict name -> (pca, tsne, umap, metrics)
    rows = list(outputs.keys())
    fig, axes = plt.subplots(len(rows), 3, figsize=(4 * 3.5, 3.5 * len(rows)))
    if axes.ndim == 1:
        axes = axes.reshape(1, -1)
    for i, name in enumerate(rows):
        p2, t2, u2, _ = outputs[name]
        scatter(axes[i, 0], p2, labels, f"{name} — PCA")
        scatter(axes[i, 1], t2, labels, f"{name} — t-SNE")
        scatter(axes[i, 2], u2, labels, f"{name} — UMAP")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)