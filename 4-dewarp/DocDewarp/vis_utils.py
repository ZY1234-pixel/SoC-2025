# viz_utils.py
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

def save_feature_map(tensor, save_path, name, max_channels=16):
    """
    tensor: (1, C, H, W)
    """
    feat = tensor.detach().cpu().float().numpy()[0]  # (C, H, W)
    C, H, W = feat.shape

    feat = (feat - feat.min()) / (feat.max() - feat.min() + 1e-8)

    n = min(C, max_channels)
    rows = int(np.ceil(n / 4))
    fig, axes = plt.subplots(rows, 4, figsize=(16, 4 * rows))
    axes = axes.flat if hasattr(axes, 'flat') else [axes]

    for i in range(n):
        axes[i].imshow(feat[i], cmap='viridis')
        axes[i].axis('off')
        axes[i].set_title(f'ch{i}', fontsize=8)

    for j in range(n, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'{name}.png'), dpi=150)
    plt.close()


def save_spatial_map(tensor, save_path, name):
    """
    对 channel 取均值，画一张"注意力热力图"
    """
    feat = tensor.detach().cpu().float().mean(dim=1, keepdim=False)[0]  # (H, W)
    feat = (feat - feat.min()) / (feat.max() - feat.min() + 1e-8)

    plt.imshow(feat, cmap='hot')
    plt.axis('off')
    plt.savefig(os.path.join(save_path, f'{name}_spatial.png'), dpi=150)
    plt.close()