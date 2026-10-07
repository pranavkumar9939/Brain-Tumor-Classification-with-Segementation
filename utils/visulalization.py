import matplotlib.pyplot as plt 
import numpy as np
import torch
import cv2
from pathlib import Path
from typing import List, Tuple, Optional
import seaborn as sns
from matplotlib.gridspec import GridSpec

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import *

def visualize_sample(
    image: np.ndarray,
    mask: Optional[np.ndarray] = None,
    pred_mask: Optional[np.ndarray] = None,
    title: str = "",
    save_path: Optional[Path] = None
):
    num_images = 1 + (mask is not None) + (pred_mask is not None)

    fig, axes = plt.subplots(1, num_images, figsize=(5 * num_images, 5))
    if num_images == 1:
        axes = [axes]

    idx = 0

    # original image
    axes[idx].imshow(image, cmap='gray')
    axes[idx].set_title('Original Image')
    axes[idx].axis('off')
    idx += 1

    # Ground truth masks
    if mask is not None:
        axes[idx].imshow(mask, cmap='hot')
        axes[idx].set_title('Ground Truth Mask')
        axes[idx].axis('off')
        idx += 1

    # Predicted mask
    if pred_mask is not None:
        axes[idx].imshow(pred_mask, cmap='hot')
        axes[idx].set_title('Predicted Mask')
        axes[idx].axis('off')

    if title:
        fig.suptitle(title, fontsize=16, fontweight = 'bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi= FIGURE_DPI, bbox_inches='tight')
        print(f"Saved Visualization to {save_path}")

    plt.show()


def visualize_batch(
    images: torch.Tensor,
    masks: torch.Tensor,
    pred_masks: Optional[torch.Tensor] = None,
    num_samples: int = 4,
    save_path: Optional[Path] = None
):
    batch_size = min(images.shape[0], num_samples)
    num_cols = 3 if pred_masks is not None else 2
    
    fig, axes = plt.subplots(batch_size, num_cols, figsize=(5 * num_cols, 5 * batch_size))
    
    if batch_size == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(batch_size):
        # Convert to numpy and squeeze channel dimension
        img = images[i, 0].cpu().numpy()
        mask = masks[i, 0].cpu().numpy()
        
        # Original image
        axes[i, 0].imshow(img, cmap='gray')
        axes[i, 0].set_title(f'Sample {i+1}: Image')
        axes[i, 0].axis('off')
        
        # Ground truth mask
        axes[i, 1].imshow(mask, cmap='hot')
        axes[i, 1].set_title(f'Sample {i+1}: GT Mask')
        axes[i, 1].axis('off')
        
        # Predicted mask (if available)
        if pred_masks is not None:
            pred = pred_masks[i, 0].cpu().numpy()
            axes[i, 2].imshow(pred, cmap='hot')
            axes[i, 2].set_title(f'Sample {i+1}: Pred Mask')
            axes[i, 2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight')
        print(f"Saved batch visualization to {save_path}")
    
    plt.show()


def plot_training_curves(
    history: dict,
    metrics: List[str] = ['loss'],
    save_path: Optional[Path] = None
):
    num_metrics = len(metrics)
    fig, axes = plt.subplots(1, num_metrics, figsize=(7 * num_metrics, 5))
    
    if num_metrics == 1:
        axes = [axes]
    
    for idx, metric in enumerate(metrics):
        train_key = f'train_{metric}'
        val_key = f'val_{metric}'
        
        if train_key in history:
            axes[idx].plot(history[train_key], label='Train', linewidth=2)
        if val_key in history:
            axes[idx].plot(history[val_key], label='Validation', linewidth=2)
        
        axes[idx].set_xlabel('Epoch', fontsize=12)
        axes[idx].set_ylabel(metric.replace('_', ' ').title(), fontsize=12)
        axes[idx].set_title(f'{metric.replace("_", " ").title()} over Epochs', fontsize=14, fontweight='bold')
        axes[idx].legend(fontsize=10)
        axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight')
        print(f"Saved training curves to {save_path}")
    
    plt.show()
