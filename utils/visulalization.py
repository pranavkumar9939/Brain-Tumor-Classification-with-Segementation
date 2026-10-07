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


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    save_path: Optional[Path] = None,
    normalize: bool = False
):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Count' if not normalize else 'Proportion'}
    )
    
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.ylabel('True Label', fontsize=12, fontweight='bold')
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight')
        print(f"Saved confusion matrix to {save_path}")
    
    plt.show()


def plot_class_distribution(
    class_counts: dict,
    title: str = "Class Distribution",
    save_path: Optional[Path] = None
):
    plt.figure(figsize=(10, 6))
    
    classes = list(class_counts.keys())
    counts = list(class_counts.values())
    
    bars = plt.bar(classes, counts, color=sns.color_palette("husl", len(classes)))
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.,
            height,
            f'{int(height)}',
            ha='center',
            va='bottom',
            fontsize=11,
            fontweight='bold'
        )
    
    plt.xlabel('Class', fontsize=12, fontweight='bold')
    plt.ylabel('Count', fontsize=12, fontweight='bold')
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight')
        print(f"Saved class distribution to {save_path}")
    
    plt.show()


def plot_metrics_comparison(
    results: dict,
    metric_name: str,
    save_path: Optional[Path] = None
):
    plt.figure(figsize=(12, 6))
    
    experiments = list(results.keys())
    values = [results[exp][metric_name] for exp in experiments]
    
    bars = plt.bar(experiments, values, color=sns.color_palette("Set2", len(experiments)))
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.,
            height,
            f'{height:.4f}',
            ha='center',
            va='bottom',
            fontsize=10,
            fontweight='bold'
        )
    
    plt.xlabel('Experiment / Model', fontsize=12, fontweight='bold')
    plt.ylabel(metric_name.replace('_', ' ').title(), fontsize=12, fontweight='bold')
    plt.title(f'{metric_name.replace("_", " ").title()} Comparison', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight')
        print(f"Saved metrics comparison to {save_path}")
    
    plt.show()