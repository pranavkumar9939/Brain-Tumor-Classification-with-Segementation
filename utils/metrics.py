import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from typing import Tuple, Dict

# Segmentation metrics

def dice_coefficent(
    pred: torch.Tensor,
    target: torch.Tensor,
    smooth: float = 1e-6
):
    """
    Calculate Dice Coefficient for segmentation( F1 Score )

    pred: Prediction masks -> (B, 1, W, H)
    target: Ground Truth Masks -> (B, 1, W, H)
    smooth: Smoothing factor to avoid division by Zero
    """

    
    pred = pred.contiguous().view(-1)
    target = target.contiguous().view(-1)

    intersection = (pred * target).sum()

    dice = (2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth)

    return dice


def iou_score(
    pred: torch.Tensor,
    target: torch.Tensor,
    smooth: float = 1e-6
):
    """
    Calculate Intersection over Union (IoU)

    pred: Prediction masks -> (B, 1, W, H)
    target: Ground Truth Masks -> (B, 1, W, H)
    smooth: Smoothing factor to avoid division by Zero
    """
    pred = pred.contiguous().view(-1)
    target = target.contiguous().view(-1)

    intersection = (pred * target).sum()
    union = (pred.sum() + target.sum() - intersection)

    iou = (intersection + smooth) / (union + smooth)

    return iou


def pixel_accuracy(
    pred: torch.Tensor,
    target: torch.Tensor
):
    """
    Calculate pixel wise accuracy

    pred: Predicted masks -> (B, 1, W, H)
    target: Ground Truth masks -> (B, 1, W, H)
    """
    pred = (pred > 0.5).float()
    target = (target > 0.5).float()

    correct = (pred == target).float().sum()
    total = torch.numel(pred) # total n0. of pixels

    return correct / total



