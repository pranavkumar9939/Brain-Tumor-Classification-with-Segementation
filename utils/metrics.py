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


class DiceLoss(nn.Module):
    """
    Implementing dice loss for binary segmentation3
    """

    def __init__(self, smooth: float = 1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ):
        return 1 - dice_coefficent(pred, target, self.smooth)


class DiceBCELoss(nn.Module):
    """
    Combined Dice and BCE loss for better convergence
    """

    def __init__(
        self,
        dice_weight: float = 0.5,
        bce_weight: float = 0.5
    ):
        super(DiceBCELoss, self).__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.dice_loss = DiceLoss()
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ):
        pred_sigmoid = torch.sigmoid(pred)

        dice = self.dice_loss(pred_sigmoid, target)
        bce = self.bce_loss(pred, target)

        return self.dice_weight * dice + self.bce_weight * bce


class SegmentationMetrics:
    """
    Class to track and compute segmentation metrics
    """
    def __init__(self):
        self.reset()

    def reset(self):
        """
        Reset all metrics
        """
        self.dice_scores = []
        self.iou_scores = []
        self.pixel_accs = []

    def update(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ):
        """Update metrics with a batch"""

        pred = torch.sigmoid(pred)
        pred = (pred > 0.5).float()
        target = (target > 0.5).float()

        dice = dice_coefficent(pred, target).item()
        iou = iou_score(pred, target).item()
        pixel_acc = pixel_accuracy(pred, target).item()

        self.dice_scores.append(dice)
        self.iou_scores.append(iou)
        self.pixel_accs.append(pixel_acc)

    def get_metrics(self):

        return {
            'dice_coefficient': np.mean(self.dice_scores) if self.dice_scores else 0.0,
            'mIoU': np.mean(self.iou_scores) if self.iou_scores else 0.0,
            'pixel_accuracy': np.mean(self.pixel_accs) if self.pixel_accs else 0.0
        }


# Classification metrics 

class ClassificationMetrics:
    """
    Class to compute classification metrics
    """

    def __init__(self, num_classes: int = 4):
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        """Resetting all the metrics"""
        self.all_preds = []
        self.all_targets = []

    def update(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ):
        """Updating metrics with each batch"""

        # get predicted class
        if pred.dim() > 1:
            pred_classes = torch.argmax(pred, dim = 1)
        else:
            pred_classes = pred

        self.all_preds.extend(pred_classes.cpu().numpy())
        self.all_targets.extend(target.cpu().numpy())


    def get_metrics(self):
        """
        Calculate and return all metrics of classification
        """

        if not self.all_preds:
            return {
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0
            }

        preds = np.array(self.all_preds)
        targets = np.array(self.all_targets)

        metrics = {
            'accuracy': accuracy_score(targets, preds),
            'precision': precision_score(targets, preds, average='weighted', zero_division=0),
            'recall': recall_score(targets, preds, average='weighted', zero_division=0),
            'f1_score': f1_score(targets, preds, average='weighted', zero_division=0)
        }

        return metrics


    def get_confusion_matrix(self):

        if not self.all_preds:
            return np.zeros((self.num_classes, self.num_classes))

        return confusion_matrix(self.all_targets, self.all_preds)

    def get_classification_report(self, class_names: list = None):

        if not self.all_preds:
            return "No Predictions available"

        return classification_report(
            self.all_targets,
            self.all_preds,
            target_names= class_names,
            zero_division= 0
        )



