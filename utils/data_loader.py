# Data Loading and Preprocessing

import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, List, Dict, Optional
from sklearn.model_selection import train_test_split
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
from torch.utils.data import Dataset, DataLoader
import json

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import *

class BRISCDatasetInfo:

    """Display BRISC Dataset infrmation"""

    def __init__(self, data_root: Path):
        self.data_root = Path(data_root)
        self.classification_root = self.data_root / 'classification_task'
        self.segmentation_root = self.data_root / 'segmentation_task'

    def analyze_dataset(self):

        stats = {
            'classification: ': self._analyze_classification(),
            'segmentation: ': self._analyze_segmentation()
        }

        return stats

    def _analyze_classification(self):
        """
        Analyze classification dataset
        """
        stats = {'train': {},'test': {}}

        for split in ['train', 'test']:

            split_path = self.classification_root / split
            stats[split]['total'] = 0
            stats[split]['classes'] = {}

            for class_name in CLASS_NAMES:
                class_path = split_path / class_name

                if class_path.exists():
                    num_images = len(list(class_path.glob('*.jpg')))
                    stats[split]['classes'][class_name] = num_images
                    stats[split]['total'] += num_images

                else:
                    stats[split]['classes'][class_name] = 0

        return stats


    def _analyze_segmentation(self):
        """Analyze segmentation dataset"""

        stats = {'train': {}, 'test': {}}

        for split in ['train', 'test']:

            images_path = self.segmentation_root / split / 'images'
            masks_path = self.segmentation_root / split / 'masks'

            if images_path.exists():
                num_images = len(list(images_path.glob('*.jpg')))
                stats[split]['num_images'] = num_images

            else:
                stats[split]['num_images'] = 0

            if masks_path.exists():
                num_masks = len(list(masks_path.glob('*.jpg')))
                stats[split]['num_marks'] = num_masks

            else:
                stats[split]['num_masks'] = 0

        return stats


    def print_summary(self):

        stats = self.analyze_dataset()

        print("Classification task: ")
        for split in ['train', 'test']:

            print(f"\n{split.upper()}:")
            print(f"  Total images: {stats['classification'][split]['total']}")
            print(f"  Class distribution:")
            for class_name, count in stats['classification'][split]['classes'].items():
                print(f"    - {class_name:12s}: {count:5d} images")
        
        print("\n SEGMENTATION TASK:")
        for split in ['train', 'test']:
            print(f"\n{split.upper()}:")
            print(f"  Images: {stats['segmentation'][split]['num_images']}")
            print(f"  Masks:  {stats['segmentation'][split]['num_masks']}")



  