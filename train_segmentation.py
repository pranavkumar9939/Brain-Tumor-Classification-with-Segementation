import torch 
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam, SGD, AdamW, RMSprop
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
import time

import sys
sys.path.append(str(Path(__file__).parent.parent))

from config import *
from models import Unet, AttentionUNet 
from utils.data_loader import BRISCClassificationDataset, get_train_transforms, get_val_transforms
from utils.metrics import DiceBCELoss, SegmentationMetrics
from utils.visulalization import plot_training_curves

class SegmentationTrainer:

    """training class for Segmentation models"""

    def __init__(
        self,
        model,
        device,
        train_loader,
        val_loader,
        test_loader,
        optimizer,
        criterion,
        scheduler = None,
        model_name = "model",
        save_dir = Models_Dir
    ):

        self.model = model.to(device)
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.model_name = model_name
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True , parents=True)

        self.history = {
            'train_loss':[],
            'val_loss':[],
            'train_dice':[],
            'val_dice':[],
            'train_miou':[],
            'val_miou':[],
            'train_pixel_acc':[],
            'val_pixel_acc':[]
        }

        self.best_val_loss = float('inf')
        self.patience_counter = 0

    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        metrics = SegmentationMetrics()
        total_loss = 0

        pbar = tqdm(self.train_loader , desc='Training')
        for images , masks in pbar:
            images = images.to(self.device)
            masks = masks.to(self.device)

            #Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, masks)

            #Backward pass
            loss.backward()
            self.optimizer.step()

            #Update metrics
            total_loss += loss.item()
        metrics.update(outputs.detach(), masks)

        #Update progress bar
        pbar.set_postfix({'loss':f'{loss.item():.4f}'})

        # Calculate epoch metrics
        avg_loss = total_loss / len(self.train_loader)
        epoch_metrics = metrics.get_metrics()

        return avg_loss , epoch_metrics

    def validate(self ,loader = None):
        """Validate model"""
        if loader is None:
            loader= self.val_loader

        self.model.eval()
        metrics = SegmentationMetrics()
        total_loss = 0

        with torch.no_grad():
            for images, masks in tqdm(loader , desc='Validation'):
                images = images.to(self.device)
                masks = masks.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs , masks)

                total_loss += loss.item()
                metrics.update(outputs, masks)

            avg_loss = total_loss/len(loader)
            epoch_metrics = metrics.get_metrics()

            return avg_loss , epoch_metrics
    def train(self, epochs , early_stopping_patience=EARLY_STOPPING_PATIENCE):
        """Complete training loop"""
        print(f"\nTraining {self.model_name}")
        print("="*80)

        start_time = time.time()

        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            print("-" * 80)

            # Train
            train_loss , train_metrics = self.train_epoch()

            #Validate
            val_loss , val_metrics = self.validate()

            #Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_dice'].append(train_metrics['dice_coefficient'])
            self.history['val_dice'].append(val_metrics['dice-coefficient'])
            self.history['train_miou'].append(train_metrics['mIoU'])
            self.history['val_miou'].append(val_metrics['mIoU'])
            self.history['train_pixel_acc'].append(train_metrics['pixel-accuracy'])
            self.history['val_pixel_acc'].append(val_metrics['pixel_accuracy'])

            #Print metrics
            print(f"Train Loss:{train_loss:.4f} | Val Loss: {val_loss:.4f}")
            print(f"Train Dice:{train_metrics['dice_coefficient']:.4f} | Val Dice: {val_metrics['dice-coefficient']:.4f}")
            print(f"Train mIoU:{train_metrics['mIoU']:.4f} | Val mIoU: {val_metrics['mIoU']:.4f}")
            print(f"Train Pixel Acc:{train_metrics['pixel-accuracy']:.4f} | Val Pixel Acc: {val_metrics['pixel_accuracy']:.4f}")

            #Learning rate scheduler
            if self.scheduler:
                self.scheduler.step(val_loss)
                print(f"Learning Rate : {self.optimizer.param_groups[0]['lr']:.6f}")

            #Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter=0
                self.save_checkpoint('best')
                print("Saved best model")
            else:
                self.patience_counter +=1

            #Early stopping
            if self.patience_counter >= early_stopping_patience:
                print(f"\nEarly stopping triggered after {epoch +1} epochs")
                break

        training_time = time.time() - start_time
        print(f"\nTraining completed in {training_time/60:.2f} minutes")

        #Test on best model
        self.load_checkpoint('best')
        test_loss , test_metrics = self.validate(self.test_loader)

        print("\n Test Set Results:")
        print(f"Test loss: {test_loss:.4f}")
        print(f"Test Dice: {test_metrics['dice_coefficient']:.4f}")
        print(f"Test mIoU: {test_metrics['mIoU']:.4f}")
        print(f"Test Pixel Acc: {test_metrics['pixel_accuracy']:.4f}")

        #Save final results
        results = {
            'model_name' : self.model_name,
            'training_time':training_time,
            'best_val_loss':self.best_val_loss,
            'test_metrics' : test_metrics,
            'history':self.history
        }

        results_path = self.save_dir/ f"{self.model_name}_results.json"
        with open(results_path,'w') as f:
            json.dump(results ,f, indent=4)

        return results

    def save_checkpoint(self , name='checkpoint'):
        """Save model checkpoint"""

        checkpoint_path = self.save_dir/ f"{self.model_name}_{name}.pth"
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history':self.history,
            'best_val_loss':self.best_val_loss
        }, checkpoint_path)

    def load_checkpoint(self , name='checkpoint'):
        """Load model checkpoint"""
        checkpoint_path = self.save_dir/ f"{self.model_name}_{name}.pth"
        checkpoint = torch.load(checkpoint_path , map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint.get('history', self.history)
        self.best_val_loss = checkpoint.get('best_val_loss', self.best_val_loss)


