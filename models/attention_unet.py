import torch 
import torch.nn as nn
import torch.nn.functional as F
from .unet import DoubleConv, Down, OutConv, count_parameters

class AttentionGate(nn.Module):
    """
    Attention Gate module
    Highlights salient features from skip connections
    """
    
    def __init__(self, F_g: int, F_l: int, F_int: int):
        """
        Args:
            F_g: Number of feature maps (channels) in previous layer (gating signal)
            F_l: Number of feature maps in corresponding encoder layer (skip connection)
            F_int: Number of feature maps in intermediate layer
        """
        super(AttentionGate, self).__init__()
        
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, g, x):
        """
        Args:
            g: Gating signal from deeper layer (B, F_g, H_g, W_g)
            x: Skip connection from encoder (B, F_l, H_x, W_x)
        
        Returns:
            Attention-weighted features
        """
        # Transform gating signal
        g1 = self.W_g(g)
        
        # Transform skip connection
        x1 = self.W_x(x)
        
        # Combine and apply attention
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        
        # Apply attention weights to skip connection
        return x * psi