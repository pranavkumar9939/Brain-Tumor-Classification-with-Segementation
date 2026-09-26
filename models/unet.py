import torch
import torch.nn as nn
import torch.nn.functional as  F

class DoubleConv(nn.module):

    def __init__(self, in_channels: int, out_channels: int):

        super(DoubleConv, self).__init__()

        self.double_conv = nn.Sequential(

            # first conv layer
            nn.Conv2d(in_channels= in_channels, out_channels= out_channels,
                      kernel_size= 3, padding = 1, bias= False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace= True),

            # double Convolutional network

            nn.Conv2d(in_channels= out_channels, out_channels= out_channels,
                      kernel_size= 3, padding= 1, bias= False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace= True)
        )

        def forward(self, x):

            return self.double_conv(x)