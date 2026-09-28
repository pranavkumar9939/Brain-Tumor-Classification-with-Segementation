import torch
import torch.nn as nn
import torch.nn.functional as  F

class DoubleConv(nn.Module):

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


class Down(nn.Module):

    def __init__(self, in_channels: int, out_channels: int):

        super(down, self).__init__()

        self.maxpool_conv = nn.Sequential(
            # applying maxpool
            nn.MaxPool2d(2),

            # double conv

            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):

        return self.maxpool_conv(x)


class Up(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True):

        super(Up, self).__init__()

        if bilinear():
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners= True)
            self.conv = DoubleConv(in_channels, out_channels)

        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size= 2, stride= 2)
            self.conv = DoubleConv(in_channels, out_channels)


    def forward(self, x1, x2):
        """
        x1 = Input from previous layer
        x2 = Skip Connection from encoder
        """

        x1 = self.up(x1)

        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX //2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim= 1)

        return self.conv(x)

    # x1 comes from below (1, 1024, 16, 16) --> UpSampled to (1, 1024, 32, 32)
    # x2 comes from left (skip connection) (1, 512, 32, 32)
    # torch.cat combine them to (1, 1536, 32, 32)
    # Double conv shrinks them back into (1, 512, 32, 32)


class OutConv(nn.Module):

    def __init__(self, in_channels: int, out_channels: int):
        super(OutConv, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size = 1)

    def forward(self, x):

        return self.conv(x)


class Unet(nn.Module):

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_filters: int = 64,
        bilinear: bool = True
    ):

        super(Unet, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bilinear = bilinear

        # Encodng

        # First the Double conv
        self.inc = DoubleConv(in_channels, base_filters)

        # DownSampling of features
        self.down1 = Down(base_filters, base_filters * 2)
        self.down2 = Down(base_filters * 2, base_filters * 4)
        self.down3 = Down(base_filters * 4, base_filters * 8)

        factor = 2 if bilinear else 1

        self.down4 = Down(base_filters * 8, base_filters * 16 // factor)

        # Decoder UpSampling

        self.up1 = Up(base_filters * 16, base_filters * 8 // factor, bilinear)
        self.up2 = Up(base_filters * 8, base_filters * 4 // factor, bilinear)
        self.up3 = Up(base_filters * 4, base_filters * 2 // factor, bilinear)
        self.up4 = Up(base_filters * 2, base_filters, bilinear)

        # Output layer

        self.outc = OutConv(base_filters, out_channels)


    def forward(self, x):

        # Encoder

        x1 = self.inc(x) # shape (1, 64, 256, 256)
        x2 = self.down1(x1) # shape (1, 128, 128, 128)
        x3 = self.down2(x2) # shape (1, 256, 64, 64)
        x4 = self.down3(x3) # shape (1, 512, 32, 32)
        x5 = self.down4(x4) # shape (1, 1024, 16, 16) # Bottleneck

        # Decoder with skip connection

        x = self.up1(x5, x4) # UpSample x5(16->32), cat with x4 out:- (1, 512, 32, 32)
        x = self.up1(x, x3)
        x = self.up1(x, x2)
        x = self.up1(x, x1)

        # output

        logits = self.outc(x) # 1*1 Conv to squash 64 channels to 1 channel (mask)

        return logits # final shape(1, 1, 256, 256)


    def get_encoder_feature(self, x):

        """
        return bottleneck features
        """

        # Encoder

        x1 = self.inc(x) # shape (1, 64, 256, 256)
        x2 = self.down1(x1) 
        x3 = self.down2(x2) 
        x4 = self.down3(x3) 
        x5 = self.down4(x4)

        return x5