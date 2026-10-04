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

        super(Down, self).__init__()

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

        if bilinear:
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
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

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


class UNetWithClassifier(nn.Module):
    """
    Unet with classification head attached to encoder 
    for joint training or seperate training experiment
    """

    def __init__(
        self, 
        in_channels: int = 1,
        num_classes: int = 4,
        base_filters: int = 64,
        dropout: float = 0.5
    ):

        super(UNetWithClassifier, self).__init__()

        self.unet = Unet(in_channels= in_channels, out_channels= 1, base_filters= base_filters)

        bottleneck_channels = base_filters * 8 # 512

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)), # 512*16*16 -> 512*1*1
            nn.Flatten(), # 512*1*1 -> 512
            nn.Linear(bottleneck_channels, 512), # feature extraction
            nn.ReLU(inplace= True), # Rgularization
            nn.Dropout(dropout),
            nn.Linear(512, 256), # Reduce dimension
            nn.ReLU(inplace = True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes) # final prediction # 4 classes
        )


    def forward(
        self,
        x, 
        return_seg_only: bool = False, 
        return_cls_only: bool = False
    ):
        """
        1. Run the Unet encoder manually to get the feature
        """

        # get encoder feature

        x1 = self.unet.inc(x)    # (1, 1, 256, 256) -> (1, 64, 256, 256)
        x2 = self.unet.down1(x1) # (1, 64, 256, 256) -> (1, 128, 128, 128)
        x3 = self.unet.down2(x2) # (1, 128, 128, 128) -> (1, 256, 64, 64)
        x4 = self.unet.down3(x3) # (1, 256, 64, 64) -> (1, 512, 32, 32)
        x5 = self.unet.down4(x4) # (1, 512, 32, 32) -> (1, 1024, 16, 16)

        # 2. Branch A: Classification
        # Classification from bottleneck

        cls_logits = self.classifier(x5) # predict class on bottleneck feature

        if return_cls_only:
            return cls_logits

        # 3. Branch B: Segmentation
        # Segmentation decoder

        x_up = self.unet.up1(x5, x4)
        x_up = self.unet.up2(x_up, x3)
        x_up = self.unet.up3(x_up, x2)
        x_up = self.unet.up4(x_up, x1)

        seg_logits = self.unet_outc(x_up)

        if return_seg_only:
            return seg_logits

        return seg_logits, cls_logits


def count_parameters(model):
    """
    Counts Parameters in the model

    1. .numel()-> use to calculate total number of elemets in a given tensor
    2. .parameters -> Method in nn.Module that returns an iterator over the trainable parameters of a model
    3. .requires_grad-> it is a boolean flag that determines whether a tensor should be tracked for gradient
        computaion during the backpropagation process
    """

    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "main":

    # test the U-Net

    model = Unet(in_channels= 1, out_channels= 1, base_filters= 64)
    x = torch.randn(2, 1, 256, 256)
    out = model(x)

    print(f"Input Shape: {x.shape}")
    print(f"Output Shape: {out.shape}")
    print(f"Parameters: {count_parameters(model):,}")

    # test UNet with classifier

    model_with_cls = UNetWithClassifier(in_channels= 1, num_classes= 4, base_filters= 64)
    seg_out, cls_out = model_with_cls(x)

    print("\nUnet with Classifier:")
    print(f"Segmentation output: {seg_out.shape}")
    print(f"Classification output: {cls_out.shape}")
    print(f"Parameters: {count_parameters(model_with_cls):,}")