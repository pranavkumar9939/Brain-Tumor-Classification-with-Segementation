import torch
import torch.nn as nn
import torchvision.models as models


class MobileNetClassifier(nn.Module):
    """
    MobileNetV2 classifier for resource constrained environments
    """

    def __init__(
        self,
        num_classes: int= 4,
        pretrained: bool = False,
        dropout: float = 0.5
    ):
        super(MobileNetClassifier, self).__init__()

        # Loading MobileNetV2
        self.mobilenet = models.mobilenet_v2(pretrained = pretrained)

        # MOdifying for grayscale input for first conv layer
        original_conv = self.mobilenet.features[0][0]
        self.mobilenet.features[0][0] = nn.Conv2d(
            1, 
            original_conv.out_channels,
            kernel_size = original_conv.kernel_size,
            stride = original_conv.stride,
            padding = original_conv.padding,
            bias = False
        )

        # Replacing classifier
        in_features = self.mobilenet.last_channel
        self.mobilenet.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace = True),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )


    def forward(self, x):
        return self.mobilenet(x)


class EfficientNetClassifier(nn.Module):
    """
    EfficientNet-B0 based classifier
    State-of-the-art accuracy  with good efficiency
    """
    def __init__(
        self,
        num_classes: int = 4,
        pretrained: bool = False,
        dropout: float = 0.5
    ):
        super(EfficientNetClassifier, self).__init__()

        # Loading EfficientNet-B0
        self.efficientnet = models.efficientnet_b0(pretrained = pretrained)

        # Modifying first layer for grayscale input
        original_conv = self.efficientnet.features[0][0]
        self.efficientnet.features[0][0] = nn.Conv2d(
            1,
            original_conv.out_channels,
            kernel_size= original_conv.kernel_size,
            stride = original_conv.stride,
            padding = original_conv.padding,
            bias = False
        )

        # Replacing classifier
        in_features = self.efficientnet.classifier[1].in_features
        self.efficientnet.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace = True),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.efficientnet(x)


class DenseNetClassifier(nn.Module):
    """
    DenseNet121-based classifier
    Excellent feature reuse with dense connection
    """

    def __init__(
        self,
        num_classes: int = 4,
        pretrained: bool = False,
        dropout: float = 0.5
    ):
        super(DenseNetClassifier, self).__init__()

        # Loading Densenet121
        self.densenet = models.densenet121(pretrained = pretrained)

        # modifying first conv layer for gray scale input
        original_conv = self.densenet.features.conv0
        self.densenet.features.conv0 = nn.Conv2d(
            1,
            original_conv.out_channels,
            kernel_size= original_conv.kernel_size,
            stride = original_conv.stride,
            padding= original_conv.padding,
            bias = False
        )

        # REplacing classifier
        in_features = self.densenet.classifier.in_features
        self.densenet.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(inplace= True),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(inplace= True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.densenet(x)


