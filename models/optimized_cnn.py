import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dropout=0.2):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout2d(p=dropout)
        self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False) if in_channels != out_channels or stride != 1 else nn.Identity()

    def forward(self, x):
        residual = self.residual(x)
        x = self.conv(x)
        x = self.bn(x)
        x = x + residual
        x = self.act(x)
        x = self.drop(x)
        return x


class SpectrogramResNet(nn.Module):
    def __init__(self, num_classes: int = 2, in_channels: int = 3):
        super(SpectrogramResNet, self).__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.layer1 = nn.Sequential(
            ConvBlock(64, 128, dropout=0.2),
            ConvBlock(128, 128, dropout=0.2),
        )
        self.layer2 = nn.Sequential(
            ConvBlock(128, 192, stride=2, dropout=0.25),
            ConvBlock(192, 192, dropout=0.25),
        )
        self.layer3 = nn.Sequential(
            ConvBlock(192, 256, stride=2, dropout=0.3),
            ConvBlock(256, 256, dropout=0.3),
        )
        self.layer4 = nn.Sequential(
            ConvBlock(256, 384, stride=2, dropout=0.35),
            ConvBlock(384, 384, dropout=0.35),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(384, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x
