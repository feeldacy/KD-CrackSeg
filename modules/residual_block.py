import torch
import torch.nn as nn


class SEBlock(nn.Module):
    """Returns channel attention weights (b, c, 1, 1), NOT the scaled features."""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.glb_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        w = self.glb_avg_pool(x)
        w = self.fc(w)
        return w


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=16):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, dilation=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=2, dilation=2, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(out_channels, out_channels, 3, padding=5, dilation=5, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.se_block = SEBlock(out_channels, reduction)

        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        self.final_relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))        # d=5 output (branch point)

        attention = self.se_block(out)         # SE -> weights
        out = out * attention                  # ⊗

        out = out + identity                   # ⊕
        return self.final_relu(out)