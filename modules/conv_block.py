import torch 
import torch.nn as nn 

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):

        super().__init__()

        if mid_channels is None:
            mid_channels = out_channels

        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=5, stride=1, padding=2, bias= False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.gelu = nn.GELU()

        self.dwconv1 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=1, padding=1, groups=mid_channels, bias=False)
        self.pwconv1 = nn.Conv2d(mid_channels, mid_channels, kernel_size=1, stride=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)

        self.dwconv2 = nn.Conv2d(mid_channels, out_channels, kernel_size=3, stride=2, padding=1, groups=mid_channels, bias=False)
        self.pwconv2 = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        # 1. 5x5 Conv -> BN -> GeLU
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.gelu(x)

        # 2. 3x3 Depthwise Conv (stride=1)
        x = self.dwconv1(x)
        x = self.pwconv1(x)
        x = self.bn2(x)
        x = self.gelu(x)

        # 3. 3x3 Depthwise Conv (stride=2) -> Downsampling H/2, W/2
        x = self.dwconv2(x)
        x = self.pwconv2(x)
        x = self.bn3(x)
        x = self.gelu(x)

        return x