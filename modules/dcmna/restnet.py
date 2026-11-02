import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    """Two 3x3 conv-BN-ReLU layers, plus a 1x1 conv-BN-ReLU skip path."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()

        # Main path: two 3x3 convs, each followed by BN/ReLU
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3,
                      stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        # Skip path: 1x1 conv with BN/ReLU
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1,
                      stride=stride, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.main(x) + self.shortcut(x)


class ResNetModule(nn.Module):
    """Two cascaded ResNet blocks."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.block1 = BasicBlock(in_channels, out_channels, stride=stride)
        self.block2 = BasicBlock(out_channels, out_channels, stride=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        return x


if __name__ == "__main__":
    m = ResNetModule(64, 128, stride=2)
    print(m(torch.randn(2, 64, 32, 32)).shape)  # torch.Size([2, 128, 16, 16])