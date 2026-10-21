import torch
import torch.nn as nn
from torchvision.ops import DeformConv2d


class DeformBlock(nn.Module):
    """Modulated deformable conv (DCNv2-style) + BN + ReLU."""
    def __init__(self, ch, k=3):
        super().__init__()
        self.offset_mask = nn.Conv2d(ch, 3 * k * k, k, padding=k // 2)   # 2k² offsets + k² mask
        nn.init.zeros_(self.offset_mask.weight)
        nn.init.zeros_(self.offset_mask.bias)
        self.dcn = DeformConv2d(ch, ch, k, padding=k // 2)
        self.bn = nn.BatchNorm2d(ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        o1, o2, m = torch.chunk(self.offset_mask(x), 3, dim=1)
        offset = torch.cat([o1, o2], dim=1)
        mask = torch.sigmoid(m)
        return self.act(self.bn(self.dcn(x, offset, mask)))


class SideHead(nn.Module):
    """One per scale: predicts a semantic map and a boundary map (the RM blocks)."""
    def __init__(self, in_ch, num_classes):
        super().__init__()
        self.sem = nn.Conv2d(in_ch, num_classes, 1)
        self.bnd = nn.Conv2d(in_ch, 1, 1)

    def forward(self, x):
        return self.sem(x), self.bnd(x)


class FusionHead(nn.Module):
    """concat maps -> conv(16) -> DCN(16) -> 1x1 conv -> prediction."""
    def __init__(self, in_ch, out_ch, mid=16):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, mid, 3, padding=1),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
        )
        self.dcn = DeformBlock(mid)
        self.out = nn.Conv2d(mid, out_ch, 1)

    def forward(self, x):
        return self.out(self.dcn(self.conv(x)))