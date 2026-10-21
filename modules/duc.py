import torch 
import torch.nn as nn 

from .residual_block import ResidualBlock
from .psu_module import PSU

class DUC(nn.Module):
    def __init__(self, dec_channels, skip_channels, out_channels, reduction=16):
        super().__init__()

        self.psu = PSU(dec_channels, out_channels)

        self.res_block = ResidualBlock(out_channels + skip_channels, out_channels, reduction)

    def forward(self, dec_feat, enc_skip):

        up = self.psu(dec_feat)
        x = torch.cat([up, enc_skip], dim=1)

        return self.res_block(x)


