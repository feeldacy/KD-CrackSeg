import torch
import torch.nn as nn 

class PSU(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super(PSU, self).__init__()
        
        intermediate_channels = out_channels * (scale_factor ** 2)
        
        # Z = ReLU(BN(Conv_{3x3}(F_dec)))
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, intermediate_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(intermediate_channels),
            nn.ReLU(inplace=True)
        )
        
        # F_dec_hat = PixelShuffler_r(Z)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor=scale_factor)

    def forward(self, f_dec):
        # 1. Proses konvolusi, BatchNorm, dan ReLU menghasilkan Z
        z = self.conv_block(f_dec)
        
        # 2. Proses PixelShuffle mengubah tensor berdimensi [B, C*r^2, H, W] 
        #    menjadi [B, C, r*H, r*W]
        out = self.pixel_shuffle(z)
        
        return out
        