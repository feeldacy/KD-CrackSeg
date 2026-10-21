import torch 
import torch.nn as nn 
import torchvision



class DeformableConv2d(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 bias=False):

        super(DeformableConv2d, self).__init__()

        self.padding = padding

        self.offset_conv = nn.Conv2d(in_channels,
                                    2 * kernel_size * kernel_size,
                                    kernel_size=kernel_size,
                                    stride=stride,
                                    padding=self.padding,
                                    bias=True)

        nn.init.constant_(self.offset_conv.weight, 0.)
        nn.init.constant_(self.offset_conv.bias, 0.)

        self.modulator_conv = nn.Conv2d(in_channels,
                                     1 * kernel_size * kernel_size,
                                     kernel_size=kernel_size,
                                     stride=stride,
                                     padding=self.padding,
                                     bias=True)

        nn.init.constant_(self.modulator_conv.weight, 0.)
        nn.init.constant_(self.modulator_conv.bias, 0.)

        self.regular_conv = nn.Conv2d(in_channels=in_channels,
                                      out_channels=out_channels,
                                      kernel_size=kernel_size,
                                      stride=stride,
                                      padding=self.padding,
                                      bias=bias)

    def forward(self, x):
        h, w = x.shape[2:]
        max_offset = max(h, w)/4.

        offset = self.offset_conv(x).clamp(-max_offset, max_offset)
        modulator = 2. * torch.sigmoid(self.modulator_conv(x))

        x = torchvision.ops.deform_conv2d(input=x,
                                          offset=offset,
                                          weight=self.regular_conv.weight,
                                          bias=self.regular_conv.bias,
                                          padding=self.padding,
                                          mask=modulator
                                          )
        return x
    
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, use_deform=False):
        super().__init__()

        if mid_channels is None:
            mid_channels = out_channels

        self.use_deform = use_deform

        # 1. 5x5 Conv (Depthwise untuk hemat komputasi jika in_channels==mid_channels, atau standard)
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.gelu = nn.GELU()

        # 2. Fitur adaptif: Deformable Conv untuk stage dalam, Depthwise Conv untuk stage resolusi tinggi
        if self.use_deform:
            self.conv2 = DeformableConv2d(mid_channels, mid_channels, kernel_size=3, stride=1, padding=1, bias=False)
        else:
            self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=1, padding=1, groups=mid_channels, bias=False)
            
        self.pwconv1 = nn.Conv2d(mid_channels, mid_channels, kernel_size=1, stride=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)

        # 3. Downsampling (stride=2): Depthwise + Pointwise untuk efisiensi maksimal
        self.dwconv_down = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=2, padding=1, groups=mid_channels, bias=False)
        self.pwconv_down = nn.Conv2d(mid_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        # 1. 5x5 Conv -> BN -> GELU
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.gelu(x)

        # 2. Deformable / Depthwise -> Pointwise -> BN -> GELU
        x = self.conv2(x)
        x = self.pwconv1(x)
        x = self.bn2(x)
        x = self.gelu(x)

        # 3. Downsampling -> BN -> GELU
        x = self.dwconv_down(x)
        x = self.pwconv_down(x)
        x = self.bn3(x)
        x = self.gelu(x)

        return x