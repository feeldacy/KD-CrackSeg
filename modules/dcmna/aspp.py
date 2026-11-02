import torch 
import torch.nn as nn
import torch.nn.functional as F

class ASPP(nn.Module):
    def __init__(
        self, 
        in_channels, 
        out_channels, 
        rates: tuple = (3, 5, 7, 9),
    ) -> None:
        super().__init__()

        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False), 
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Branch 2-5 
        self.atrous_brnches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        in_channels, 
                        out_channels, 
                        kernel_size=3, 
                        padding=rate, 
                        dilation=rate, 
                        bias=False
                    ),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                )
                for rate in rates
            ]
        )
        
        self.branch6 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)
        )
        
        self.fusion = nn.Sequential(
            nn.Conv2d((2+len(rates)) * out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[2:]
        
        outputs = [self.branch1(x)]
        outputs += [branch(x) for branch in self.atrous_brnches]
        
        gap = self.branch6(x)
        gap = F.interpolate(gap, size=(h, w), mode="bilinear", align_corners=False)
        outputs.append(gap)
        
        return self.fusion(torch.cat(outputs, dim=1))