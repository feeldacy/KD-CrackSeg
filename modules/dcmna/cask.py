"""Hybrid attention mechanism module CASK (paper: DCMA-Net, Sec. 3.5).

CASK sits in the bottleneck between the two encoder channels and the decoder.
It receives three inputs:

    Fch1 : skip features from encoder channel 1 (cropped/sliced image branch)
    Fch2 : skip features from encoder channel 2 (whole image branch)
    Fu   : decoder feature, already upsampled to the current resolution

Steps (paper equations):

    Eq. (3)  FIn   = ReLU( f1x1( ReLU(Fch1 + Fch2) ) + f1x1(Fu) )
    Eq. (4)  Fmix1 = Sum_i( f_ixi(FIn) ),        i in {3, 5, 7}   (SKNet branch)
    Eq. (5)  a_i   = Softmax( fc2( fc1( PGAP(Fmix1) ) ) ), i in {3, 5, 7}
    Eq. (6)  Fout1 = Sum_i( F_i * a_i )                          (SKNet output)
    Eq. (7)  Fmix2 = dHS( BN( f1x1( Concat( PH_Avg(FIn), PW_Avg(FIn) ) ) ) )
    Eq. (8)  FH, FW = Split(Fmix2)
    Eq. (9)  a_j   = sigma( f1x1( F_j ) ),        j in {H, W}    (CAM branch)
    Eq. (10) Fout2 = FIn * a_H * a_W                             (CAM output)
    Eq. (11) FOut  = ReLU( f1x1( Concat(Fout1, Fout2) ) )
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CASK(nn.Module):
    """Coordinate Attention (CAM) + Selective Kernel (SKNet) hybrid attention.

    Args:
        channels: number of channels of ``Fch1``/``Fch2``/``Fu`` (must match).
        out_channels: channels of the produced feature map. Defaults to
            ``channels``.
        reduction: channel reduction ``r`` of the coordinate attention.
        kernel_sizes: convolution kernel sizes of the SKNet branches.
        cam_reduction: separate reduction for the SKNet fully connected layers.
    """

    def __init__(
        self,
        channels: int,
        out_channels: int = None,
        reduction: int = 16,
        kernel_sizes: tuple = (3, 5, 7),
        cam_reduction: int = 16,
    ) -> None:
        super().__init__()
        out_channels = out_channels or channels

        self.channels = channels
        self.kernel_sizes = kernel_sizes
        num_k = len(kernel_sizes)

        # --- Eq. (3): merge the two encoder skips and the decoder feature -----
        self.merge_encoder = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        self.merge_decoder = nn.Conv2d(channels, channels, kernel_size=1, bias=True)

        # --- SKNet branch (Eq. 4-6) -----------------------------------------
        self.sk_convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        channels,
                        channels,
                        kernel_size=k,
                        padding=k // 2,
                        bias=False,
                    ),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(inplace=True),
                )
                for k in kernel_sizes
            ]
        )

        sk_hidden = max(channels // cam_reduction, 1)
        self.sk_fc1 = nn.Sequential(
            nn.Linear(channels, sk_hidden, bias=False),
            nn.BatchNorm1d(sk_hidden),
            nn.ReLU(inplace=True),
        )
        # One channel-attention map per kernel size (Eq. 5).
        self.sk_fc2 = nn.Linear(sk_hidden, channels * num_k, bias=False)

        # --- Coordinate attention branch (Eq. 7-10) --------------------------
        cam_hidden = max(channels // reduction, 1)
        self.cam_conv = nn.Conv2d(channels, cam_hidden, kernel_size=1, bias=False)
        self.cam_bn = nn.BatchNorm2d(cam_hidden)
        self.cam_conv_h = nn.Conv2d(cam_hidden, channels, kernel_size=1, bias=True)
        self.cam_conv_w = nn.Conv2d(cam_hidden, channels, kernel_size=1, bias=True)

        # --- Eq. (11): fuse the two attention outputs ------------------------
        self.fusion = nn.Sequential(
            nn.Conv2d(2 * channels, out_channels, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
        )

    # ------------------------------------------------------------------ utils
    def _sknet(self, x: torch.Tensor) -> torch.Tensor:
        """SKNet selective-kernel attention: Eq. (4) - Eq. (6)."""
        features = [conv(x) for conv in self.sk_convs]

        # Eq. (4): the different-scale features are summed for channel pooling.
        fmix1 = torch.stack(features, dim=0).sum(dim=0)

        # Eq. (5): channel attention coefficients for each kernel size.
        gap = F.adaptive_avg_pool2d(fmix1, 1).flatten(1)   # (B, C)
        weights = self.sk_fc2(self.sk_fc1(gap))            # (B, K*C)
        weights = weights.view(weights.size(0), len(features), self.channels)
        weights = F.softmax(weights, dim=1)                # softmax over scales

        # Eq. (6): scale each branch feature with its channel attention map.
        out = 0
        for idx, feat in enumerate(features):
            out = out + feat * weights[:, idx].view(-1, self.channels, 1, 1)
        return out

    def _coordinate_attention(self, x: torch.Tensor) -> torch.Tensor:
        """Coordinate attention: Eq. (7) - Eq. (10)."""
        b, _, h, w = x.shape

        # Eq. (7): 1-D global pooling along the vertical and horizontal axes.
        f_h = F.adaptive_avg_pool2d(x, (h, 1))   # (B, C, H, 1)
        f_w = F.adaptive_avg_pool2d(x, (1, w))   # (B, C, 1, W)
        # Concatenate along the spatial axis -> (B, C, H + W, 1)
        fmix2 = torch.cat((f_h, f_w.permute(0, 1, 3, 2)), dim=2)

        fmix2 = self.cam_conv(fmix2)
        fmix2 = self.cam_bn(fmix2)
        fmix2 = F.hardswish(fmix2)

        # Eq. (8): split back into the vertical and horizontal attention maps.
        f_h, f_w = torch.split(fmix2, [h, w], dim=2)

        # Eq. (9): restore the channel dimension and normalise with sigmoid.
        alpha_h = torch.sigmoid(self.cam_conv_h(f_h))   # (B, C, H, 1)
        alpha_w = torch.sigmoid(self.cam_conv_w(f_w))   # (B, C, W, 1)
        alpha_w = alpha_w.permute(0, 1, 3, 2)            # (B, C, 1, W)

        # Eq. (10): modulate the input feature with both attention maps.
        return x * alpha_h * alpha_w

    # ---------------------------------------------------------------- forward
    def forward(
        self,
        fch1: torch.Tensor,
        fch2: torch.Tensor,
        fu: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the CASK feature map.

        Args:
            fch1: encoder channel-1 features, shape (B, C, H, W).
            fch2: encoder channel-2 features, shape (B, C, H, W).
            fu: upsampled decoder features, shape (B, C, H, W).

        Returns:
            Fused attention feature map, shape (B, out_channels, H, W).
        """
        if fch1.shape[-2:] != fch2.shape[-2:]:
            fch2 = F.interpolate(
                fch2, size=fch1.shape[-2:], mode="bilinear", align_corners=False
            )
        if fch1.shape[-2:] != fu.shape[-2:]:
            fu = F.interpolate(
                fu, size=fch1.shape[-2:], mode="bilinear", align_corners=False
            )

        # Eq. (3): the two encoder skips are added and activated, then merged
        # with the upsampled decoder feature.
        f_in = torch.relu(fch1 + fch2)
        f_in = torch.relu(self.merge_encoder(f_in) + self.merge_decoder(fu))

        # Parallel attention channels.
        fout1 = self._sknet(f_in)          # SKNet  -> channel attention
        fout2 = self._coordinate_attention(f_in)  # CAM -> spatial attention

        # Eq. (11): concatenate and fuse.
        return self.fusion(torch.cat((fout1, fout2), dim=1))


if __name__ == "__main__":
    module = CASK(channels=64, out_channels=64)
    x1 = torch.randn(2, 64, 32, 32)
    x2 = torch.randn(2, 64, 32, 32)
    xu = torch.randn(2, 64, 32, 32)
    out = module(x1, x2, xu)
    print(out.shape)  # torch.Size([2, 64, 32, 32])
    print("params (M):", sum(p.numel() for p in module.parameters()) / 1e6)
