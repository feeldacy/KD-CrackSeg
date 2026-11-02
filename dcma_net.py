"""DCMA-Net: dual channel multi-scale feature attention network.

Reference: Y. Yan et al., "DCMA-Net: A dual channel multi-scale feature attention
network for crack image segmentation", Eng. Appl. Artif. Intell. 148 (2025) 110411.

The network keeps two encoder channels (paper Sec. 3.1, Fig. 2):

* channel 2 (global) encodes the whole 544x384 crack image;
* channel 1 (local)  encodes the 12 cropped 136x128 patches the whole image is
  divided into (4 columns x 3 rows). The crops are derived on the fly from the
  same input tensor, so the model still consumes a single full image.

Both channels use the same stage layout, so the local features of the 12 crops
can be stitched back into a full-image-sized map (3 rows x 4 columns) and matched
with the global features of the corresponding stage.

Decoder: the bottleneck aggregates both channels through ASPP, and every skip
level is refined by the hybrid attention module CASK before being forwarded to
the next decoder layer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.dcmna.aspp import ASPP
from modules.dcmna.cask import CASK
from modules.dcmna.mfe import MFE
from modules.dcmna.preprocessing import merge_patches, split_image
from modules.dcmna.restnet import ResNetModule


class DCMAEncoder(nn.Module):
    """Single encoder channel: stem conv + residual stages with MFE refinement.

    Args:
        in_channels: input channels (3 for RGB).
        widths: output channels of each stage.
        strides: downsampling stride of each stage.
        mfe_stages: indices of the stages that get an MFE refinement.
    """

    def __init__(
        self,
        in_channels: int = 3,
        widths: tuple = (20, 40, 80, 160),
        strides: tuple = (1, 2, 2, 2),
        mfe_stages: tuple = (1, 2),
    ) -> None:
        super().__init__()
        if len(widths) != len(strides):
            raise ValueError("widths and strides must have the same length")

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, widths[0], kernel_size=3, stride=strides[0],
                      padding=1, bias=False),
            nn.BatchNorm2d(widths[0]),
            nn.ReLU(inplace=True),
        )

        self.stages = nn.ModuleList()
        self.mfe = nn.ModuleList()
        for index in range(1, len(widths)):
            self.stages.append(
                ResNetModule(widths[index - 1], widths[index], stride=strides[index])
            )
            self.mfe.append(
                MFE(widths[index], widths[index])
                if index in mfe_stages
                else nn.Identity()
            )

    def forward(self, x: torch.Tensor) -> list:
        """Return one feature map per entry of ``widths``."""
        features = []
        out = self.stem(x)
        features.append(out)
        for stage, mfe in zip(self.stages, self.mfe):
            out = mfe(stage(out))
            features.append(out)
        return features


class DCMANet(nn.Module):
    """Dual channel multi-scale feature attention network.

    Args:
        in_channels: input image channels.
        num_classes: number of output channels (1 for binary crack masks).
        widths: encoder stage widths.
        n_w, n_h: number of crop columns / rows for the local channel.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1,
        widths: tuple = (20, 40, 80, 160),
        n_w: int = 4,
        n_h: int = 3,
    ) -> None:
        super().__init__()

        self.n_w = n_w
        self.n_h = n_h
        self.widths = widths

        # --- the two encoder channels -------------------------------------
        self.encoder_global = DCMAEncoder(in_channels, widths)
        self.encoder_local = DCMAEncoder(in_channels, widths)

        # --- bottleneck: ASPP over the concatenated deepest features -------
        bottleneck_in = 2 * widths[-1]
        self.aspp = ASPP(bottleneck_in, widths[-1])

        # --- decoder ------------------------------------------------------
        # One CASK per skip level; the decoder feature is projected onto the
        # encoder width of that level before entering CASK.
        self.cask = nn.ModuleList(
            [CASK(width, width) for width in widths]
        )
        # Projects the incoming decoder feature onto the width of the level.
        self.project = nn.ModuleList(
            [
                nn.Conv2d(widths[i + 1], widths[i], kernel_size=1)
                if i + 1 < len(widths)
                else nn.Identity()
                for i in range(len(widths))
            ]
        )
        self.refine = nn.ModuleList(
            [ResNetModule(widths[i], widths[i]) for i in range(len(widths))]
        )

        # --- output head --------------------------------------------------
        self.head = nn.Sequential(
            nn.Conv2d(widths[0], widths[0], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(widths[0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(widths[0], num_classes, kernel_size=1),
        )

    # ------------------------------------------------------------------ utils
    def _local_features(self, x: torch.Tensor) -> list:
        """Encode the 12 crops and stitch every level back to full-image size."""
        patches = split_image(x, self.n_w, self.n_h)          # (B, N, C, ph, pw)
        batch, count, channels, patch_h, patch_w = patches.shape
        flat = patches.reshape(batch * count, channels, patch_h, patch_w)

        staged = self.encoder_local(flat)

        stitched = []
        for feature in staged:
            _, c, h, w = feature.shape
            feature = feature.reshape(batch, count, c, h, w)
            stitched.append(merge_patches(feature, self.n_w, self.n_h))
        return stitched

    # ---------------------------------------------------------------- forward
    def forward(self, x: torch.Tensor, gts=None) -> dict:
        """Segment the crack mask of a full 384x544 crack image.

        Args:
            x: input image batch, shape (B, 3, 384, 544).
            gts: ignored, kept for API compatibility with the LACTNet models.

        Returns:
            Dict with ``pred_semantic`` (B, 1, 384, 544) logits and
            ``supervision_loss`` (0.0, the model has no auxiliary heads).
        """
        global_features = self.encoder_global(x)
        local_features = self._local_features(x)

        # --- bottleneck: ASPP over both channels' deepest features --------
        bottleneck = self.aspp(
            torch.cat((global_features[-1], local_features[-1]), dim=1)
        )

        # --- decoder: CASK fuses the two encoder skips with the upsampled
        #     decoder feature at every level ------------------------------
        decoder = bottleneck
        for index in reversed(range(len(self.widths))):
            if index != len(self.widths) - 1:
                decoder = F.interpolate(
                    decoder,
                    size=global_features[index].shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
                decoder = self.project[index](decoder)

            fused = self.cask[index](
                local_features[index], global_features[index], decoder
            )
            decoder = self.refine[index](fused)

        pred_semantic = self.head(decoder)

        return {
            "pred_semantic": pred_semantic,
            "pred_boundary": None,
            "side_semantic": [],
            "side_boundary": [],
            "supervision_loss": 0.0,
        }


if __name__ == "__main__":
    model = DCMANet()
    out = model(torch.randn(2, 3, 384, 544))
    print("output:", out["pred_semantic"].shape)
    print("params (M):", sum(p.numel() for p in model.parameters()) / 1e6)
