import torch
import torch.nn as nn
import torch.nn.functional as F


class LAM(nn.Module):
    """Label-Aware Module.
    Returns the refined feature map U_bar and the raw prediction (logits) for supervision.
    """
    def __init__(self, in_channels, num_classes=1):
        super().__init__()
        # f1(.): 1x1 conv projection (the brown arrow in the figure)
        self.proj = nn.Conv2d(in_channels, num_classes, kernel_size=1)

    def forward(self, U):
        logits = self.proj(U)             # O1 (before sigmoid), (b, 1, h, w)
        mask = torch.sigmoid(logits)      # σ(O1)
        U_bar = U * mask                  # ⊗, broadcast over channels
        return U_bar, logits