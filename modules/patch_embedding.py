import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=3, embed_dim=96, patch_size=4):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape

        assert H % self.patch_size == 0 and W % self.patch_size == 0, \
            f"Input image size ({H}*{W}) is not divisible by patch size ({self.patch_size})"

        x = self.proj(x)  # (B, embed_dim, H/patch_size, W/patch_size)
        x = x.permute(0, 2, 3, 1)  # (B, H/patch_size, W/patch_size, embed_dim)
        x = self.norm(x)  # (B, H/patch_size, W/patch_size, embed_dim)

        return x