import torch 
import torch.nn as nn

class PositionEmbedding(nn.Module):
    def __init__(self, embed_dim=96, img_size=256, patch_size=4):
        super().__init__()
        num_patches = (img_size // patch_size) ** 2
        self.position_embeddings = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        nn.init.trunc_normal_(self.position_embeddings, std=0.02)

    def forward(self, x):
        if x.dim() == 4:
            B, H, W, C = x.shape
            x = x.reshape(B, H * W, C)
            x = x + self.position_embeddings[:, : H * W, :]
            x = x.reshape(B, H, W, C)
        else:
            x = x + self.position_embeddings[:, : x.shape[1], :]
        return x

