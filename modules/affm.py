import torch
import torch.nn as nn

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        # Menggunakan Global Average Pooling, diikuti transformasi non-linear (1x1 conv + ReLU + 1x1 conv + Sigmoid)
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Global Avg Pool -> (B, C, 1, 1)[cite: 7]
        out = self.global_avg_pool(x)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out

class AFFM(nn.Module):
    def __init__(self, in_channels_transformer, in_channels_cnn, out_channels=None):
        super().__init__()
        if out_channels is None:
            out_channels = max(in_channels_transformer, in_channels_cnn)

        # 1. Pemrosesan Transformer Feature: 3x3 Depthwise Conv + 1x1 Pointwise Conv[cite: 6]
        self.trans_conv = nn.Sequential(
            nn.Conv2d(in_channels_transformer, in_channels_transformer, kernel_size=3, padding=1, groups=in_channels_transformer, bias=False),
            nn.Conv2d(in_channels_transformer, out_channels // 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels // 2),
            nn.ReLU(inplace=True)
        )

        # 2. Pemrosesan CNN Feature: 1x1 Conv untuk menyelaraskan dimensi[cite: 6]
        self.cnn_conv = nn.Sequential(
            nn.Conv2d(in_channels_cnn, out_channels // 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels // 2),
            nn.ReLU(inplace=True)
        )

        # 3. Setelah Concatenation: Depthwise-pointwise convolution pair[cite: 6]
        fused_channels = (out_channels // 2) + (out_channels // 2) # Total channel setelah concat

        self.fuse_conv = nn.Sequential(
            nn.Conv2d(fused_channels, fused_channels, kernel_size=3, padding=1, groups=fused_channels, bias=False),
            nn.Conv2d(fused_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # 4. Channel Attention Mechanism[cite: 7]
        self.channel_attention = ChannelAttention(out_channels)

    def forward(self, trans_feat, cnn_feat):
        """
        - trans_feat: Fitur dari encoder (Transformer / EfficientViT)[cite: 6]
        - cnn_feat: Fitur dari decoder/CNN[cite: 6]
        """
        # Proses cabang transformer & cnn
        t_out = self.trans_conv(trans_feat)
        c_out = self.cnn_conv(cnn_feat)
        
        # Concatenate sepanjang dimensi channel[cite: 6]
        concat_feat = torch.cat([t_out, c_out], dim=1)
        
        # Proses interaksi lanjutan dengan depthwise-pointwise conv[cite: 6]
        fused = self.fuse_conv(concat_feat)
        
        # Channel Attention & Element-wise Multiplication[cite: 7]
        att = self.channel_attention(fused)
        out = fused * att  # Persamaan (4): F_out = sigma(CA(F_f))odot F_f[cite: 7]
        
        return out