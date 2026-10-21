import torch
import torch.nn as nn
import torch.nn.functional as F

from modules import patch_embedding, position_embedding, swin_transformer, duc, conv_block, affm, lam, fusion_heads

class LACTNet(nn.Module):
    def __init__(self, in_ch=3, num_classes=1):
        super().__init__()

        # ---------- Swin branch (left) ----------
        self.patch_embed = patch_embedding.PatchEmbedding(in_ch, 64, patch_size=4)
        self.pos_embed = position_embedding.PositionEmbedding(embed_dim=64, img_size=256, patch_size=4)

        # Each Swin stage applies window attention and shifted-window attention.
        self.swin1 = swin_transformer.SwinBlock(64, (64, 64))
        self.swin2 = swin_transformer.SwinBlock(128, (32, 32))
        self.swin3 = swin_transformer.SwinBlock(256, (16, 16))
        self.swin4 = swin_transformer.SwinBlock(512, (8, 8))

        # Downsample between Swin stages: halve spatial, double channels
        self.swin_down1 = nn.Conv2d(64, 128, kernel_size=2, stride=2)
        self.swin_down2 = nn.Conv2d(128, 256, kernel_size=2, stride=2)
        self.swin_down3 = nn.Conv2d(256, 512, kernel_size=2, stride=2)

        # ---------- ConvBlock branch (middle), each one downsamples x2 ----------
        self.conv1 = conv_block.ConvBlock(in_ch, 32)
        self.conv2 = conv_block.ConvBlock(32, 64)
        self.conv3 = conv_block.ConvBlock(64, 128)
        self.conv4 = conv_block.ConvBlock(128, 256)
        self.conv5 = conv_block.ConvBlock(256, 512)

        # ---------- AFFM: fuse (swin_i, conv_i+1) ----------
        self.affm1 = affm.AFFM(64, 64)
        self.affm2 = affm.AFFM(128, 128)
        self.affm3 = affm.AFFM(256, 256)
        self.affm4 = affm.AFFM(512, 512)

        # ---------- Decoder (Upblocks D1..D4) ----------
        self.d1 = duc.DUC(512, 256, 256)
        self.d2 = duc.DUC(256, 128, 128)
        self.d3 = duc.DUC(128, 64, 64)
        self.d4 = duc.DUC(64, 32, 32)

        # ---------- LAMs ----------
        self.lam3 = lam.LAM(128)
        self.lam2 = lam.LAM(64)

        # ---------- final head: 128 -> 256, then 1x1 conv ----------
        self.main_sem = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(32, num_classes, 1))
        self.main_bnd = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(32, 1, 1))

        # side heads (RM-1/8, RM-1/16, RM-1/32)
        self.side8  = fusion_heads.SideHead(128, num_classes)
        self.side16 = fusion_heads.SideHead(256, num_classes)
        self.side32 = fusion_heads.SideHead(512, num_classes)

        # fusion heads (main + 3 side maps are concatenated)
        self.sem_fusion = fusion_heads.FusionHead(4 * num_classes, num_classes)
        self.bnd_fusion = fusion_heads.FusionHead(4, 1)

    def forward(self, x, gts=None):
        # Swin branch — PatchEmbed/PosEmbed output BHWC, permute to BCHW for SwinTransformerBlock
        s = self.patch_embed(x)                    # (B, 64, 64, 64) BHWC
        s = self.pos_embed(s)                      # (B, 64, 64, 64) BHWC
        s = s.permute(0, 3, 1, 2)                  # (B, 64, 64, 64) BCHW

        s1 = self.swin1(s)                         # (B, 64,  64, 64)
        s2 = self.swin2(self.swin_down1(s1))       # (B, 128, 32, 32)
        s3 = self.swin3(self.swin_down2(s2))       # (B, 256, 16, 16)
        s4 = self.swin4(self.swin_down3(s3))       # (B, 512,  8,  8)

        # ConvBlock branch
        c1 = self.conv1(x)                         # (B, 32,  128, 128)
        c2 = self.conv2(c1)                        # (B, 64,   64,  64)
        c3 = self.conv3(c2)                        # (B, 128,  32,  32)
        c4 = self.conv4(c3)                        # (B, 256,  16,  16)
        c5 = self.conv5(c4)                        # (B, 512,   8,   8)

        # Fusion (both branches are BCHW now)
        f1 = self.affm1(s1, c2)
        f2 = self.affm2(s2, c3)
        f3 = self.affm3(s3, c4)
        f4 = self.affm4(s4, c5)

        # Decoder
        d1 = self.d1(f4, f3)
        d2 = self.d2(d1, f2)
        d2, l3 = self.lam3(d2)
        d3 = self.d3(d2, f1)
        d3, l2 = self.lam2(d3)
        d4 = self.d4(d3, c1)

        size = x.shape[-2:]
        up = lambda t: F.interpolate(t, size=size, mode="bilinear", align_corners=False)

        main_sem, main_bnd = self.main_sem(d4), self.main_bnd(d4)
        s8_sem,  s8_bnd  = self.side8(d2)     # d2 is already post-lam3 here
        s16_sem, s16_bnd = self.side16(d1)
        s32_sem, s32_bnd = self.side32(f4)

        pred_semantic = self.sem_fusion(torch.cat(
            [main_sem, up(s8_sem), up(s16_sem), up(s32_sem)], dim=1))
        pred_boundary = self.bnd_fusion(torch.cat(
            [main_bnd, up(s8_bnd), up(s16_bnd), up(s32_bnd)], dim=1))

        supervision_loss = 0.0
        if gts is not None and len(gts) >= 2:
            aux_logits = [l2, l3]
            aux_targets = []
            for target, logit in zip(gts[:2], aux_logits):
                target = target.to(logit.device, dtype=logit.dtype)
                if target.shape[-2:] != logit.shape[-2:]:
                    target = F.interpolate(target, size=logit.shape[-2:], mode="nearest")
                aux_targets.append(target)
            supervision_loss = (
                F.binary_cross_entropy_with_logits(aux_logits[0].squeeze(1), aux_targets[0].squeeze(1).float())
                + F.binary_cross_entropy_with_logits(aux_logits[1].squeeze(1), aux_targets[1].squeeze(1).float())
            ) / 2.0

        return {
            "pred_semantic": pred_semantic,
            "pred_boundary": pred_boundary,
            "side_semantic": [main_sem, up(s8_sem), up(s16_sem), up(s32_sem)],
            "side_boundary": [main_bnd, up(s8_bnd), up(s16_bnd), up(s32_bnd)],
            "supervision_loss": supervision_loss,
        }