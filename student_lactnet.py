import torch
import torch.nn as nn
import torch.nn.functional as F

from modules import patch_embedding, position_embedding, swin_transformer, duc, conv_block, affm, lam, fusion_heads

class LACTNet(nn.Module):
    def __init__(self, in_ch=3, num_classes=1):
        super().__init__()

        
        # ---------- ConvBlock branch (middle), each one downsamples x2 ----------
        self.conv1 = conv_block.ConvBlock(in_ch, 32)
        self.conv2 = conv_block.ConvBlock(32, 64)
        self.conv3 = conv_block.ConvBlock(64, 128)
        self.conv4 = conv_block.ConvBlock(128, 256)
        self.conv5 = conv_block.ConvBlock(256, 512)

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
    
        # ConvBlock branch
        c1 = self.conv1(x)                         # (B, 32,  128, 128)
        c2 = self.conv2(c1)                        # (B, 64,   64,  64)
        c3 = self.conv3(c2)                        # (B, 128,  32,  32)
        c4 = self.conv4(c3)                        # (B, 256,  16,  16)
        c5 = self.conv5(c4)                        # (B, 512,   8,   8)

        # Decoder
        d1 = self.d1(c5, c4)
        d2 = self.d2(d1, c3)
        d2, l3 = self.lam3(d2)
        d3 = self.d3(d2, c2)
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