import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = pred.contiguous().view(-1)
        target = target.contiguous().view(-1)
        
        intersection = (pred * target).sum()
        dice = (2.0 * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)
        return 1.0 - dice

class CombinedLoss(nn.Module):
    def __init__(self, bce_weight=1.0, dice_weight=1.0):
        super().__init__()
        self.bce = nn.BCELoss()
        self.dice = DiceLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, pred, target):
        """
        - pred: Output probabilitas model (sudah melewati Sigmoid) [B, 1, H, W]
        - target: Ground Truth mask [B, 1, H, W]
        """
        loss_bce = self.bce(pred, target)
        loss_dice = self.dice(pred, target)
        
        total_loss = (self.bce_weight * loss_bce) + (self.dice_weight * loss_dice)
        return total_loss

class FocalLoss(nn.Module):
    """
    Focal Loss for binary segmentation.
    Downweights easy negative pixels and focuses training on hard crack pixels.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """
    def __init__(self, alpha=0.25, gamma=2.0, eps=1e-7):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps

    def forward(self, pred, target):
        pred = pred.clamp(self.eps, 1.0 - self.eps)
        # p_t: probability of true class
        p_t = pred * target + (1.0 - pred) * (1.0 - target)
        # alpha_t: weighting factor
        alpha_t = self.alpha * target + (1.0 - self.alpha) * (1.0 - target)
        
        focal_weight = alpha_t * torch.pow(1.0 - p_t, self.gamma)
        loss = -focal_weight * torch.log(p_t)
        return loss.mean()

class FocalDiceLoss(nn.Module):
    """
    Combined Focal Loss + Dice Loss.
    Ideal for extreme class imbalance like crack segmentation.
    """
    def __init__(self, focal_weight=1.0, dice_weight=1.0, alpha=0.25, gamma=2.0):
        super().__init__()
        self.focal = FocalLoss(alpha=alpha, gamma=gamma)
        self.dice = DiceLoss()
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight

    def forward(self, pred, target):
        loss_focal = self.focal(pred, target)
        loss_dice = self.dice(pred, target)
        return (self.focal_weight * loss_focal) + (self.dice_weight * loss_dice)

class LogitsLoss(nn.Module):
    """Takes raw logits, applies sigmoid, and runs the wrapped loss in fp32."""
    def __init__(self, loss):
        super().__init__()
        self.loss = loss

    def forward(self, logits, target):
        with torch.autocast(device_type="cuda", enabled=False):   # BCELoss is not autocast-safe
            prob = torch.sigmoid(logits.float())
            return self.loss(prob, target.float())
        
class TverskyLoss(nn.Module):
    """
    Tversky Loss for binary segmentation.
    Generalization of Dice Loss with tunable false positive/negative weighting.
    TL = 1 - (TP + smooth) / (TP + alpha * FP + beta * FN + smooth)
    """
    def __init__(self, alpha=0.5, beta=0.5, smooth=1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, pred, target):
        pred = pred.contiguous().view(-1)
        target = target.contiguous().view(-1)

        TP = (pred * target).sum()
        FP = ((1 - target) * pred).sum()
        FN = (target * (1 - pred)).sum()

        tversky_index = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)
        return 1.0 - tversky_index
    
class FocalTverskyLoss(nn.Module):
    """
    Combined Focal Loss + Tversky Loss.
    Ideal for extreme class imbalance like crack segmentation.
    """
    def __init__(self, focal_weight=1.0, tversky_weight=1.0, alpha=0.5, beta=0.5, gamma=2.0):
        super().__init__()
        self.focal = FocalLoss(alpha=alpha, gamma=gamma)
        self.tversky = TverskyLoss(alpha=alpha, beta=beta)
        self.focal_weight = focal_weight
        self.tversky_weight = tversky_weight

    def forward(self, pred, target):
        loss_focal = self.focal(pred, target)
        loss_tversky = self.tversky(pred, target)
        return (self.focal_weight * loss_focal) + (self.tversky_weight * loss_tversky)


LOSSES = {
    "dice_bce":   lambda **kw: LogitsLoss(CombinedLoss(**kw)),
    "dice_focal": lambda **kw: LogitsLoss(FocalDiceLoss(**kw)),
    "tversky_focal": lambda **kw: LogitsLoss(FocalTverskyLoss(**kw)),
}

def build_loss(name, **kwargs):
    if name not in LOSSES:
        raise KeyError(f"Unknown loss '{name}'. Available: {list(LOSSES)}")
    return LOSSES[name](**kwargs)