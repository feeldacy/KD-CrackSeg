import torch

class EvalMetrics:
    """Accumulates TP/FP/FN/TN over the whole dataset, plus per-image Dice/IoU."""
    def __init__(self, thr=0.5, eps=1e-7):
        self.thr, self.eps = thr, eps
        self.reset()

    def reset(self):
        self.tp = self.fp = self.fn = self.tn = 0.0
        self.dice_sum = self.iou_sum = 0.0
        self.n = 0

    @torch.no_grad()
    def update(self, logits, target):
        pred = (torch.sigmoid(logits) > self.thr).flatten(1)   # (B, H*W) bool
        gt = (target > 0.5).flatten(1)

        tp = (pred & gt).sum(1).float()
        fp = (pred & ~gt).sum(1).float()
        fn = (~pred & gt).sum(1).float()
        tn = (~pred & ~gt).sum(1).float()

        # dataset-level counts
        self.tp += tp.sum().item(); self.fp += fp.sum().item()
        self.fn += fn.sum().item(); self.tn += tn.sum().item()

        # per-image scores, averaged later
        e = self.eps
        self.dice_sum += ((2 * tp + e) / (2 * tp + fp + fn + e)).sum().item()
        self.iou_sum += ((tp + e) / (tp + fp + fn + e)).sum().item()
        self.n += pred.shape[0]

    def compute(self):
        tp, fp, fn, tn, e = self.tp, self.fp, self.fn, self.tn, self.eps
        precision = tp / (tp + fp + e)
        recall = tp / (tp + fn + e)
        return {
            "acc": (tp + tn) / (tp + tn + fp + fn + e),
            "precision": precision,
            "recall": recall,
            "f1": 2 * precision * recall / (precision + recall + e),
            "iou": tp / (tp + fp + fn + e),            # dataset-level IoU
            "dice": self.dice_sum / max(self.n, 1),    # mean of per-image Dice
            "iou_img": self.iou_sum / max(self.n, 1),  # mean of per-image IoU
        }