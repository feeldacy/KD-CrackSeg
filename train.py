import argparse, random, os
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
from data.registry import get_pairs
from data.registry import SegDataset
from teacher_lactnet import LACTNet

# ---------- 1. setup ----------
p = argparse.ArgumentParser()
p.add_argument("--dataset", required=True)
p.add_argument("--data_root", default=None)
p.add_argument("--epochs", type=int, default=100)
p.add_argument("--bs", type=int, default=8)
p.add_argument("--lr", type=float, default=1e-4)
p.add_argument("--out", default="runs/exp1")
args = p.parse_args()
os.makedirs(args.out, exist_ok=True)

random.seed(0); np.random.seed(0); torch.manual_seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------- 2. data ----------
pairs, thr = get_pairs(args.dataset, "train", data_root=args.data_root)
train_loader = DataLoader(SegDataset(pairs, thr, 256, train=True),
                          batch_size=args.bs, shuffle=True, num_workers=8,
                          pin_memory=True, drop_last=True)
pairs, thr = get_pairs(args.dataset, "test", data_root=args.data_root)
val_loader = DataLoader(SegDataset(pairs, thr, 256, train=False),
                        batch_size=args.bs, num_workers=4)

# ---------- 3. model, loss, optimizer ----------
model = LACTNet().to(device)

def dice_loss(logits, target, eps=1.0):
    p = torch.sigmoid(logits).flatten(1)
    t = target.flatten(1)
    return 1 - ((2 * (p * t).sum(1) + eps) / (p.sum(1) + t.sum(1) + eps)).mean()

def seg_loss(logits, target):
    return F.binary_cross_entropy_with_logits(logits, target) + dice_loss(logits, target)

def make_boundary(mask, k=3):
    """boundary GT from the mask: dilation - erosion"""
    dil = F.max_pool2d(mask, k, 1, k // 2)
    ero = -F.max_pool2d(-mask, k, 1, k // 2)
    return (dil - ero).clamp(0, 1)

opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
scaler = torch.amp.GradScaler(enabled=(device == "cuda"))

# ---------- 4. validation ----------
@torch.no_grad()
def evaluate():
    model.eval()
    tp = fp = fn = 0
    for img, mask in val_loader:
        img, mask = img.to(device), mask.to(device)
        pred = torch.sigmoid(model(img)["pred_semantic"]) > 0.5
        m = mask > 0.5
        tp += (pred & m).sum().item()
        fp += (pred & ~m).sum().item()
        fn += (~pred & m).sum().item()
    prec = tp / (tp + fp + 1e-8); rec = tp / (tp + fn + 1e-8)
    return dict(iou=tp / (tp + fp + fn + 1e-8),
                f1=2 * prec * rec / (prec + rec + 1e-8),
                precision=prec, recall=rec)

# ---------- 5. training loop ----------
best_f1 = 0.0
for epoch in range(args.epochs):
    model.train()
    running = 0.0
    for img, mask in train_loader:
        img, mask = img.to(device), mask.to(device)
        bnd = make_boundary(mask)

        with torch.autocast(device_type="cuda", enabled=(device == "cuda")):
            out = model(img, gts=[mask, mask])
            loss = (seg_loss(out["pred_semantic"], mask)
                    + 0.5 * seg_loss(out["pred_boundary"], bnd)
                    + 0.5 * out["supervision_loss"])

        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update()
        running += loss.item()

    sched.step()
    m = evaluate()
    print(f"epoch {epoch+1}/{args.epochs} loss {running/len(train_loader):.4f} "
          f"IoU {m['iou']:.4f} F1 {m['f1']:.4f} P {m['precision']:.4f} R {m['recall']:.4f}")

    # ---------- 6. checkpoints ----------
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                "epoch": epoch}, f"{args.out}/last.pth")
    if m["f1"] > best_f1:
        best_f1 = m["f1"]
        torch.save(model.state_dict(), f"{args.out}/best.pth")